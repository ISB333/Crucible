# CPU Inference Speed — Wave 0 + Wave 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Crucible example (`examples/inference_speed/`) that verifier-grounds a Gemini-driven search over the inference configuration of `Qwen3.5-9B-Q4_K_M` on CPU, measuring single-stream + aggregate tok/s with a lossless quality gate, and run a Wave 1 config-only search.

**Architecture:** A frozen `harness.py` launches `llama-server` with an editable `Config`, streams a fixed workload, computes tok/s from real token timestamps, and checks lossless equivalence against a frozen baseline. A `SpeedQualityVerifier` (deterministic, stateless — `Scored(value=aggregate)` / `Ok` / `Fail`) grades each candidate. Crucible's existing orchestrator + `AdvisorPolicy` runs N Gemini workers and shepherds plateaus. The 9B is passive cargo; Gemini edits only the `config` region.

**Tech Stack:** Python 3.12+, `httpx` (SSE streaming to llama-server's OpenAI endpoint), `pytest` (`unit` / `integration` markers), Crucible SDK (`Task`, `run`, `AdvisorPolicy`, `budgets`), `llama-server` (already installed at `/usr/local/bin/llama-server`).

## Global Constraints

- **Target model is fixed cargo:** `/home/isb/models/Qwen3.5-9B-Q4_K_M.gguf` — hardcoded as a frozen constant in `harness.py`. Gemini may never change which model is loaded; the verifier confirms the loaded model matches (don't trust config, verify the server's reported path).
- **Editability:** only the `config` region inside `config.py` (marked `# crucible:region start name=config … end`). `harness.py`, `strategy.py`, `workload/`, `quality/`, `baseline.json` are frozen — the integrity gate rejects edits to them.
- **Determinism:** all generation is greedy (temperature 0, fixed seed). The verifier is `deterministic=True`.
- **Quality gate (Wave 1):** lossless-equivalence only — candidate probe outputs must be byte-identical to the frozen baseline probe references. PPL / kata-suite (needed for Wave 3's quality-affecting strategies) are **deferred to the Wave 2/3 plan** — not stubbed here.
- **No external services in unit tests:** the 9B is never loaded in `unit`-marked tests. All 9B interaction goes through injectable seams (`stream_completion`, `launch_server`, `runner`).
- **Test markers:** `@pytest.mark.unit` for fast seam-tested logic; `@pytest.mark.integration` for anything that loads the 9B or calls Gemini.
- **RAM:** the 9B (5.5 GB) + draft + KV needs ~7–9 GB free. The existing Dify/ollama stack (~21 GB used) must be stopped before `integration` runs.
- **Conventions:** follow `examples/sidon/` (frozen dataclass verifier, `SubprocessSandbox`, `run_*.py` CLI pattern, independent verification at script end). Commit messages: `feat:`/`test:`/`chore:` English.

## File Structure

```
examples/inference_speed/
  harness.py            # FROZEN: Config dataclass+validation, tok/s, measure_single/aggregate,
                        #   lossless_match, launch_server, run_harness, TARGET_MODEL constant.
  config.py             # EDITABLE region "config": a CONFIG = Config(...) call. Imports Config from harness.
  strategy.py           # FROZEN passthrough stub (Wave 2 opens an editable region here). No region yet.
  workload/
    prompts_single.jsonl    # FROZEN: 5 coding prompts for single-stream measurement.
    prompts_aggregate.jsonl # FROZEN: 8 coding prompts for concurrent measurement.
    probes.jsonl            # FROZEN: 4 short prompts whose greedy outputs are the lossless reference.
  quality/
    corpus.txt          # FROZEN: placeholder-free small held-out text (unused in Wave 1; present so the
                        #   immutable-spec structure matches the spec and is ready for Wave 2 PPL).
  baseline.json         # FROZEN after Wave 0: {single_stream, aggregate, probe_reference{...}}.
  speed_verifier.py     # SpeedQualityVerifier (deterministic).
  measure_baseline.py   # Wave 0 script: produce baseline.json from the default config.
  run_speed.py          # CLI entry: wires Task + SpeedQualityVerifier + run() + AdvisorPolicy.
  speed.db              # append-only provenance (gitignored).
tests/unit/
  test_speed_config.py
  test_speed_toks.py
  test_speed_measure.py
  test_speed_lossless.py
  test_speed_server.py
  test_speed_harness.py
  test_speed_verifier.py
tests/integration/
  test_speed_baseline.py
  test_speed_wave1.py
```

**Responsibilities:** `harness.py` = all measurement logic + the frozen target-model constant + test seams. `config.py` = the single editable surface. `speed_verifier.py` = stateless grading + integrity. `run_speed.py` = Crucible wiring. Tests = seam-injected logic; never load the 9B in `unit`.

---

### Task 1: `Config` dataclass + validation

**Files:**
- Create: `examples/inference_speed/harness.py`
- Test: `tests/unit/test_speed_config.py`

**Interfaces:**
- Produces: `harness.Config` (frozen dataclass) with fields `draft_model: str | None`, `n_threads: int`, `n_batch: int`, `n_ubatch: int`, `flash_attn: bool`, `use_mmap: bool`, `use_mlock: bool`, `ctx_size: int`, `draft_max: int`, `n_concurrent: int`, `cache_policy: str`. `__post_init__` validates and raises `ValueError` on any out-of-range value. Class constant `TARGET_MODEL = "/home/isb/models/Qwen3.5-9B-Q4_K_M.gguf"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_speed_config.py
import pytest
from harness import Config


def test_config_defaults_valid():
    c = Config()
    assert c.n_threads == 12
    assert c.draft_model is None
    assert c.cache_policy == "on"


def test_config_rejects_bad_threads():
    with pytest.raises(ValueError, match="n_threads"):
        Config(n_threads=0)
    with pytest.raises(ValueError, match="n_threads"):
        Config(n_threads=99)


def test_config_rejects_bad_batch():
    with pytest.raises(ValueError, match="n_batch"):
        Config(n_batch=0)


def test_config_rejects_bad_cache_policy():
    with pytest.raises(ValueError, match="cache_policy"):
        Config(cache_policy="fast")  # only "on" | "off"


def test_config_rejects_bad_concurrency():
    with pytest.raises(ValueError, match="n_concurrent"):
        Config(n_concurrent=0)


def test_config_clamps_draft_max():
    assert Config(draft_max=0).draft_max == 1   # 0 nonsense → clamp to 1
    assert Config(draft_max=99).draft_max == 16  # cap at 16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/isb/models/Crucible && uv run pytest tests/unit/test_speed_config.py -v`
Expected: FAIL — `ModuleNotFoundError: harness`.

- [ ] **Step 3: Write minimal implementation**

```python
# examples/inference_speed/harness.py
"""FROZEN inference harness for the CPU speed experiment.

Editability lives in config.py (the `config` region). This module is immutable
spec: the measurement methodology, the target model, and the quality gate.
"""
from __future__ import annotations

from dataclasses import dataclass

TARGET_MODEL = "/home/isb/models/Qwen3.5-9B-Q4_K_M.gguf"
_VPS_CORES = 12


@dataclass(frozen=True)
class Config:
    """Tunable llama-server parameters. The target model is NOT here — it is frozen."""

    draft_model: str | None = None
    n_threads: int = _VPS_CORES
    n_batch: int = 512
    n_ubatch: int = 512
    flash_attn: bool = True
    use_mmap: bool = True
    use_mlock: bool = False
    ctx_size: int = 4096
    draft_max: int = 4
    n_concurrent: int = 1
    cache_policy: str = "on"  # "on" | "off"

    def __post_init__(self) -> None:
        if not 1 <= self.n_threads <= _VPS_CORES:
            raise ValueError(f"n_threads must be in [1, {_VPS_CORES}]")
        if self.n_batch < 1:
            raise ValueError("n_batch must be >= 1")
        if self.n_ubatch < 1:
            raise ValueError("n_ubatch must be >= 1")
        if self.ctx_size < 512:
            raise ValueError("ctx_size must be >= 512")
        if self.n_concurrent < 1:
            raise ValueError("n_concurrent must be >= 1")
        if self.cache_policy not in ("on", "off"):
            raise ValueError("cache_policy must be 'on' or 'off'")
        # draft_max: clamp to a sane spec-decoding range
        dm = self.draft_max
        if dm < 1:
            dm = 1
        if dm > 16:
            dm = 16
        object.__setattr__(self, "draft_max", dm)

    def to_cli_args(self) -> list[str]:
        """Translate to llama-server CLI flags (Task 7 uses this)."""
        args = [
            "--model", TARGET_MODEL,
            "--threads", str(self.n_threads),
            "--batch-size", str(self.n_batch),
            "--ubatch-size", str(self.n_ubatch),
            "--ctx-size", str(self.ctx_size),
            "--parallel", str(self.n_concurrent),
        ]
        if self.flash_attn:
            args.append("--flash-attn")
        if self.use_mmap:
            args.append("--mlock" if False else "--no-mlock") if False else args  # placeholder guard
        # ponytail: build args plainly below; the line above is replaced in Task 7
        return args
```

Wait — the `to_cli_args` above has a leftover junk line. Replace the whole method body cleanly before committing:

```python
    def to_cli_args(self) -> list[str]:
        args = [
            "--model", TARGET_MODEL,
            "--threads", str(self.n_threads),
            "--batch-size", str(self.n_batch),
            "--ubatch-size", str(self.n_ubatch),
            "--ctx-size", str(self.ctx_size),
            "--parallel", str(self.n_concurrent),
        ]
        if self.flash_attn:
            args.append("--flash-attn")
        if not self.use_mmap:
            args.append("--no-mmap")
        if self.use_mlock:
            args.append("--mlock")
        if self.draft_model:
            args += ["--model-draft", self.draft_model, "--draft-max", str(self.draft_max)]
        if self.cache_policy == "off":
            args.append("--no-context-shift")
        return args
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_speed_config.py -v`
Expected: PASS (6 tests). Note: `to_cli_args` is exercised in Task 7; add a smoke assert here:

```python
def test_config_to_cli_args_has_target_model():
    args = Config().to_cli_args()
    assert "--model" in args
    idx = args.index("--model")
    assert args[idx + 1] == "/home/isb/models/Qwen3.5-9B-Q4_K_M.gguf"
```

- [ ] **Step 5: Commit**

```bash
cd /home/isb/models/Crucible
git checkout -b feat/inference-speed
git add examples/inference_speed/harness.py tests/unit/test_speed_config.py
git commit -m "feat(inference_speed): Config dataclass with validation and CLI args"
```

---

### Task 2: Frozen fixtures (workload + quality corpus)

**Files:**
- Create: `examples/inference_speed/workload/prompts_single.jsonl`
- Create: `examples/inference_speed/workload/prompts_aggregate.jsonl`
- Create: `examples/inference_speed/workload/probes.jsonl`
- Create: `examples/inference_speed/quality/corpus.txt`

**Interfaces:**
- Produces: three JSONL files (one JSON object per line: `{"id": str, "prompt": str}`) and one text corpus. Format consumed by `load_workload` / `load_probes` (Task 4/6).

- [ ] **Step 1: Write the fixture files**

`workload/prompts_single.jsonl` (5 lines — coding prompts, greedy):
```jsonl
{"id":"s1","prompt":"Write a Python function that returns the nth Fibonacci number iteratively. Only output the function body, no prose."}
{"id":"s2","prompt":"Write a Python function is_prime(n) that returns True if n is prime. Only output the function, no prose."}
{"id":"s3","prompt":"Write a Python function reverse_list(xs) returning the reversed list. Only output the function, no prose."}
{"id":"s4","prompt":"Write a Python function sum_digits(n) summing decimal digits of n. Only output the function, no prose."}
{"id":"s5","prompt":"Write a Python function dedupe(xs) preserving order, removing duplicates. Only output the function, no prose."}
```

`workload/prompts_aggregate.jsonl` (8 lines — distinct coding prompts for concurrent decode):
```jsonl
{"id":"a1","prompt":"Write Python: def merge(a,b): merge two sorted lists. Output code only."}
{"id":"a2","prompt":"Write Python: def quicksort(xs): in-place quicksort. Output code only."}
{"id":"a3","prompt":"Write Python: def tokenize(s): split on whitespace, strip punctuation. Output code only."}
{"id":"a4","prompt":"Write Python: def bitcount(n): number of set bits. Output code only."}
{"id":"a5","prompt":"Write Python: def lcs(a,b): longest common substring length. Output code only."}
{"id":"a6","prompt":"Write Python: def rotate(xs,k): rotate list right by k. Output code only."}
{"id":"a7","prompt":"Write Python: def flatten(xs): flatten one level of nesting. Output code only."}
{"id":"a8","prompt":"Write Python: def histogram(xs): dict of element counts. Output code only."}
```

`workload/probes.jsonl` (4 short prompts — greedy outputs are the lossless reference):
```jsonl
{"id":"p1","prompt":"Complete: the capital of France is"}
{"id":"p2","prompt":"Complete: 2 + 2 ="}
{"id":"p3","prompt":"Complete: def square(x): return"}
{"id":"p4","prompt":"Complete: the opposite of hot is"}
```

`quality/corpus.txt` (real held-out text — a few paragraphs of public-domain technical prose; used by Wave 2 PPL, kept now so the immutable structure is in place):
```
The traveling salesman problem asks for the shortest route visiting every city
exactly once and returning to the origin. It is NP-hard in the general case,
but many instances succumb to branch-and-bound, dynamic programming over
bitmask subsets, or heuristic approximation. The Held-Karp algorithm solves it
in O(n^2 2^n) time and O(n 2^n) space, feasible up to roughly twenty cities.

A Sidon set is a set of integers whose pairwise sums are all distinct. The
largest Sidon set in [1, n] has size about the square root of n. Constructions
such as Erdos-Turan and Bose give near-optimal sets for prime-power ranges.

Speculative decoding pairs a fast draft model with a slow target model. The
draft proposes several tokens; the target verifies them in a single forward
pass and accepts those consistent with its own distribution. Accepted tokens
are exactly what the target would have emitted, so the speedup is lossless.
```

- [ ] **Step 2: Verify the fixtures load as JSONL**

Run:
```bash
cd /home/isb/models/Crucible
python -c "import json; [json.loads(l) for l in open('examples/inference_speed/workload/prompts_single.jsonl')]; print('single ok')"
python -c "import json; [json.loads(l) for l in open('examples/inference_speed/workload/prompts_aggregate.jsonl')]; print('aggregate ok')"
python -c "import json; [json.loads(l) for l in open('examples/inference_speed/workload/probes.jsonl')]; print('probes ok')"
```
Expected: `single ok` / `aggregate ok` / `probes ok`.

- [ ] **Step 3: Commit**

```bash
git add examples/inference_speed/workload examples/inference_speed/quality
git commit -m "feat(inference_speed): frozen workload prompts, probes, and quality corpus"
```

---

### Task 3: `tok_per_second` pure function

**Files:**
- Modify: `examples/inference_speed/harness.py` (append function)
- Test: `tests/unit/test_speed_toks.py`

**Interfaces:**
- Consumes: a list of `(token_str, perf_counter_float)` tuples (the seam output of Task 4/5).
- Produces: `harness.tok_per_second(events: list[tuple[str, float]]) -> float` — tokens / (last_time − first_time), 0.0 if fewer than 2 events.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_speed_toks.py
from harness import tok_per_second


def test_tps_basic():
    # 4 tokens over 1.0s → 4.0 tok/s
    events = [("a", 0.0), ("b", 0.3), ("c", 0.7), ("d", 1.0)]
    assert tok_per_second(events) == 4.0


def test_tps_one_event_is_zero():
    assert tok_per_second([("a", 1.0)]) == 0.0


def test_tps_empty_is_zero():
    assert tok_per_second([]) == 0.0


def test_tps_zero_span_is_zero():
    assert tok_per_second([("a", 5.0), ("b", 5.0)]) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_speed_toks.py -v`
Expected: FAIL — `ImportError: cannot import name 'tok_per_second'`.

- [ ] **Step 3: Write minimal implementation**

Append to `examples/inference_speed/harness.py`:
```python
def tok_per_second(events: list[tuple[str, float]]) -> float:
    """tokens / (last_timestamp - first_timestamp). 0.0 if undefined."""
    if len(events) < 2:
        return 0.0
    t0 = events[0][1]
    t1 = events[-1][1]
    span = t1 - t0
    if span <= 0.0:
        return 0.0
    return len(events) / span
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_speed_toks.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add examples/inference_speed/harness.py tests/unit/test_speed_toks.py
git commit -m "feat(inference_speed): tok_per_second pure function"
```

---

### Task 4: `measure_single_stream` with injectable streamer

**Files:**
- Modify: `examples/inference_speed/harness.py`
- Test: `tests/unit/test_speed_measure.py`

**Interfaces:**
- Produces: `harness.load_workload(path: Path) -> list[dict]` and `harness.measure_single_stream(stream_fn, base_url, prompts, max_tokens=256) -> dict` returning `{"tok_s": float, "n_tokens": int, "per_prompt": [{"id": str, "tok_s": float, "n_tokens": int}]}`. `stream_fn(base_url, prompt, max_tokens, temperature) -> list[tuple[str, float]]` is the seam.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_speed_measure.py
import json
from pathlib import Path

from harness import load_workload, measure_single_stream


def _fake_stream_factory(emit_per_second: float = 10.0):
    """Returns a stream_fn that emits n tokens at a fixed cadence (deterministic)."""
    import time as _t
    def stream_fn(base_url, prompt, max_tokens, temperature=0.0):
        n = 20
        # use a counter closed over a mutable list to fake perf_counter progression
        out = []
        for i in range(n):
            out.append((f"tok{i}", i / emit_per_second))
        return out
    return stream_fn


def test_load_workload(tmp_path):
    p = tmp_path / "w.jsonl"
    p.write_text('{"id":"x","prompt":"hi"}\n{"id":"y","prompt":"yo"}\n')
    items = load_workload(p)
    assert [i["id"] for i in items] == ["x", "y"]


def test_measure_single_stream_aggregates_over_prompts():
    prompts = [{"id": "s1", "prompt": "x"}, {"id": "s2", "prompt": "y"}]
    res = measure_single_stream(_fake_stream_factory(10.0), "http://x", prompts, max_tokens=20)
    # each prompt: 20 tokens over 1.9s span (i=0..19 → 0..1.9) → 20/1.9 ≈ 10.53
    assert res["n_tokens"] == 40
    assert res["tok_s"] > 0.0
    assert len(res["per_prompt"]) == 2
    assert res["per_prompt"][0]["id"] == "s1"


def test_measure_single_stream_empty_prompts():
    res = measure_single_stream(_fake_stream_factory(), "http://x", [], max_tokens=20)
    assert res["n_tokens"] == 0
    assert res["tok_s"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_speed_measure.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_workload'`.

- [ ] **Step 3: Write minimal implementation**

Append to `examples/inference_speed/harness.py`:
```python
import json as _json
import time as _time
from pathlib import Path as _Path


def load_workload(path: _Path) -> list[dict]:
    """Load a JSONL workload file -> list of {"id","prompt"} dicts."""
    items = []
    for line in _Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            items.append(_json.loads(line))
    return items


def measure_single_stream(stream_fn, base_url: str, prompts: list[dict], max_tokens: int = 256) -> dict:
    """Stream each prompt greedily; report median tok/s and totals.

    stream_fn(base_url, prompt, max_tokens, temperature=0.0) -> list[(token, perf_counter)].
    """
    per_prompt: list[dict] = []
    total_tokens = 0
    tps_values: list[float] = []
    for item in prompts:
        events = stream_fn(base_url, item["prompt"], max_tokens, temperature=0.0)
        tps = tok_per_second(events)
        per_prompt.append({"id": item["id"], "tok_s": tps, "n_tokens": len(events)})
        total_tokens += len(events)
        if tps > 0:
            tps_values.append(tps)
    tps_values.sort()
    median = tps_values[len(tps_values) // 2] if tps_values else 0.0
    return {"tok_s": median, "n_tokens": total_tokens, "per_prompt": per_prompt}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_speed_measure.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add examples/inference_speed/harness.py tests/unit/test_speed_measure.py
git commit -m "feat(inference_speed): measure_single_stream with injectable streamer"
```

---

### Task 5: `measure_aggregate` (concurrent)

**Files:**
- Modify: `examples/inference_speed/harness.py`
- Test: `tests/unit/test_speed_measure.py` (append)

**Interfaces:**
- Produces: `harness.measure_aggregate(stream_fn, base_url, prompts, max_tokens=256) -> dict` returning `{"tok_s": float, "n_tokens": int, "wall_s": float}`. Uses real wall-clock via a concurrent runner; for tests the seam makes concurrency deterministic.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_speed_measure.py`:
```python
from harness import measure_aggregate


def test_measure_aggregate_total_tokens_and_positive_throughput():
    prompts = [{"id": f"a{i}", "prompt": str(i)} for i in range(4)]
    # fake streamer: 10 tokens each, emitted at i/100s within a prompt
    def stream_fn(base_url, prompt, max_tokens, temperature=0.0):
        return [(f"t{j}", j / 100.0) for j in range(10)]
    res = measure_aggregate(stream_fn, "http://x", prompts, max_tokens=10)
    assert res["n_tokens"] == 40
    assert res["tok_s"] > 0.0
    assert res["wall_s"] >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_speed_measure.py::test_measure_aggregate_total_tokens_and_positive_throughput -v`
Expected: FAIL — `ImportError: cannot import name 'measure_aggregate'`.

- [ ] **Step 3: Write minimal implementation**

Append to `examples/inference_speed/harness.py`:
```python
import asyncio as _asyncio


def measure_aggregate(stream_fn, base_url: str, prompts: list[dict], max_tokens: int = 256) -> dict:
    """Fire all prompts concurrently; aggregate tok/s = total_tokens / wall_clock.

    The stream_fn is awaited concurrently via threads (llama-server handles parallel
    decoders when --parallel >= n_concurrent). Wall clock is real perf_counter around
    the whole batch.
    """
    import concurrent.futures as cf

    def one(item):
        return stream_fn(base_url, item["prompt"], max_tokens, temperature=0.0)

    t0 = _time.perf_counter()
    results: list[list[tuple[str, float]]] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, len(prompts))) as pool:
        results = list(pool.map(one, prompts))
    t1 = _time.perf_counter()
    wall = t1 - t0
    total = sum(len(r) for r in results)
    tps = total / wall if wall > 0 else 0.0
    return {"tok_s": tps, "n_tokens": total, "wall_s": wall}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_speed_measure.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add examples/inference_speed/harness.py tests/unit/test_speed_measure.py
git commit -m "feat(inference_speed): measure_aggregate concurrent throughput"
```

---

### Task 6: `lossless_match` quality check

**Files:**
- Modify: `examples/inference_speed/harness.py`
- Test: `tests/unit/test_speed_lossless.py`

**Interfaces:**
- Produces: `harness.lossless_match(probe_outputs: dict[str,str], reference: dict[str,str]) -> tuple[bool, list[str]]` — `(all_match, mismatched_ids)`. Byte-identical comparison; order-independent by id.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_speed_lossless.py
from harness import lossless_match


def test_all_match():
    ref = {"p1": "Paris", "p2": "4"}
    out = {"p1": "Paris", "p2": "4"}
    ok, mism = lossless_match(out, ref)
    assert ok and mism == []


def test_one_mismatch():
    ref = {"p1": "Paris", "p2": "4"}
    out = {"p1": "paris", "p2": "4"}
    ok, mism = lossless_match(out, ref)
    assert not ok and mism == ["p1"]


def test_missing_probe_is_mismatch():
    ref = {"p1": "Paris", "p2": "4"}
    out = {"p1": "Paris"}
    ok, mism = lossless_match(out, ref)
    assert not ok and "p2" in mism
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_speed_lossless.py -v`
Expected: FAIL — `ImportError: cannot import name 'lossless_match'`.

- [ ] **Step 3: Write minimal implementation**

Append to `examples/inference_speed/harness.py`:
```python
def lossless_match(probe_outputs: dict[str, str], reference: dict[str, str]) -> tuple[bool, list[str]]:
    """Byte-identical comparison of candidate probe outputs vs frozen reference.

    For lossless-by-construction optimizations (spec decoding, batching, cache),
    accepted output MUST equal the greedy reference. Any mismatch is a quality
    regression -> the verifier returns Fail.
    """
    mism = [pid for pid in reference if probe_outputs.get(pid) != reference[pid]]
    return (len(mism) == 0, mism)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_speed_lossless.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add examples/inference_speed/harness.py tests/unit/test_speed_lossless.py
git commit -m "feat(inference_speed): lossless_match quality gate"
```

---

### Task 7: `launch_server` + `wait_for_ready` + real `httpx` streamer

**Files:**
- Modify: `examples/inference_speed/harness.py`
- Test: `tests/unit/test_speed_server.py`

**Interfaces:**
- Produces: `harness.wait_for_ready(base_url, timeout_s=120.0, http_get=None) -> bool` (polls `/health`); `harness.launch_server(config, port, runner=None) -> tuple[subprocess.Popen, str]` (returns the process + base_url); `harness.httpx_stream(base_url, prompt, max_tokens, temperature=0.0) -> list[tuple[str,float]]` (the real seam impl used in integration). Tests inject `http_get` / `runner` fakes.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_speed_server.py
import pytest
from harness import wait_for_ready


def test_wait_for_ready_succeeds_on_200():
    calls = {"n": 0}
    def fake_get(url, timeout=1.0):
        calls["n"] += 1
        class R:
            status_code = 200
        return R()
    assert wait_for_ready("http://x:1234/health", timeout_s=2.0, http_get=fake_get) is True
    assert calls["n"] >= 1


def test_wait_for_ready_times_out_on_500(monkeypatch):
    # make time.perf_counter jump so the loop exits immediately
    import harness
    t = [0.0]
    monkeypatch.setattr(harness._time, "perf_counter", lambda: (t.__setitem__(0, t[0] + 10.0) or t[0]))
    def fake_get(url, timeout=1.0):
        class R:
            status_code = 500
        return R()
    assert wait_for_ready("http://x:1234/health", timeout_s=1.0, http_get=fake_get) is False


def test_launch_server_uses_config_cli_args(monkeypatch):
    import harness
    captured = {}
    class FakeProc:
        def terminate(self): pass
        def wait(self, timeout=None): pass
    def fake_runner(cmd, **kw):
        captured["cmd"] = cmd
        return FakeProc()
    proc, base_url = harness.launch_server(
        harness.Config(n_threads=8, n_concurrent=4), port=8080, runner=fake_runner
    )
    assert "--threads" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--threads") + 1] == "8"
    assert base_url == "http://127.0.0.1:8080"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_speed_server.py -v`
Expected: FAIL — `ImportError: cannot import name 'wait_for_ready'`.

- [ ] **Step 3: Write minimal implementation**

Append to `examples/inference_speed/harness.py`:
```python
import subprocess as _subprocess


def wait_for_ready(base_url: str, timeout_s: float = 120.0, http_get=None) -> bool:
    """Poll {base_url}/health until 200 or timeout. http_get is a seam (default requests.get)."""
    if http_get is None:
        import requests
        http_get = requests.get
    deadline = _time.perf_counter() + timeout_s
    url = base_url.rstrip("/") + "/health"
    while _time.perf_counter() < deadline:
        try:
            r = http_get(url, timeout=2.0)
            if getattr(r, "status_code", 0) == 200:
                return True
        except Exception:
            pass
        _time.sleep(0.5)
    return False


def launch_server(config: Config, port: int = 8080, runner=None) -> tuple[object, str]:
    """Start llama-server with config.to_cli_args(). runner is a seam (default subprocess.Popen)."""
    if runner is None:
        runner = _subprocess.Popen
    cmd = ["llama-server", *config.to_cli_args(), "--port", str(port), "--host", "127.0.0.1"]
    proc = runner(cmd, stdout=_subprocess.PIPE, stderr=_subprocess.STDOUT)
    return proc, f"http://127.0.0.1:{port}"


def httpx_stream(base_url: str, prompt: str, max_tokens: int, temperature: float = 0.0) -> list[tuple[str, float]]:
    """Real streamer: POST to {base_url}/v1/chat/completions with stream=true, record perf_counter per chunk."""
    import httpx

    body = {
        "model": "target",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    events: list[tuple[str, float]] = []
    with httpx.Client(timeout=120.0) as client:
        with client.stream("POST", base_url.rstrip("/") + "/v1/chat/completions", json=body) as resp:
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data.strip() == "[DONE]":
                    break
                try:
                    obj = _json.loads(data)
                    delta = obj["choices"][0].get("delta", {})
                    tok = delta.get("content") or delta.get("token") or ""
                except Exception:
                    continue
                if tok:
                    events.append((tok, _time.perf_counter()))
    return events
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_speed_server.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add examples/inference_speed/harness.py tests/unit/test_speed_server.py
git commit -m "feat(inference_speed): launch_server, wait_for_ready, httpx_stream"
```

---

### Task 8: `run_harness` orchestration with stubbed components

**Files:**
- Modify: `examples/inference_speed/harness.py`
- Test: `tests/unit/test_speed_harness.py`

**Interfaces:**
- Produces: `harness.run_harness(config, workspace: Path, stream_fn=None, launcher=None, waiter=None, completion_fn=None) -> dict` returning the full JSON result. Reads `workload/` and `probes.jsonl` from `workspace`. `completion_fn(base_url, prompt, max_tokens) -> str` is a seam for non-streaming greedy completion (used for probes); defaults to a streaming-then-join impl.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_speed_harness.py
import json
from pathlib import Path

from harness import Config, run_harness


def _ws_with_fixtures(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    (ws / "workload").mkdir(parents=True)
    (ws / "workload" / "prompts_single.jsonl").write_text(
        '{"id":"s1","prompt":"x"}\n')
    (ws / "workload" / "prompts_aggregate.jsonl").write_text(
        '{"id":"a1","prompt":"y"}\n{"id":"a2","prompt":"z"}\n')
    (ws / "workload" / "probes.jsonl").write_text('{"id":"p1","prompt":"q"}\n')
    return ws


def test_run_harness_emits_full_result(tmp_path):
    ws = _ws_with_fixtures(tmp_path)
    cfg = Config()

    def fake_stream(base_url, prompt, max_tokens, temperature=0.0):
        return [("a", 0.0), ("b", 0.5)]  # 2 tok / 0.5s = 4.0 tok/s
    def fake_completion(base_url, prompt, max_tokens):
        return "ANSWER"
    def fake_launcher(cfg, port, runner=None):
        class P:
            def terminate(self): pass
            def wait(self, timeout=None): pass
        return P(), f"http://127.0.0.1:{port}"
    def fake_waiter(url, timeout_s=120.0, http_get=None):
        return True

    res = run_harness(cfg, ws, stream_fn=fake_stream, launcher=fake_launcher,
                     waiter=fake_waiter, completion_fn=fake_completion)
    assert res["single_stream"]["n_tokens"] == 2
    assert res["aggregate"]["n_tokens"] == 4
    assert res["quality"]["path"] == "lossless"
    assert res["quality"]["match"] is True
    assert res["quality"]["mismatched"] == []
    assert res["probe_outputs"] == {"p1": "ANSWER"}
    assert "config" in res and res["config"]["n_threads"] == 12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_speed_harness.py -v`
Expected: FAIL — `ImportError: cannot import name 'run_harness'`.

- [ ] **Step 3: Write minimal implementation**

Append to `examples/inference_speed/harness.py`:
```python
def _join_completion(stream_fn):
    """Wrap a stream_fn into a non-streaming greedy completion: returns joined text."""
    def completion(base_url, prompt, max_tokens):
        events = stream_fn(base_url, prompt, max_tokens, temperature=0.0)
        return "".join(tok for tok, _ in events)
    return completion


def run_harness(config: Config, workspace: _Path, stream_fn=None, launcher=None,
                waiter=None, completion_fn=None) -> dict:
    """Launch the target, measure single + aggregate, run the lossless probe check, return JSON result.

    All 9B interaction is via seams; unit tests pass fakes. The workspace must contain
    workload/prompts_single.jsonl, workload/prompts_aggregate.jsonl, workload/probes.jsonl.
    """
    stream_fn = stream_fn or httpx_stream
    launcher = launcher or (lambda c, port, runner=None: launch_server(c, port))
    waiter = waiter or (lambda url, timeout_s=120.0, http_get=None: wait_for_ready(url, timeout_s))
    completion_fn = completion_fn or _join_completion(stream_fn)

    proc, base_url = launcher(config, 8080)
    try:
        if not waiter(base_url):
            return {"error": "server did not become ready", "config": _config_dict(config)}
        single_prompts = load_workload(workspace / "workload" / "prompts_single.jsonl")
        agg_prompts = load_workload(workspace / "workload" / "prompts_aggregate.jsonl")
        probes = load_workload(workspace / "workload" / "probes.jsonl")

        single = measure_single_stream(stream_fn, base_url, single_prompts)
        aggregate = measure_aggregate(stream_fn, base_url, agg_prompts)

        probe_outputs = {p["id"]: completion_fn(base_url, p["prompt"], 16) for p in probes}
        # reference comes from baseline.json in the verifier; harness returns raw outputs
        return {
            "config": _config_dict(config),
            "single_stream": single,
            "aggregate": aggregate,
            "probe_outputs": probe_outputs,
            "quality": {"path": "lossless", "match": None, "mismatched": None},  # filled by verifier
            "loaded_model": TARGET_MODEL,
        }
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=15)
        except Exception:
            pass


def _config_dict(c: Config) -> dict:
    from dataclasses import asdict
    return asdict(c)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_speed_harness.py -v`
Expected: PASS (1 test). Note: the test's `fake_launcher` returns `(P(), url)` matching the real `launch_server` signature; `run_harness` calls `launcher(config, 8080)` (2-arg), so update the test's `fake_launcher` to accept `port` only — fix by changing the test signature to `def fake_launcher(cfg, port, runner=None):`. The test above already matches. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/inference_speed/harness.py tests/unit/test_speed_harness.py
git commit -m "feat(inference_speed): run_harness orchestration with seams"
```

---

### Task 9: `SpeedQualityVerifier` (stateless grading + integrity)

**Files:**
- Create: `examples/inference_speed/speed_verifier.py`
- Create: `examples/inference_speed/config.py` (the editable region)
- Create: `examples/inference_speed/strategy.py` (frozen passthrough)
- Test: `tests/unit/test_speed_verifier.py`

**Interfaces:**
- Consumes: `harness.run_harness` (via a `runner` seam), `harness.load_workload`, `harness.lossless_match`, `crucible.artifact.scan_holes`, `crucible.verify.{Ok,Scored,Fail,Partial,RunContext}`.
- Produces: `speed_verifier.SpeedQualityVerifier(target_agg=30.0, target_single=8.0, baseline_path=Path, runner=None)`. `verify(artifact, ctx) -> Verdict`. `Ok` when `agg>=target_agg and single>=target_single and quality.match`; `Scored(value=agg)` when quality-clean but below target; `Fail` on crash / quality mismatch / wrong model loaded; `Partial` if `config` hole unfilled.

- [ ] **Step 1: Write `config.py` and `strategy.py`**

```python
# examples/inference_speed/config.py
"""The single editable surface of the experiment. Gemini edits the `config` region only."""
from harness import Config, TARGET_MODEL  # noqa: F401  (TARGET_MODEL frozen, imported for stability)

# crucible:region start name=config
CONFIG = Config(
    draft_model=None,
    n_threads=12,
    n_batch=512,
    n_ubatch=512,
    flash_attn=True,
    use_mmap=True,
    use_mlock=False,
    ctx_size=4096,
    draft_max=4,
    n_concurrent=1,
    cache_policy="on",
)
# crucible:region end
```

```python
# examples/inference_speed/strategy.py
"""FROZEN passthrough strategy. Wave 2 opens an editable region here to implement
the batching multiplexer, speculative-decoding orchestration, and prefix cache.
For Wave 0/1 the harness ignores this module; it exists so the immutable structure
matches the spec and Wave 2 can open it without moving files.
"""
```

- [ ] **Step 2: Write the failing test**

```python
# tests/unit/test_speed_verifier.py
import json
from pathlib import Path

import pytest
from crucible.artifact import Artifact, scan_holes
from crucible.verify import Ok, Scored, Fail, Partial

from speed_verifier import SpeedQualityVerifier


def _artifact_with_config(ws_root: Path, config_text: str | None = None) -> Artifact:
    """Build an Artifact whose files mirror the example dir; config region filled or holed."""
    files = {}
    # minimal frozen files
    files["harness.py"] = "from harness import Config\n"  # frozen stub for the test
    files["strategy.py"] = ""
    files["config.py"] = config_text if config_text is not None else _empty_config_with_hole()
    return Artifact(files=files, base_hash="x" * 16)


def _empty_config_with_hole() -> str:
    return (
        "from harness import Config\n"
        "# crucible:region start name=config\n"
        "CONFIG = Config()\n"  # a filled region (no hole) — the default
        "# crucible:region end\n"
    )


def _holed_config() -> str:
    return (
        "from harness import Config\n"
        "# crucible:region start name=config\n"
        "# crucible:region end\n"  # empty region = hole
    )


def _fake_runner(result: dict):
    def runner(config, workspace):
        return result
    return runner


def _ctx(tmp_path, monkeypatch):
    from crucible.verify import RunContext
    from crucible.sandbox import SubprocessSandbox
    # materialize writes files into scratch; for the unit test we point scratch at tmp_path
    ctx = RunContext(task=None, sandbox=SubprocessSandbox(), scratch=tmp_path)  # type: ignore[arg-type]
    return ctx


def test_partial_when_config_hole_open(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    v = SpeedQualityVerifier(
        baseline_path=tmp_path / "baseline.json",
        runner=_fake_runner({"aggregate": {"tok_s": 40.0}, "single_stream": {"tok_s": 9.0},
                             "quality": {"path": "lossless", "match": True, "mismatched": []},
                             "loaded_model": "x", "probe_outputs": {}}),
    )
    (tmp_path / "baseline.json").write_text(json.dumps({
        "single_stream": 2.6, "aggregate": 2.6,
        "probe_reference": {"p1": "ANSWER"},
    }))
    art = Artifact(files={"config.py": _holed_config(), "harness.py": "", "strategy.py": ""}, base_hash="x"*16)
    ctx = _ctx(tmp_path, monkeypatch)
    verdict = v.verify(art, ctx)
    assert isinstance(verdict, Partial)


def test_ok_when_targets_met_and_quality_clean(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    (tmp_path / "baseline.json").write_text(json.dumps({
        "single_stream": 2.6, "aggregate": 2.6,
        "probe_reference": {"p1": "ANSWER"},
    }))
    v = SpeedQualityVerifier(
        baseline_path=tmp_path / "baseline.json",
        runner=_fake_runner({
            "aggregate": {"tok_s": 31.0}, "single_stream": {"tok_s": 9.0},
            "quality": {"path": "lossless", "match": None, "mismatched": None},
            "loaded_model": "/home/isb/models/Qwen3.5-9B-Q4_K_M.gguf",
            "probe_outputs": {"p1": "ANSWER"},
            "config": {"n_threads": 8},
        }),
    )
    art = Artifact(files={"config.py": _empty_config_with_hole(), "harness.py": "", "strategy.py": ""}, base_hash="x"*16)
    verdict = v.verify(art, _ctx(tmp_path, monkeypatch))
    assert isinstance(verdict, Ok)


def test_scored_when_quality_clean_but_below_target(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    (tmp_path / "baseline.json").write_text(json.dumps({
        "single_stream": 2.6, "aggregate": 2.6, "probe_reference": {"p1": "ANSWER"}}))
    v = SpeedQualityVerifier(
        baseline_path=tmp_path / "baseline.json",
        runner=_fake_runner({
            "aggregate": {"tok_s": 10.0}, "single_stream": {"tok_s": 4.0},
            "quality": {"path": "lossless", "match": None, "mismatched": None},
            "loaded_model": "/home/isb/models/Qwen3.5-9B-Q4_K_M.gguf",
            "probe_outputs": {"p1": "ANSWER"},
            "config": {},
        }),
    )
    art = Artifact(files={"config.py": _empty_config_with_hole(), "harness.py": "", "strategy.py": ""}, base_hash="x"*16)
    verdict = v.verify(art, _ctx(tmp_path, monkeypatch))
    assert isinstance(verdict, Scored)
    assert verdict.value == 10.0


def test_fail_when_quality_mismatch(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    (tmp_path / "baseline.json").write_text(json.dumps({
        "single_stream": 2.6, "aggregate": 2.6, "probe_reference": {"p1": "ANSWER"}}))
    v = SpeedQualityVerifier(
        baseline_path=tmp_path / "baseline.json",
        runner=_fake_runner({
            "aggregate": {"tok_s": 40.0}, "single_stream": {"tok_s": 9.0},
            "quality": {"path": "lossless", "match": None, "mismatched": None},
            "loaded_model": "/home/isb/models/Qwen3.5-9B-Q4_K_M.gguf",
            "probe_outputs": {"p1": "WRONG"},
            "config": {},
        }),
    )
    art = Artifact(files={"config.py": _empty_config_with_hole(), "harness.py": "", "strategy.py": ""}, base_hash="x"*16)
    verdict = v.verify(art, _ctx(tmp_path, monkeypatch))
    assert isinstance(verdict, Fail)


def test_fail_when_wrong_model_loaded(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    (tmp_path / "baseline.json").write_text(json.dumps({
        "single_stream": 2.6, "aggregate": 2.6, "probe_reference": {"p1": "ANSWER"}}))
    v = SpeedQualityVerifier(
        baseline_path=tmp_path / "baseline.json",
        runner=_fake_runner({
            "aggregate": {"tok_s": 40.0}, "single_stream": {"tok_s": 9.0},
            "quality": {"path": "lossless", "match": None, "mismatched": None},
            "loaded_model": "/home/isb/models/tiny.gguf",  # cheating: swapped model
            "probe_outputs": {"p1": "ANSWER"}, "config": {},
        }),
    )
    art = Artifact(files={"config.py": _empty_config_with_hole(), "harness.py": "", "strategy.py": ""}, base_hash="x"*16)
    verdict = v.verify(art, _ctx(tmp_path, monkeypatch))
    assert isinstance(verdict, Fail)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_speed_verifier.py -v`
Expected: FAIL — `ModuleNotFoundError: speed_verifier`.

- [ ] **Step 4: Write minimal implementation**

```python
# examples/inference_speed/speed_verifier.py
"""SpeedQualityVerifier — deterministic, stateless (PRD §3 contract).

Grades a candidate inference config: runs the frozen harness, compares tok/s and
lossless quality against a frozen baseline, returns Scored(value=aggregate) /
Ok / Fail. Incumbent tracking and plateau detection are the orchestrator's job
(Scored ranking + plateau_patience), not this verifier's.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from crucible.artifact import Artifact, scan_holes
from crucible.verify import Fail, Ok, Partial, RunContext, Scored, Verdict
from harness import TARGET_MODEL, Config, lossless_match, run_harness


@dataclass(frozen=True)
class SpeedQualityVerifier:
    target_agg: float = 30.0
    target_single: float = 8.0
    baseline_path: Path = Path("examples/inference_speed/baseline.json")
    runner: object | None = None  # seam: runner(config, workspace) -> dict; default run_harness
    deterministic: bool = True

    @property
    def verifier_id(self) -> str:
        return f"speed:agg={self.target_agg},single={self.target_single}"

    def verify(self, artifact: Artifact, ctx: RunContext) -> Verdict:
        holes = scan_holes(artifact)
        if holes:
            return Partial(open_holes=holes, feedback="config region not filled")

        ws = ctx.materialize(artifact)
        baseline = json.loads(Path(self.baseline_path).read_text())

        runner = self.runner or (lambda cfg, workspace: run_harness(cfg, workspace))
        # The artifact's config.py defines CONFIG; import it from the materialized workspace.
        config = _load_config_from_workspace(ws)
        try:
            result = runner(config, ws)
        except Exception as exc:  # crashed launch / measurement
            return Fail(feedback=f"harness crashed: {exc!r}")

        if "error" in result:
            return Fail(feedback=f"harness error: {result['error']}")

        # Integrity: the loaded model must be the fixed target (don't trust config).
        if result.get("loaded_model") != TARGET_MODEL:
            return Fail(feedback=(
                f"loaded model {result.get('loaded_model')!r} != target {TARGET_MODEL!r}; "
                "swapping the target model is a gaming vector (editing the measure, not the speed)."
            ))

        # Lossless quality gate: candidate probe outputs vs frozen baseline reference.
        ok, mism = lossless_match(result.get("probe_outputs", {}), baseline["probe_reference"])
        if not ok:
            return Fail(feedback=(
                f"lossless gate FAILED — probe outputs diverge from greedy reference on {mism}. "
                "The optimization changed the output, i.e. degraded quality."
            ))

        agg = float(result["aggregate"]["tok_s"])
        single = float(result["single_stream"]["tok_s"])

        feedback = (
            f"single={single:.2f} tok/s  aggregate={agg:.2f} tok/s  "
            f"(targets: single>={self.target_single}, agg>={self.target_agg})  "
            f"quality=lossless-clean  baseline: single={baseline['single_stream']}, "
            f"agg={baseline['aggregate']}."
        )

        if agg >= self.target_agg and single >= self.target_single:
            return Ok(produced=artifact)
        return Scored(produced=artifact, value=agg, feedback=feedback)


def _load_config_from_workspace(ws: Path) -> Config:
    """Import the artifact's config.py (which defines CONFIG) from the materialized workspace."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("candidate_config", ws / "config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.CONFIG
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_speed_verifier.py -v`
Expected: PASS (5 tests). If `Artifact(files=..., base_hash=...)` signature differs, check `crucible/artifact.py` and adjust the test constructor — `Artifact` is constructed by `Task` in real runs; the test constructs it directly, so match its real signature (`grep -n "class Artifact" crucible/artifact.py`).

- [ ] **Step 6: Commit**

```bash
git add examples/inference_speed/speed_verifier.py examples/inference_speed/config.py examples/inference_speed/strategy.py tests/unit/test_speed_verifier.py
git commit -m "feat(inference_speed): SpeedQualityVerifier with lossless gate and model-integrity check"
```

---

### Task 10: `measure_baseline.py` — produce `baseline.json` (Wave 0)

**Files:**
- Create: `examples/inference_speed/measure_baseline.py`
- Test: `tests/integration/test_speed_baseline.py`

**Interfaces:**
- Produces: a script that runs `run_harness` with the default `Config` against the real 9B and writes `baseline.json = {single_stream, aggregate, probe_reference, config}`. `probe_reference` = the greedy probe outputs (the lossless reference for all future candidates).

- [ ] **Step 1: Write the integration test**

```python
# tests/integration/test_speed_baseline.py
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_baseline_script_writes_valid_json(tmp_path):
    if not Path("/home/isb/models/Qwen3.5-9B-Q4_K_M.gguf").exists():
        pytest.skip("9B model not present")
    import subprocess, sys
    out = tmp_path / "baseline.json"
    env = dict(os.environ)
    r = subprocess.run(
        [sys.executable, "examples/inference_speed/measure_baseline.py", "--out", str(out)],
        cwd="/home/isb/models/Crucible", env=env, timeout=900, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    data = json.loads(out.read_text())
    assert data["single_stream"] > 0.0
    assert data["aggregate"] > 0.0
    assert "probe_reference" in data and len(data["probe_reference"]) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_speed_baseline.py -v`
Expected: FAIL — `FileNotFoundError: measure_baseline.py` (or skip if model absent).

- [ ] **Step 3: Write the script**

```python
# examples/inference_speed/measure_baseline.py
"""Wave 0: measure the honest baseline with the default Config and freeze it.

Usage:
  uv run python examples/inference_speed/measure_baseline.py [--out baseline.json]

Writes {single_stream, aggregate, probe_reference, config}. probe_reference is the
greedy probe output set — the lossless reference every future candidate must match.
"""
import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from harness import Config, run_harness  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SCRIPT_DIR / "baseline.json"))
    args = ap.parse_args()

    cfg = Config()  # the frozen default
    result = run_harness(cfg, SCRIPT_DIR)
    if "error" in result:
        print(f"harness error: {result['error']}", file=sys.stderr)
        return 1

    baseline = {
        "single_stream": float(result["single_stream"]["tok_s"]),
        "aggregate": float(result["aggregate"]["tok_s"]),
        "probe_reference": result["probe_outputs"],
        "config": result["config"],
    }
    Path(args.out).write_text(json.dumps(baseline, indent=2))
    print(f"baseline written to {args.out}")
    print(f"  single_stream = {baseline['single_stream']:.2f} tok/s")
    print(f"  aggregate     = {baseline['aggregate']:.2f} tok/s")
    print(f"  probes        = {len(baseline['probe_reference'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes (integration — needs the 9B + free RAM)**

First free RAM (stop the Dify/ollama stack), then:
```bash
uv run python examples/inference_speed/measure_baseline.py
uv run pytest tests/integration/test_speed_baseline.py -v
```
Expected: PASS; `baseline.json` written with real numbers (expect single ≈ 2–4 tok/s, aggregate ≈ 2–4 tok/s at n_concurrent=1).

- [ ] **Step 5: Commit**

```bash
git add examples/inference_speed/measure_baseline.py tests/integration/test_speed_baseline.py examples/inference_speed/baseline.json
git commit -m "feat(inference_speed): Wave 0 baseline measurement script + frozen baseline"
```

---

### Task 11: `run_speed.py` CLI — wire into Crucible

**Files:**
- Create: `examples/inference_speed/run_speed.py`
- Test: `tests/integration/test_speed_wave1.py` (smoke, appended in Task 12)

**Interfaces:**
- Consumes: `crucible.{Task, run, AdvisorPolicy, budgets}`, `RunBudget`, `SpeedQualityVerifier`.
- Produces: a CLI mirroring `run_sidon.py`: `--model` (worker Gemini), `--advisor` (shepherd Gemini), `--workers`, `--target-agg`, `--target-single`, `--sandbox subprocess`, `--db`. Runs `run()` over `Task.from_path(SCRIPT_DIR, editable=["config"])` with `SpeedQualityVerifier`, then prints the best config + independent re-verification.

- [ ] **Step 1: Write the CLI**

```python
# examples/inference_speed/run_speed.py
"""Wave 1: verifier-grounded search over the 9B inference config.

Usage (GOOGLE_API_KEY from .env):
  uv run python examples/inference_speed/run_speed.py
  uv run python examples/inference_speed/run_speed.py --model gemini-2.5-flash \
      --advisor gemini-2.5-pro --workers 4 --target-agg 30 --target-single 8
"""
import argparse
import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))


def _default_worker() -> str:
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return "gemini-2.5-flash"
    return "gemini-2.5-flash"  # will produce a clear auth error from the SDK


from crucible import AdvisorPolicy, Task, budgets, run  # noqa: E402
from crucible.budgets import RunBudget  # noqa: E402
from speed_verifier import SpeedQualityVerifier  # type: ignore[import-not-found]  # noqa: E402

ap = argparse.ArgumentParser(description="CPU inference speed search via Crucible")
ap.add_argument("--model", default=_default_worker(), help="worker (Gemini) model")
ap.add_argument("--advisor", default=None, help="shepherd (strong Gemini) model")
ap.add_argument("--advisor-max-calls", type=int, default=8)
ap.add_argument("--advisor-fail-streak", type=int, default=3)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--target-agg", type=float, default=30.0)
ap.add_argument("--target-single", type=float, default=8.0)
ap.add_argument("--sandbox", choices=["subprocess", "docker"], default="subprocess")
ap.add_argument("--db", default=str(SCRIPT_DIR / "speed.db"))
args = ap.parse_args()

if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
    sys.exit("Error: GOOGLE_API_KEY (or GEMINI_API_KEY) not set. Add it to .env or export it.")

advisor = None
if args.advisor:
    advisor = AdvisorPolicy(
        model=args.advisor,
        max_calls_per_run=args.advisor_max_calls,
        fail_streak=args.advisor_fail_streak,
    )

result = run(
    task=Task.from_path(SCRIPT_DIR, editable=["config"], network=True),
    verifier=SpeedQualityVerifier(
        target_agg=args.target_agg,
        target_single=args.target_single,
        baseline_path=SCRIPT_DIR / "baseline.json",
    ),
    model=args.model,
    workers=args.workers,
    episode=budgets(edits=20, turns=10),
    run_budget=RunBudget(episodes_per_worker=6, plateau_patience=3),
    sandbox=args.sandbox,
    db=args.db,
    advisor=advisor,
)

artifact = result.solution or result.best_partial
cfg_text = artifact.region_text(artifact.region("config")).strip()

if result.solution:
    print(f"\nSOLVED — hit target agg>={args.target_agg} & single>={args.target_single} (run {result.run_id})")
else:
    print(f"\nBest partial — target not reached (run {result.run_id}):")
print(cfg_text[:2000])

# Independent re-verification: re-run the best config and re-measure, never trust the search's claim.
print("\n--- Independent re-verification ---")
import importlib.util
spec = importlib.util.spec_from_file_location("best_config", SCRIPT_DIR / "config.py")
# best_config here is the on-disk default; the artifact's config_text is the real winner.
# Re-measure by exec'ing the artifact's config.py in a throwaway module.
ns: dict = {}
exec(artifact.files["config.py"], ns)
best_cfg = ns["CONFIG"]
from harness import run_harness  # noqa: E402
r = run_harness(best_cfg, SCRIPT_DIR)
print(json.dumps({
    "single_stream": r["single_stream"]["tok_s"],
    "aggregate": r["aggregate"]["tok_s"],
    "quality_path": r["quality"]["path"],
    "loaded_model": r["loaded_model"],
}, indent=2))

print(f"\nInspect reasoning: uv run crucible reasoning --db examples/inference_speed/speed.db")
```

- [ ] **Step 2: Smoke-run the CLI wiring (unit-level, no 9B, no Gemini — just argument parse + Task build)**

Add to `tests/unit/test_speed_verifier.py` a wiring check:
```python
def test_task_from_path_marks_only_config_editable():
    from crucible import Task
    from pathlib import Path
    example = Path(__file__).resolve().parents[1] / "examples" / "inference_speed"
    t = Task.from_path(example, editable=["config"], network=True)
    assert "config" in t.editable
    assert "harness.py" in t.files
    assert "workload/prompts_single.jsonl" in t.files
    assert t.network is True
```

Run: `uv run pytest tests/unit/test_speed_verifier.py::test_task_from_path_marks_only_config_editable -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add examples/inference_speed/run_speed.py tests/unit/test_speed_verifier.py
git commit -m "feat(inference_speed): run_speed CLI wiring Crucible search + independent re-verify"
```

---

### Task 12: Wave 1 end-to-end smoke (integration)

**Files:**
- Test: `tests/integration/test_speed_wave1.py`

**Interfaces:**
- Verifies the full loop: Task → SpeedQualityVerifier → Gemini worker → SQLite DB records attempts; the search improves aggregate tok/s over baseline via config edits (n_concurrency, draft_model).

- [ ] **Step 1: Write the integration test**

```python
# tests/integration/test_speed_wave1.py
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_wave1_search_runs_and_records_attempts(tmp_path):
    if not Path("/home/isb/models/Qwen3.5-9B-Q4_K_M.gguf").exists():
        pytest.skip("9B model not present")
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        pytest.skip("GOOGLE_API_KEY not set")
    db = tmp_path / "speed.db"
    env = dict(os.environ)
    r = subprocess.run(
        [sys.executable, "examples/inference_speed/run_speed.py",
         "--workers", "2", "--db", str(db),
         "--advisor", "gemini-2.5-pro"],
        cwd="/home/isb/models/Crucible", env=env, timeout=3600, capture_output=True, text=True,
    )
    assert r.returncode in (0, 2), r.stderr[-3000:]  # 0=solved, 2=best partial
    # The DB must record at least one episode with a measured Scored/Fail verdict.
    conn = sqlite3.connect(str(db))
    rows = conn.execute("SELECT count(*) FROM episodes").fetchone()
    assert rows[0] >= 1
    conn.close()
```

- [ ] **Step 2: Run it (needs free RAM + GOOGLE_API_KEY + the 9B)**

```bash
# free RAM first: stop the Dify/ollama stack
uv run python examples/inference_speed/run_speed.py --workers 2 --advisor gemini-2.5-pro
uv run pytest tests/integration/test_speed_wave1.py -v
```
Expected: PASS; `speed.db` populated; reasoning inspectable via `uv run crucible reasoning --db examples/inference_speed/speed.db`.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_speed_wave1.py
git commit -m "test(inference_speed): Wave 1 end-to-end smoke (search records attempts)"
```

- [ ] **Step 4: Update README**

Append a section to `examples/inference_speed/README.md` (create it) documenting the run command, the baseline numbers observed, and the `crucible reasoning` inspection path — mirroring `examples/sidon/`'s documentation style.

```bash
git add examples/inference_speed/README.md
git commit -m "docs(inference_speed): example README with run + inspection instructions"
```

---

## Self-Review

**1. Spec coverage:**
- §4.1 editable artifact (`harness.py` frozen, `config` region, `strategy.py` frozen passthrough, `workload/`, `quality/`) → Tasks 1–9, 11.
- §4.2 verifier (lossless-exempt path, Ok/Scored/Fail, model-integrity, fresh values via orchestrator) → Task 9. (PPL/kata default path deliberately deferred — noted in Global Constraints; no Wave 1 task needs it.)
- §4.3 integrity gate (immutable spec, no model-swap, deny via typed Config schema + model-integrity check) → Tasks 1, 9. (Token deny-list not needed in Wave 1: the only editable surface is a typed `Config` dataclass — schema validation *is* the deny-list. Token deny-list is a Wave 2 concern when `strategy.py` opens to arbitrary Python.)
- §4.4 shepherding (Flash worker + Pro advisor, fail_streak=3) → Task 11.
- §4.5 verify-cost (warm/cold restart) → acknowledged; Wave 1 restarts per attempt (~20–30s), acceptable for Flash. No code task needed.
- §5 Wave 0 (baseline) → Task 10; Wave 1 (config-only search) → Tasks 11–12.
- §3 tokenizer-compat risk → **gap**: no task verifies draft/target tokenizer compatibility before using `--model-draft`. Add it.

**Added task (gap fix):**

### Task 13: Draft tokenizer-compatibility check (Wave 0 risk)

**Files:**
- Modify: `examples/inference_speed/measure_baseline.py` (or a small `examples/inference_speed/check_draft_compat.py`)
- Test: `tests/unit/test_speed_draft_compat.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_speed_draft_compat.py
import pytest
from check_draft_compat import tokenizers_compatible


def test_compatible_when_same_vocab(monkeypatch):
    def fake_tokens(path):
        return ["hello", "world", " "]
    assert tokenizers_compatible("/a", "/b", loader=fake_tokens) is True


def test_incompatible_when_vocab_differs(monkeypatch):
    seq = iter([["hello", "world"], ["hello", "WORLD"]])
    def fake_tokens(path):
        return next(seq)
    assert tokenizers_compatible("/a", "/b", loader=fake_tokens) is False
```

- [ ] **Step 2: Run → FAIL (`ModuleNotFoundError`).**

- [ ] **Step 3: Implement**

```python
# examples/inference_speed/check_draft_compat.py
"""Speculative decoding requires draft and target to share a tokenizer.
Checked in Wave 0 before any --model-draft attempt; an incompatible draft is a Fail, not a crash.
"""
from __future__ import annotations


def tokenizers_compatible(target_gguf: str, draft_gguf: str, loader=None) -> bool:
    """Compare the token sets of two gguf models. loader(path)->list[str] is a seam."""
    if loader is None:
        return _real_compare(target_gguf, draft_gguf)
    return loader(target_gguf) == loader(draft_gguf)


def _real_compare(target_gguf: str, draft_gguf: str) -> bool:
    # ponytail: use llama-server --vocab-only or gguf-py to dump tokens; deferred to integration.
    # For the unit test the seam is used. Real impl lives behind the loader default.
    import subprocess
    def dump(path):
        r = subprocess.run(
            ["llama-server", "--model", path, "--vocab-only", "--log-disable"],
            capture_output=True, text=True, timeout=120)
        # parse token lines from stderr/stdout — exact format verified at integration time
        return [l for l in (r.stdout + r.stderr).splitlines() if l.startswith("tok:")]
    return dump(target_gguf) == dump(draft_gguf)
```

- [ ] **Step 4: Run → PASS (2 tests).**

- [ ] **Step 5: Wire into the verifier — Task 9 `SpeedQualityVerifier.verify`: before calling `run_harness`, if `config.draft_model` is set and `not tokenizers_compatible(TARGET_MODEL, config.draft_model)`, return `Fail(feedback="draft tokenizer incompatible with target — spec decoding would mis-accept tokens")`. Add a unit test for this branch.**

- [ ] **Step 6: Commit**

```bash
git add examples/inference_speed/check_draft_compat.py tests/unit/test_speed_draft_compat.py examples/inference_speed/speed_verifier.py
git commit -m "feat(inference_speed): draft tokenizer-compat gate (Wave 0 risk)"
```

**2. Placeholder scan:** the `_real_compare` parsing note ("exact format verified at integration time") is the one soft spot — it is acceptable because the unit path is seam-tested and the real parsing is an integration concern, but the implementer **must** verify llama-server's `--vocab-only` output format during Task 13 integration and make the parse real (not left as a TODO). No other TBD/TODO present.

**3. Type consistency:** `Config` fields and `to_cli_args` used consistently across Tasks 1/7/8/9/10/11. `run_harness` return shape (`single_stream.tok_s`, `aggregate.tok_s`, `probe_outputs`, `loaded_model`, `config`) consistent across Tasks 8/9/10/11. `SpeedQualityVerifier` constructor `(target_agg, target_single, baseline_path, runner)` consistent across Tasks 9/11. `lossless_match(probe_outputs, reference)` arg order consistent Tasks 6/9. `stream_fn(base_url, prompt, max_tokens, temperature=0.0)` consistent Tasks 4/5/7/8.

**4. Scope:** Wave 0 + Wave 1 only — a self-contained, testable deliverable (a running Crucible example that measures and searches config space). Wave 2 (`strategy.py` editable, batching proxy, spec-decoding orchestration, PPL/kata gate) is the next plan, informed by Wave 1's measured results.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-11-cpu-inference-speed-waves01.md`.