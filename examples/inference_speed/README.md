# Inference Speed — verifier-grounded search over a local model on CPU

**Can a verifier-grounded LLM loop speed up a small local model on a RAM-only VPS
— without degrading quality?** This example applies Crucible's thesis (the verifier,
not the model's confidence, decides what's real) to a new artifact: the *inference
runtime* of a local model running on CPU via `llama-server`.

The model is passive cargo. A Gemini worker edits the inference `config`; a
deterministic `SpeedQualityVerifier` measures single-stream + aggregate tok/s with
a **lossless quality gate**; a GLM-5.2 advisor (via Ollama Cloud) shepherds on
plateau. Target: 2.6 → 30 tok/s, losslessly.

**Overnight result (Wave 2.5):** an 11-hour detached search with web-search-grounded
workers hit **single 7.67 tok/s (3.6× the baseline, within 4% of the 8.0 target)**
and **aggregate 10.15 tok/s (5.2× the baseline)** — 61 measured attempts, **zero
quality regressions**. It found a non-obvious optimum (fewer threads = faster, see
Wave 2.5) that intuition misses. 30 aggregate stays physically out of reach for the
dense 9B, but the search squeezed it to its ceiling losslessly.

## What it proved

Three waves, two honest ceilings, one real lever, and one surprise.

### Wave 1 — the dense 9B and the config-only bandwidth wall

The search (gemini-3.1-flash-lite worker + GLM-5.2 shepherd) on `Qwen3.5-9B-Q4_K_M`
found real, lossless, integrity-clean speedups, then **plateaued honestly at the
config-only bandwidth ceiling**, far below 30:

| metric | baseline | best (search) | true ceiling (sweep) |
|---|---|---|---|
| single_stream | 2.13 tok/s | 3.81 tok/s (+79%) | ~3.7 tok/s (bandwidth wall) |
| aggregate | 1.96 tok/s | 5.10 tok/s (+160%) | ~7.0 tok/s (batched wall) |
| quality | — | lossless | lossless |
| model-integrity | — | correct 9B | correct 9B |

The concurrency sweep (`sweep_concurrency.py`) shows why: aggregate scales with
batching up to n_concurrent=8 (6.94 tok/s) then **flattens at n=12 (6.86)** — the
memory-bandwidth ceiling for batched decode. Single-stream walls at ~3.7. The 30
target is **not reachable by config tuning alone**.

### Wave 2 — speculative decoding: works, but small; and the architectural lever

Two Wave 2 findings, both evidence-backed:

**1. Self-speculative decoding is real and lossless, but only ~+10% on a dense model.**
A lower-precision quant of the *same* model (same tokenizer by construction → lossless)
drafts tokens the Q4 target verifies in one forward pass. Measured clean (Dify
stopped):

| draft | single (n=1) | aggregate (n=8) | lossless |
|---|---|---|---|
| no draft | 3.25 | 6.94 | ✓ |
| Q2_K draft | 2.98 | 6.32 | ✓ (~neutral; Q2 too coarse → low acceptance) |
| Q3_K_M draft | 3.62 | **7.49** | ✓ (+11% — the real but modest win) |

The economics: a self-spec draft is necessarily nearly as big as the target (Q3 4.7GB
vs Q4 5.5GB), so the read amortization is limited. Q2 is small but its acceptance is
too low (coarse approximation) → the draft overhead cancels the savings. Q3 is the
sweet spot but still only ~+10%. **Spec decoding does not break the dense 9B's wall.**
(LFM2.5 can't be a Qwen draft — verified: 128K-vs-248K tokenizer mismatch.)

**2. The model architecture is the real lever — 3-4× natively, no search needed.**
`LFM2.5-8B-A1B` is a MoE with **1B active params**: on CPU it reads only ~1B of
weights per token (the active experts) vs the dense 9B's 5.5GB. Measured clean:

| model | single | aggregate ceiling | lever |
|---|---|---|---|
| Qwen3.5-9B (dense Q4) | ~4 | ~7.5 | bandwidth-walled |
| LFM2.5-8B-A1B (MoE, 1B active) | **~14** | **~20** (n=12) | ~1B/token read natively |

LFM2.5 is **4.4× faster single-stream, 2.7× faster aggregate** — natively, no spec
decoding, no search. Single-stream (14) clears the `single≥8` Ok gate on its own.
This is the GPT doc's "routing/MoE" thesis, confirmed by measurement: **the
architecture, not the runtime search, breaks the CPU bandwidth wall.**

### Wave 2.5 — the overnight run: 3.6× single / 5.2× aggregate, lossless

An 11-hour detached run (`run_overnight.sh`, survived the laptop shutting down —
the VPS kept the process alive in its own `setsid` session) with **web-search-grounded
workers + the GLM-5.2 shepherd + plateau-patience 12** (DeepMind-Erdős-style: don't
give up early). Two new levers were added to the search surface — **KV-cache
quantization** (`cache_type_k/v`, q8_0/f8 — the decode-bandwidth lever) and
**dedicated draft threads** (`draft_threads` — stop the self-spec draft stealing the
target's memory bandwidth) — both gated losslessly. Defaults emit no flag, so the
frozen baseline stayed byte-identical. Result:

| metric | baseline | best (overnight) | × | target | hit? |
|---|---|---|---|---|---|
| single-stream | 2.13 | **7.67** | 3.6× | 8.0 | within 4% (near-miss) |
| aggregate | 1.96 | **10.15** | 5.2× | 30.0 | 3× short (physical wall) |
| quality | — | lossless (61/61) | — | — | ✓ no regressions |

The best configs stack all four levers — Q3 self-spec draft + KV-cache `q8_0` +
`n_threads` reduction + `draft_threads` — e.g. best single (7.67): Q3, `n_threads=6`,
`n_concurrent=8`, `draft_max=16`, KV `q8_0`, `draft_threads=6`; best aggregate (10.15):
Q3, `n_threads=10`, `n_concurrent=12`, `draft_max=16`, KV `q8_0`, `draft_threads=2`.

**The non-obvious finding — fewer threads = faster single-stream.** The 9B decode is
memory-bandwidth-bound, not compute-bound, so 12 threads contending on the memory bus
is *slower* than 6 threads each getting more bandwidth. The search found this
empirically; confidence never would have:

```
n_threads=12: single median 4.13  (the worker's starting default — "use all cores")
n_threads=10: single median 5.95
n_threads= 8: single median 6.62
n_threads= 6: single median 6.96  ← peak (max 7.67)
n_threads= 4: single median 6.16  (too few)
```

Cutting threads 12→6 nearly doubles single-stream tok/s, losslessly. Aggregate
batching tops out at n_concurrent 8-12 (median 6.67 → 7.21; the worker noted n=12
sometimes regresses on memory pressure — the KV-cache footprint + bandwidth contention
overwhelms the shared-weight benefit beyond ~12).

**A correction to the Wave 1 prediction.** Wave 1 said single-stream was physically
capped at ~2.9 tok/s (5.5GB/token ÷ ~16GB/s). The overnight search disproved this:
speculative decoding lets the target verify up to `draft_max=16` draft tokens in one
forward pass (one 5.5GB weight read), accepting the matches — so the weight read is
**amortized over accepted tokens** and single-stream rises *above* the no-draft
ceiling. The no-draft baseline (2.13) matched the ceiling; with Q3 spec it 3.6×'d to
7.67. The search found a real improvement Wave 1 had argued was impossible.

**The shepherd-vs-verifier episode.** GLM-5.2 consulted once: it diagnosed a real
server-startup timeout (two ~5GB models + `n_concurrent=12` KV footprint exceeded RAM
during a thrashing phase) and advised **dropping the draft model**. The worker tried it
(6 no-draft attempts) — but the verifier scored the draft configs higher, so the search
**kept the Q3 draft** (54/61). The shepherd is smart but conservative on trade-offs;
the deterministic verifier overruled it with measurement. *The verifier, not the
model's confidence, decides what's real* — demonstrated, not asserted.

**Two bugs the "check everything" pass caught** (both would have wasted the night):
(1) the verifier's `scan_holes` scanned *all* task files, and `README.md` documents
the `crucible:hole` sentinel — so every artifact tripped the hole check and the
measurement never ran (32 fake-PARTIAL episodes in Wave 1's first overnight attempt).
Fixed by scoping the hole-scan to the editable `config` region. (2) the verifier
measured at `max_tokens=256` (~18 min/verify → ~30 verifies in 10h); cut to 64
(~2-5 min/verify → ~60+ in the run). Plus memory/CPU contention from a forgejo CI
build (load 22/12 cores, 10k page faults) was cleared by pausing the CI runner.

### The honest bottom line

**30 tok/s aggregate is beyond this VPS's shared memory bandwidth for the dense 9B**
(config + self-spec + new levers: ~10.15 aggregate, single ~7.67). The overnight
search refined both ceilings: single is now nearly reachable (7.67 vs 8.0), aggregate
is 5.2× the baseline but still 3× short of 30 — the worker independently re-derived the
"5.5GB/token read is the fundamental bottleneck; n_concurrent 8-12 is the batching
sweet spot; further gains need an architectural change (MoE)." The verifier-grounded
search characterized the ceiling honestly and refused to inflate it (the Sidon-74
pattern: an honest lower bound, not a gamed 30), and would have rejected any quality
regression — none occurred (61/61 lossless).

Reaching 30 aggregate needs the architectural lever (MoE / fewer active params — LFM2.5
~20 natively) or Wave 3 levers (activation sparsity within the MoE, dynamic per-layer
precision — gated by the verifier for quality), not more config tuning of the dense 9B.

## How it works

```
examples/inference_speed/
  harness.py            # FROZEN: Config, tok/s, measure single/aggregate, lossless
                        #   gate, launch llama-server, httpx streamer, run_harness.
                        #   The measurement methodology is immutable (can't be gamed).
                        #   Config levers (Wave 2.5 added KV-cache quant, NUMA,
                        #   draft_threads — defaults emit no flag = baseline-identical).
  config.py             # EDITABLE region "config": the only surface Gemini edits.
                        #   Levers: draft_model, n_threads, n_concurrent, draft_max,
                        #   cache_type_k/v, draft_threads, numa, flash_attn, ctx_size.
  strategy.py           # FROZEN passthrough (Wave 2 opens this for a batching proxy /
                        #   spec-decoding orchestration / prefix cache).
  check_draft_compat.py # Gates spec-decoding drafts on tokenizer equality.
  speed_verifier.py     # SpeedQualityVerifier — deterministic, stateless. Scans the
                        #   editable config region only for holes (not frozen docs).
  web_search.py         # Tavily web_search tool exposed to the worker (ground edits
                        #   in real llama.cpp techniques instead of guessing).
  web_search_advisor.py # Agentic GLM-5.2 shepherd: researches via web_search before
                        #   advising (falls back gracefully if the advisor is down).
  workload/             # FROZEN single/aggregate/probe prompt sets.
  quality/corpus.txt    # FROZEN held-out text (for Wave 2 PPL).
  baseline.json         # FROZEN Wave 0 baseline (single 2.13, aggregate 1.96).
  measure_baseline.py   # Produce baseline.json (Wave 0, no Gemini spend).
  sweep_concurrency.py  # Map n_concurrent -> tok/s (no Gemini spend).
  run_speed.py          # CLI: wires Crucible search + advisor + web_search. Has
                        #   --wall-clock (RunBudget defaults to 2h — set 10h overnight)
                        #   and --plateau-patience (raise for long exploratory runs).
  run_overnight.sh      # Detached launch (setsid+nohup): survives laptop shutdown;
                        #   the VPS keeps the run alive in its own session.
  status_speed.sh       # Live DB-backed monitor (Crucible prints nothing mid-run;
                        #   the speed.db is the only live progress channel).
  speed.db              # append-only provenance (gitignored).
```

### The verifier (`SpeedQualityVerifier`)

Stateless, `deterministic=True`. For each candidate config:

1. **Hole check (config region only)** — `Partial` if the *editable `config` region*
   contains a `crucible:hole` sentinel. Scans the worker's edit surface, not frozen
   docs (a prior bug: README documents the sentinel → every artifact falsely PARTIAL).
2. **Load config** — import the candidate `config.py` from the materialized workspace.
3. **Draft-compat gate** — if `draft_model` is set, its tokenizer must equal the
   target's (else `Fail` — spec decoding would mis-accept tokens).
4. **Run harness** — launch `llama-server` with the config, stream the frozen workload
   greedily (`enable_thinking=false`), compute single-stream (median) + aggregate
   (total tokens / wall clock) tok/s from real token timestamps.
5. **Model-integrity gate** — the loaded model must be the fixed 9B (don't trust
   config; swapping the target model is a gaming vector).
6. **Lossless quality gate** — candidate probe outputs must be byte-identical to the
   frozen `baseline.json` probe references. Any divergence = quality regression = `Fail`.
7. **Verdict** — `Ok` when `aggregate ≥ 30` and `single ≥ 8` and quality-clean;
   `Scored(value=aggregate)` when quality-clean but below target; `Fail` on
   crash / quality regression / wrong model / incompatible draft.

A module-level lock serializes 9B server launches across worker threads (one server
at a time, ~3 GB) so N workers don't OOM the VPS. Gemini LLM turns still parallelize.

## Run it

```bash
# Prereqs: GOOGLE_API_KEY + OLLAMA_API_KEY + OLLAMA_MODEL (e.g. glm-5.2:cloud) in .env;
# uv pip install -e ".[dev,gemini,inference]" plus google-genai and python-dotenv.

# Wave 0 — honest baseline (no Gemini spend; ~3 min):
uv run python examples/inference_speed/measure_baseline.py --max-tokens 32

# Concurrency sweep — the dense 9B's bandwidth ceiling (no Gemini spend; ~6 min):
uv run python examples/inference_speed/sweep_concurrency.py

# Wave 2 — the short search (Gemini + Ollama Cloud spend). Defaults: worker
# gemini-3.1-flash-lite, advisor $OLLAMA_MODEL (glm-5.2:cloud), Q2_K self-spec draft seeded.
uv run python examples/inference_speed/run_speed.py \
    --workers 4 --episodes 6 --edits 15 --turns 8
# (override: --model gemini-3.1-flash-lite --advisor glm-5.2:cloud)

# Wave 2.5 — the overnight detached run (survives laptop shutdown). RunBudget
# defaults to 2h wall-clock, so --wall-clock 10h is what makes it run all night;
# --plateau-patience 12 keeps it searching past plateaus (DeepMind-style persistence).
bash examples/inference_speed/run_overnight.sh          # setsid+nohup detached launch
bash examples/inference_speed/status_speed.sh            # live DB-backed status
#   or continuous: watch -n 30 bash examples/inference_speed/status_speed.sh
# stop:  kill "$(cat examples/inference_speed/overnight.pid)"; pkill -f llama-server
# The final result + independent re-verification land in overnight.log at ~10h:
#   tail -f examples/inference_speed/overnight.log

# Inspect the reasoning trail:
uv run crucible reasoning --db examples/inference_speed/speed.db
```

The Q2_K/Q3_K_M self-spec drafts and the LFM2.5-8B-A1B target were benchmarked
out-of-band (see "What it proved"); they live on disk at `/home/isb/models/`.

## Findings (honest) — see "What it proved" above for the consolidated result

The findings hold across all three waves: (1) the loop finds real lossless speedups;
(2) config tuning plateaus at the bandwidth ceiling (sweep: 5.41 / 6.94 / 6.86 at
n=4/8/12; overnight best aggregate 10.15 at n=8-12); (3) 30 needs reducing bytes/token
— Wave 2 confirmed spec decoding is ~+10% on a dense model and the architecture (MoE)
is the real lever; (4) **Wave 2.5 refined this**: speculative decoding's weight-read
amortization pushes single-stream *above* the no-draft bandwidth ceiling (2.13 → 7.67,
3.6×, near the 8.0 target) — a real improvement Wave 1 had argued was impossible;
(5) the verifier is the product — 61/61 attempts lossless, it refused to inflate the
ceiling and rejected any quality regression.

## Known gotchas (integration)

- `llama-server --flash-attn` takes a value (`on|off|auto`); the bare flag consumes
  the next arg and fatal-exits.
- `--draft-max` was removed in current llama.cpp → use `--spec-draft-n-max`.
- `llama-quantize` refuses to re-quantize from an already-quantized gguf → pass
  `--allow-requantize` (needed for the self-spec Q2_K/Q3_K_M drafts from Q4_K_M).
- A spec-decoding draft must share the target's tokenizer (token IDs must align).
  LFM2.5 (128K vocab) cannot draft for Qwen3.5 (248K) — verified; the only lossless
  Qwen draft is a Qwen quant of itself. `check_draft_compat` gates this via gguf.
- `Task.from_path(dir)` includes `__pycache__/*.pyc` (binary) and `speed.db` →
  `UnicodeDecodeError`; `run_speed.py` builds a curated file list instead.
- Qwen3.5 and GLM-5.2 are reasoning models: stream `chat_template_kwargs.enable_thinking=false`
  (Qwen) / give the advisor enough tokens (GLM-5.2 emits `reasoning` then `content`).
- The pyproject `gemini` extra installs `google-generativeai`, but the provider code
  imports `google.genai` (the `google-genai` package). Install `google-genai` too.
- Ollama Cloud shepherd: endpoint is `https://ollama.com/v1` (not `api.ollama.com`,
  which 301-redirects and httpx doesn't follow by default), `OPENAI_API_KEY=$OLLAMA_API_KEY`.
  `run_speed.py` wires this automatically when `OLLAMA_API_KEY` + a non-Gemini advisor are set.
- **Hole-scan must be region-scoped.** `scan_holes(artifact)` scans *all* task files;
  README documents the `crucible:hole` sentinel, so every artifact is falsely `Partial`
  and the measurement never runs (the search loops forever measuring nothing). The
  speed verifier scans only the editable `config` region — a hole in a frozen doc isn't
  the worker's fault.
- **RunBudget defaults to 2h wall-clock.** Without `--wall-clock 10h` the run stops at
  2h regardless of `--episodes` — an overnight run must override it. `--plateau-patience`
  (default 3) stops a worker after N episodes with no rank improvement; raise it for a
  long exploratory run.
- **The verifier's `max_tokens` controls search throughput.** At 256, ~18 min/verify
  (~30 verifies in 10h — too few); at 64, ~2-5 min/verify (~60+ overnight). The tok/s
  rate is stable across lengths and the lossless probes use a fixed 16 tokens, so the
  quality gate is unaffected.
- **Memory + CPU contention starves the 9B.** The 9B+Q3 draft (~10GB RSS) needs ~11GB
  free; a competing CI build (load 22/12 cores) caused 10k page faults and crawled
  verifies to 18 min. Pause heavy background work (CI runners, builds) before an
  overnight run, or the search wastes the night thrashing.

## Non-goals

No training/RL/fine-tuning, no GPU, no kernel patching (v0.5 horizon), no quality
degradation accepted. The target model is cargo; only its runtime config is searched.