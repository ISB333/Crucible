# Inference Speed — verifier-grounded search over a local 9B on CPU

**Can a verifier-grounded LLM loop speed up a small local model on a RAM-only VPS
— without degrading quality?** This example applies Crucible's thesis (the verifier,
not the model's confidence, decides what's real) to a new artifact: the *inference
runtime* of `Qwen3.5-9B-Q4_K_M` running on CPU via `llama-server`.

The 9B is passive cargo. A Gemini worker edits the inference `config`; a deterministic
`SpeedQualityVerifier` measures single-stream + aggregate tok/s with a **lossless
quality gate**; a Gemini Pro advisor shepherds on plateau. Target: 2.6 → 30 tok/s,
losslessly.

## What it proved

**The loop works and finds real, lossless, integrity-clean speedups** — and then
**plateaus honestly at the config-only bandwidth ceiling**, far below 30. That
plateau is the result (the Sidon-74 analog: an honest lower bound, not a gamed 30).

| metric | baseline | best (search) | true ceiling (sweep) |
|---|---|---|---|
| single_stream | 2.13 tok/s | 3.81 tok/s (+79%) | ~3.7 tok/s (bandwidth wall) |
| aggregate | 1.96 tok/s | 5.10 tok/s (+160%) | ~7.0 tok/s (batched wall) |
| quality | — | lossless | lossless |
| model-integrity | — | correct 9B | correct 9B |

The 30 target is **not reachable by config tuning alone**. The concurrency sweep
(`sweep_concurrency.py`) shows why: aggregate scales with batching up to n_concurrent=8
(6.94 tok/s) then **flattens at n=12 (6.86)** — the memory-bandwidth ceiling for
batched decode. Single-stream walls at ~3.7. To reach 30 you must **reduce
bytes-read-per-token** (speculative decoding, dynamic precision, activation
sparsity) — the Wave 2/3 levers — and the verifier gates any quality trade.

## How it works

```
examples/inference_speed/
  harness.py            # FROZEN: Config, tok/s, measure single/aggregate, lossless
                        #   gate, launch llama-server, httpx streamer, run_harness.
                        #   The measurement methodology is immutable (can't be gamed).
  config.py             # EDITABLE region "config": the only surface Gemini edits.
  strategy.py           # FROZEN passthrough (Wave 2 opens this for a batching proxy /
                        #   spec-decoding orchestration / prefix cache).
  check_draft_compat.py # Gates spec-decoding drafts on tokenizer equality.
  speed_verifier.py     # SpeedQualityVerifier — deterministic, stateless.
  workload/             # FROZEN single/aggregate/probe prompt sets.
  quality/corpus.txt    # FROZEN held-out text (for Wave 2 PPL).
  baseline.json         # FROZEN Wave 0 baseline (single 2.13, aggregate 1.96).
  measure_baseline.py   # Produce baseline.json (Wave 0, no Gemini spend).
  sweep_concurrency.py  # Map n_concurrent -> tok/s (no Gemini spend).
  run_speed.py          # Wave 1 CLI: wires Crucible search + advisor.
  speed.db              # append-only provenance (gitignored).
```

### The verifier (`SpeedQualityVerifier`)

Stateless, `deterministic=True`. For each candidate config:

1. **Hole check** — `Partial` if the `config` region contains a `crucible:hole` sentinel.
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
# Prereqs: GOOGLE_API_KEY in .env; uv pip install -e ".[dev,gemini,inference]"
# plus google-genai and python-dotenv (the gemini extra's package name is a mismatch
# with the code, which imports google.genai — see "Known gotchas" below).

# Wave 0 — honest baseline (no Gemini spend; ~3 min):
uv run python examples/inference_speed/measure_baseline.py --max-tokens 32

# Concurrency sweep — the bandwidth ceiling (no Gemini spend; ~6 min):
uv run python examples/inference_speed/sweep_concurrency.py

# Wave 1 — the search (Gemini spend):
uv run python examples/inference_speed/run_speed.py \
    --workers 4 --episodes 6 --edits 15 --turns 8 \
    --model gemini-2.5-flash --advisor gemini-2.5-pro

# Inspect the reasoning trail:
uv run crucible reasoning --db examples/inference_speed/speed.db
```

## Findings (honest)

1. **The verifier-grounded loop finds real lossless speedups.** In one episode, one
   Gemini Flash worker raised `n_concurrent` 1 → 8 → aggregate 1.96 → 5.16 tok/s,
   single 2.13 → 3.76, quality lossless, model-integrity clean. The full search
   (4 workers × 6 episodes + Pro advisor) confirmed the regime.

2. **Config tuning plateaus at the bandwidth ceiling.** The sweep shows aggregate
   flattening at n_concurrent ≥ 8 (~7 tok/s) and single-stream walling at ~3.7. The
   4-vs-8-vs-12 aggregate (5.41 / 6.94 / 6.86) is the bandwidth wall, not a knob we
   haven't turned. `--parallel` beyond the bandwidth ceiling buys nothing.

3. **30 is a Wave 2/3 problem, not a config problem.** Reaching 30 aggregate needs
   ~4× beyond the batched ceiling, which means **reducing bytes/token**: speculative
   decoding (lossless, needs a tokenizer-compatible draft — `Agents-A1` is
   unverified; `qwen2.5:3b` is the same-family candidate), dynamic per-layer
   quantization (trades quality — the verifier gates it), or activation sparsity.
   The Wave 1 search did **not** try a draft model — `draft_model=None` in every
   attempt — so the single-stream lossless lever is unexplored. Wave 2 opens
   `strategy.py` for a spec-decoding harness and tells the worker which drafts are
   available.

4. **The verifier is the product.** It reported an honest ~7, refused to inflate it,
   and would have rejected any quality regression. Compare the chem ladder
   (`examples/chem/`) where a weak verifier got Goodharted — here the immutable
   measurement methodology + lossless gate + model-integrity check prevent that.

## Known gotchas (integration)

- `llama-server --flash-attn` takes a value (`on|off|auto`); the bare flag consumes
  the next arg and fatal-exits.
- `--draft-max` was removed in current llama.cpp → use `--spec-draft-n-max`.
- `Task.from_path(dir)` includes `__pycache__/*.pyc` (binary) and `speed.db` →
  `UnicodeDecodeError`; `run_speed.py` builds a curated file list instead.
- Qwen3.5 is a reasoning model: stream `chat_template_kwargs.enable_thinking=false`
  or it emits `reasoning_content` and a small `max_tokens` never reaches an answer.
- The pyproject `gemini` extra installs `google-generativeai`, but the provider code
  imports `google.genai` (the `google-genai` package). Install `google-genai` too.

## Non-goals

No training/RL/fine-tuning, no GPU, no kernel patching (v0.5 horizon), no quality
degradation accepted. The 9B is cargo; only its runtime config is searched.