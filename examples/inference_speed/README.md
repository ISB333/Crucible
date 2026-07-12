# Inference Speed — verifier-grounded search over a local model on CPU

**Can a verifier-grounded LLM loop speed up a small local model on a RAM-only VPS
— without degrading quality?** This example applies Crucible's thesis (the verifier,
not the model's confidence, decides what's real) to a new artifact: the *inference
runtime* of a local model running on CPU via `llama-server`.

The model is passive cargo. A Gemini worker edits the inference `config`; a
deterministic `SpeedQualityVerifier` measures single-stream + aggregate tok/s with
a **lossless quality gate**; a GLM-5.2 advisor (via Ollama Cloud) shepherds on
plateau. Target: 2.6 → 30 tok/s, losslessly.

## What it proved

Two waves, two honest ceilings, one real lever.

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

### The honest bottom line

**30 tok/s aggregate is beyond this VPS's shared memory bandwidth for both models**
(config tuning + self-spec: dense 9B ~7.5, LFM2.5 MoE ~20). The verifier-grounded
search characterized both ceilings honestly and surfaced the architectural lever —
the Sidon-74 pattern: an honest lower bound, not a gamed 30. The verifier reported
real ~7.5 / ~20, refused to inflate them, and would have rejected any quality
regression (compare the chem ladder, where a weak verifier got Goodharted).

Reaching 30 would need either a smaller-draft-compatible architecture, Wave 3
levers (activation sparsity within the MoE, dynamic per-layer precision — gated by
the verifier for quality), or more VPS memory bandwidth.

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
# Prereqs: GOOGLE_API_KEY + OLLAMA_API_KEY + OLLAMA_MODEL (e.g. glm-5.2:cloud) in .env;
# uv pip install -e ".[dev,gemini,inference]" plus google-genai and python-dotenv.

# Wave 0 — honest baseline (no Gemini spend; ~3 min):
uv run python examples/inference_speed/measure_baseline.py --max-tokens 32

# Concurrency sweep — the dense 9B's bandwidth ceiling (no Gemini spend; ~6 min):
uv run python examples/inference_speed/sweep_concurrency.py

# Wave 2 — the search (Gemini + Ollama Cloud spend). Defaults: worker
# gemini-3.1-flash-lite, advisor $OLLAMA_MODEL (glm-5.2:cloud), Q2_K self-spec draft seeded.
uv run python examples/inference_speed/run_speed.py \
    --workers 4 --episodes 6 --edits 15 --turns 8
# (override: --model gemini-3.1-flash-lite --advisor glm-5.2:cloud)

# Inspect the reasoning trail:
uv run crucible reasoning --db examples/inference_speed/speed.db
```

The Q2_K/Q3_K_M self-spec drafts and the LFM2.5-8B-A1B target were benchmarked
out-of-band (see "What it proved"); they live on disk at `/home/isb/models/`.

## Findings (honest) — see "What it proved" above for the consolidated result

The four Wave 1 findings hold: (1) the loop finds real lossless speedups; (2)
config tuning plateaus at the bandwidth ceiling (sweep: 5.41 / 6.94 / 6.86 at
n=4/8/12); (3) 30 needs reducing bytes/token — Wave 2 confirmed spec decoding is
only ~+10% on a dense model and the architecture (MoE) is the real lever; (4) the
verifier is the product — it reported an honest ~7.5 / ~20, refused to inflate it.

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

## Non-goals

No training/RL/fine-tuning, no GPU, no kernel patching (v0.5 horizon), no quality
degradation accepted. The target model is cargo; only its runtime config is searched.