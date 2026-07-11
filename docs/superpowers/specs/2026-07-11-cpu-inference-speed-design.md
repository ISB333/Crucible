# CPU Inference Speed — Verifier-Grounded Search over a Local 9B

**Date:** 2026-07-11
**Status:** Spec-locked (user-approved Approach B, review gate waived)
**Location:** `examples/inference_speed/` (a new Crucible example, sibling of `sidon/` and `chem/`)

## 1. The experiment

Apply Crucible's thesis — *the verifier, not the model's confidence, decides what's real* —
to a different artifact: **the inference runtime of a small local model on a RAM-only VPS.**

- **Artifact being optimized (passive):** `Qwen3.5-9B-Q4_K_M` running on CPU via
  `llama-server` (llama.cpp). The 9B does no drafting, no verifying — it is the thing we
  make fast.
- **Worker / shepherd (active):** Gemini, via the user's `GOOGLE_API_KEY`. A cheap Gemini
  (Flash) does the bulk search; a strong Gemini (Pro) shepherds on plateau. Gemini proposes
  edits to the inference configuration and runtime harness.
- **Verifier (the product):** `SpeedQualityVerifier` — deterministic. Runs a fixed coding-
  agent workload through the candidate harness, measures **single-stream tok/s** and
  **aggregate tok/s** (across N concurrent workers), and checks **no quality loss** vs a
  frozen baseline. An optimization is a *win* only if it is faster **and** quality-clean.
- **Memory:** `examples/inference_speed/speed.db` — append-only SQLite. Every attempt's
  config, both tok/s metrics, quality verdict, and Gemini's reasoning are recorded and
  replayable via `crucible reasoning`. Lessons carry across episodes.
- **Target:** 2.6 → **30 tok/s**, deliberately ambitious to force the search toward
  structural (not flag-tuning) wins.

This is Crucible with the inference runtime as the editable artifact. The recursion is the
headline: *Crucible optimizing the inference runtime that Crucible (and the user's coding
agent) runs on.*

## 2. Why this is faithful to the two papers

- **AlphaProof Nexus (the engine):** LLM edits an artifact-with-holes; a verifier checks
  every edit; the verdict is the search gradient. Here the artifact is `harness.py` + editable
  `config`/`strategy` regions; the verifier runs it and returns `Scored(tok/s)` / `Ok` / `Fail`.
  The first-wins, no-shared-state orchestrator (Agent A) is reused unchanged.
- **CategoryScienceClaw (the type-system / the gate is the product):** the gate decides
  whether "what you got" equals "what you asked for." Here the gate enforces *faster AND no
  quality loss* — and, critically, an **immutable measurement methodology** so tok/s cannot
  be Goodharted (the chem-ladder failure mode, reproduced in this very repo). The progression
  search → optimization → discovery maps to: Wave 1 config search → Wave 2 harness
  optimization → v0.5 kernel discovery.

## 3. The honest physics read (evidence before assumptions)

Measured on this VPS (research-first, not assumed):

- CPU: AMD EPYC-Genoa, 12 vCPUs (12 sockets × 1 core × 1 thread), single NUMA node.
- SIMD: full AVX-512 (F/DQ/BW/VL/CD/IFMA/VBMI/VNNI/BF16). No AMX (Intel-only). VNNI+BF16
  are excellent for quantized inference.
- RAM: 31 GB total, **~9 GB free** (a Dify/ollama stack is running and consumes ~21 GB).
  No swap. The 9B-Q4 (5.5 GB) + a draft (0.3–1.9 GB) + KV cache fits, but tightly — the
  other stack must be stopped during measurement runs.
- Models on disk: `Qwen3.5-9B-Q4_K_M.gguf` (5.9 GB, the target), `Agents-A1-Q4_K_M.gguf`
  (533 MB, a ~0.5B draft candidate), `qwen2.5:3b` in ollama (1.9 GB, a 3B same-family draft
  candidate).
- `llama-server` installed at `/usr/local/bin/llama-server`; llama.cpp source at
  `/home/isb/llama.cpp` (available for the v0.5 kernel horizon).

**Bandwidth ceiling:** 9B-Q4 reads ~5.5 GB/token. Contabo EPYC-Genoa VPS memory bandwidth is
~13–16 GB/s (virtualized, shared) → 16/5.5 ≈ 2.9 tok/s. The observed 2.6 tok/s is at this
ceiling — it is **not a config bug**, it is the physics of full-forward-pass-per-token.

Consequences:
- Kernel/NUMA tuning alone (GPT doc "Niveau 1") reaches maybe ~3.5 tok/s. Not 30.
- **30 single-stream is physics-breaking without quality loss** — it almost certainly needs
  kernel-level work (Approach C, v0.5) or a quality trade the verifier rejects. The
  single-stream leaderboard will show an honest plateau (a Sidon-74-style useful result).
- **30 aggregate is ambitious-but-reachable**: batching N concurrent decodes amortizes the
  one weight-read across the batch (lossless, ~N×), and speculative decoding stacks a further
  ~K× (lossless). Batch 8 × 2.6 ≈ 21, + spec decoding → 30+ is in reach.

The two metrics are tracked **separately** (user decision): the aggregate leaderboard can
hit 30; the single-stream leaderboard documents an honest lower bound. A win on either
(with quality clean) is recorded.

## 4. Architecture

### 4.1 The editable artifact — `harness.py` + holes

```
examples/inference_speed/
  harness.py            # FROZEN scaffolding: load config, launch server, run workload,
                        #   compute tok/s from real timestamps, run quality gate, print JSON.
                        #   The measurement methodology lives here and is immutable.
  config.py             # EDITABLE region "config": llama-server launch params.
  strategy.py           # EDITABLE region "strategy": runtime proxy logic (Wave 2+).
  quality/
    corpus.txt          # FROZEN held-out PPL corpus (immutable, outside editable regions).
    katas/              # FROZEN coding kata suite with tests (immutable).
  workload/
    prompts_single.jsonl   # FROZEN single-stream prompt set.
    prompts_aggregate.jsonl# FROZEN N-concurrent coding-agent prompts.
  speed_verifier.py     # SpeedQualityVerifier (deterministic).
  baseline.json         # FROZEN baseline measurements (single-stream, aggregate, PPL, kata).
  run_speed.py          # CLI entry, mirrors run_sidon.py.
  speed.db              # append-only provenance (gitignored).
```

Editable regions (Crucible `# crucible:region start name=... / end`):

- **`config` (Wave 1 +):** `model_path`, `draft_model_path`, `n_threads`, `n_batch`,
  `n_ubatch`, `flash_attn`, `mlock`, `use_mmap`, `ctx_size`, `draft_max`, `cache_policy`,
  `quant_override`, `n_concurrent`. A pure data region — Gemini edits values, not logic.
- **`strategy` (Wave 2 +):** a Python callable `build_strategy() -> Strategy` returning an
  object the frozen harness calls. `Strategy` exposes hooks: `prefill_batch(prompts)`,
  `decode_step()`, `should_verify(token)`. Wave 1 ships a frozen passthrough `Strategy` so
  the harness runs unchanged; Wave 2 opens the region and Gemini implements the batching
  multiplexer, speculative-decoding orchestration, and prefix/activation cache here.

The frozen `harness.py` is the integrity anchor: it computes tok/s from real token
timestamps emitted by the server, runs the fixed workload, and invokes the quality gate.
Gemini never touches it. This is the chem-ladder lesson made structural — *the measurement
code is immutable, so the measure cannot be Goodharted.*

### 4.2 The verifier — `SpeedQualityVerifier`

Frozen dataclass, `deterministic = True`. `verify(artifact, ctx) -> Verdict`:

1. `scan_holes` → `Partial` if `config` (or, in Wave 2+, `strategy`) is unfilled.
2. `ctx.materialize(artifact)` → workspace. Launch `llama-server` with the candidate
   `config` via `ctx.sandbox.run` (SubprocessSandbox — no Docker, like `sidon/`).
3. **Single-stream measurement:** stream the `prompts_single` set greedily (temperature 0)
   through the harness; compute tok/s from real token-arrival timestamps.
4. **Aggregate measurement:** fire the `prompts_aggregate` set as N concurrent requests
   (N = `n_concurrent` from config); compute aggregate tok/s = total tokens / wall clock.
5. **Quality gate** (the no-degradation contract):
   - **Default path:** PPL on `quality/corpus.txt` vs `baseline.json["ppl"]` (accept if
     `ppl ≤ baseline * 1.005` — 0.5% relative tolerance, to absorb measurement noise), **and**
     kata pass-rate on `quality/katas/` vs `baseline.json["kata_pass"]` (accept only if
     `≥ baseline` — any kata regression fails; no tolerance on coding correctness). Both
     deterministic (greedy decoding, fixed seed).
   - **Lossless-by-construction exemption:** if `strategy.declares_lossless` is true
     (batching, prefix-cache, speculative decoding — pure amortization / exact
     acceptance), the verifier skips PPL+kata and instead checks the strategy's
     *correctness invariant*: for spec decoding, that accepted tokens equal the greedy
     reference on a probe set; for batching/cache, that outputs are byte-identical to the
     single-stream reference. This mirrors Crucible adapting the verifier to the artifact
     class, and avoids burning budget re-checking mathematically-lossless changes.
6. **Verdict:**
   - `Fail` — crash, or quality regressed (default path), or lossless invariant violated
     (exempt path), or aggregate tok/s did not beat the incumbent best by ≥5% (anti-noise
     margin; a 4% wobble is measurement noise, not a win).
   - `Scored(value=aggregate_tok/s, feedback="single=X.X agg=Y.Y ppl=… kata=… lossless=…")`
     — ran and quality-clean. The orchestrator ranks by `value` (aggregate); single-stream
     and quality are carried in feedback and logged to the DB for the separate leaderboard.
   - `Ok` — `aggregate ≥ 30` **and** `single_stream ≥ 8` **and** quality-clean. The 8
     single-stream floor is an aspirational-but-not-physics-breaking stretch; hitting it
     likely needs Wave 2 + shepherd ideas. Pure `Ok` is the 30-aggregate prize.

### 4.3 The integrity gate (Crucible's 4 conditions, mapped)

1. **Verify OK** — `SpeedQualityVerifier` returns `Ok` (or `Scored` for the leaderboard).
2. **Immutable spec untouched** — `harness.py`, `quality/`, `workload/`, `baseline.json`
   are byte-identical to frozen originals. The editable regions are `config` and `strategy`
   only. *This is what prevents gaming the tok/s measure* — the chem-ladder antidote.
3. **No escape tokens** — deny-list scans `config`/`strategy` for gaming vectors:
   `time.perf_counter`/`time.monotonic` overrides, monkeypatching the timer, printing
   fabricated tok/s, editing the workload/corpus, disabling the quality gate, swapping the
   model path to a tiny destructively-quantized model while claiming "no quality loss",
   `pytest.skip`/`# type: ignore` in katas. The harness *computes* metrics from real
   signals, never from what the config claims.
4. **Fresh re-verify** — any `Ok`/best-`Scored` is re-run in a brand-new sandbox with a
   cold server start; both tok/s and quality must reproduce.

### 4.4 Shepherding (the plateau-breaker)

- Worker: `gemini-2.5-flash` — cheap, fast, proposes `config`/`strategy` edits per episode.
- Advisor: `gemini-2.5-pro` (or `gemini-3-pro` if available) — strong, proposes structural
  ideas when the aggregate tok/s stalls for `fail_streak` episodes (plateau detection).
- `AdvisorPolicy(model=<advisor>, max_calls_per_run=8, fail_streak=3, plateau_trigger=True)`.
- Graceful degradation inherited from Crucible: if Gemini is unavailable, the worker
  proceeds alone and the run does not fail.

### 4.5 Verify-cost engineering constraint

Launching the 9B takes ~10–20s cold. Many `config` edits (threads, batch, draft model,
quant) require a server restart → ~20–30s per Wave-1 attempt → ~120 attempts/hour, fine
for a multi-hour Flash search. `strategy` edits (Wave 2+) run on top of a warm, stable
server → faster verify. **Mitigation:** warm-reuse a server across attempts where the
config diff is restart-free (flash-attn, cache flags, concurrency); cold-restart only when
a restart-required param changes, and always for fresh re-verify of a claimed win.

## 5. Waves (the execution order; detailed in the implementation plan)

> **Risk to resolve in Wave 0 (before assuming spec decoding works):** speculative decoding
> requires the draft and target to share a tokenizer (token IDs must align). `Qwen3.5-9B`'s
> tokenizer must be checked against each draft candidate — `qwen2.5:3b` (likely compatible
> within the Qwen family, but not guaranteed across major versions) and `Agents-A1`
> (tokenizer unknown — may be unusable as a Qwen draft). Wave 0 verifies tokenizer
> compatibility and records which drafts are viable; an incompatible draft is a `Fail`, not a
> crash. If no on-disk draft is compatible, the plan falls back to self-speculative decoding
> (a Q2/Q3 quantized copy of the 9B as its own draft — same tokenizer by construction) at the
> cost of extra RAM.

- **Wave 0 — Honest baseline.** Measure the 9B with a well-tuned default config (12 threads,
  flash-attn, mmap) → record `baseline.json` (single-stream, aggregate, PPL, kata). This is
  the verifier's comparison anchor. *Evidence before assumptions* — the baseline may already
  be 3–4 tok/s, not 2.6.
- **Wave 1 — Config-only search (cheap, seeds DB).** `config` editable, `strategy` frozen
  passthrough. Gemini tunes flags: spec decoding via `draft_model` (try `Agents-A1` 0.5B and
  `qwen2.5:3b`), `n_concurrent` for aggregate, prefix cache, batch sizes. Establishes what
  flag-tuning alone reaches (bandwidth-wall single-stream + good aggregate via batching).
- **Wave 2 — Harness editable (the structural wins).** Open `strategy`. Gemini implements
  the batching multiplexer, speculative-decoding orchestration, prefix/activation cache in
  Python. Lossless wins compound toward 30 aggregate.
- **Wave 3 — Shepherd escalation.** On plateau, the strong advisor proposes bigger ideas:
  uncertainty-gated verification, dynamic per-layer quant, activation sparsity. These trade
  quality → the verifier checks. Target: 30 aggregate confirmed; single-stream honest
  plateau documented.
- **v0.5 horizon (Approach C, deferred).** Kernel patches to llama.cpp (custom sparsity /
  AVX-512 / mixed-Q2/Q4 kernels). Where 30 single-stream probably lives. Out of scope for the
  first experiment; the shepherd may *propose* kernel ideas, which are recorded as lessons
  for the v0.5 follow-up.

## 6. Success criteria

- **Primary:** an `Ok` verdict with `aggregate ≥ 30 tok/s` and quality clean (re-verified
  fresh). If only `Scored` is reached, the best aggregate with quality clean is the result.
- **Secondary (the honest-plateau result):** the maximum single-stream tok/s achieved with
  quality clean, documented as a lower bound — the Sidon-74 analog. Reaching 30 single-stream
  without quality loss is *not* expected in-scope; if it happens, it is a headline.
- **Integrity:** every claimed number reproduces under fresh re-verify; the integrity gate
  rejects any attempt that edits the measurement code, the workload, or the quality corpus.
- **Provenance:** the full search (every Gemini proposal, measured tok/s both ways, quality
  verdict, lessons) is replayable via `crucible reasoning --db examples/inference_speed/speed.db`.

## 7. Non-goals (YAGNI)

- No training, no fine-tuning, no RL, no distillation of the 9B.
- No GPU work (RAM-only VPS).
- No kernel patching in scope (v0.5 horizon).
- No new engine — reuse llama.cpp / `llama-server` and Crucible's own engine.
- No quality degradation accepted; the verifier is the product.