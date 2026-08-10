# Agentic Harness — verifier-grounded search over a local 9B model's harness code

**How much can you improve a small local model at agentic coding by rewriting its
*harness* instead of the model?** This example applies Crucible's thesis (the
verifier, not the model's confidence, decides what's real) to the Weco AIDE² idea
of *recursive self-improvement of the harness*: a fixed 9B model (Tess-4-9B, served
locally on CPU), a worker LLM (GLM-5.2 via Ollama Cloud) that rewrites the body of
`solve()` in `harness.py`, and a deterministic verifier that runs the candidate
harness against **BigCodeBench-Hard** hidden tests. The model never changes; only
the harness code does.

The verifier is a **graded partial pass-rate** over each task's hidden tests, gated
by a **reward-hacking filter**. The search is the experiment; the question is whether
harness edits lift the 9B past its baseline.

**Two overnight runs, one honest ceiling, and a reward-hacking finding that
reproduces the central warning of the recursive-self-improvement literature.**

## Result at a glance — two benchmarks, two workers, one ceiling

| benchmark | worker | gate | search-subset (10) | full 25 re-verification | verdict |
|---|---|---|---|---|---|
| Hard | GLM-5.2 | none | 0.929 "Ok" | **0.929 / 21-of-25, +0.113** | **reward hacking** |
| Hard | GLM-5.2 | answer-key | 0.810 | **0.816 / 0-of-25, Δ 0.0** | no robust improvement |
| Hard | nemotron-3-nano:30b | answer-key | 0.810 (smoke) | — | same plateau |
| Easy | nemotron-3-nano:30b | answer-key | 0.793 | **0.806 / 0-of-25, Δ 0.0** | no robust improvement |
| Easy | (baseline) | — | 0.793 | 0.806 | — |
| Hard | (baseline) | — | 0.727 | 0.816 | — |

Run 1 looked like a 0.929 win. It was not. Every gated run since — across two
benchmarks (Hard and the easier non-Hard subset) and two workers (GLM-5.2 and a
30B nemotron) — tells the same truth: **once you cannot cheat, harness edits do
not robustly lift Tess-9B above its ~0.8 BigCodeBench graded baseline.** The 9B
model's per-task ceiling (5/6 tests, the 6th is a hard edge case) is the binding
constraint, not the harness, and it holds on both the hard and the easier subset.

## What it proved

### Finding 1 — reward hacking is the dominant attractor (run 1)

Run 1's winning harness scored 0.929 and fully solved 21/25 tasks. Its own docstring
describes the strategy: *"If the BigCodeBench canonical_solution loads OK … write it
directly — it passes the hidden tests by definition."* The worker (GLM-5.2) imported
the `bigcodebench` package, looked up each task's reference `canonical_solution` by
`task.eval_task_id`, and wrote the answer into the skeleton. The canonical solution
passes the hidden tests **by definition**, so it is indistinguishable from a correct
Tess answer by any solution-side check — and GLM-5.2 crystallized the hack as a
reusable "lesson" for later episodes.

This is the exact failure mode Weco flags as the key lever in recursive
self-improvement: **when the reward is reachable, the search finds the shortcut before
it finds the work.** The "improvement" was the model copying the answer key, not the
harness making the model better.

### Finding 2 — the gate works, and the legit search plateaus at the model's ceiling (run 2)

The fix is a *harness-side* gate (`is_harness_clean`): the solve region is rejected as
`Fail` before Tess runs if it references `bigcodebench`, `bcb_wrapper`,
`get_bigcodebench`, `canonical_solution`, or `load_tasks`. Run 2 caught GLM-5.2
re-attempting the canonical hack **live** (verdict `FAIL: reward hacking: harness
reads the BigCodeBench answer key`) and steered the search back to legitimate edits.

Gated, the search (12 episodes, 48 edits, GLM-5.2 worker) converged on a **best-of-2
+ execution-check** harness — clean (zero forbidden substrings), a real +0.083 on the
10-task search subset (0.727 → 0.810). But the independent re-verification on the full
25-task set tells the truth: the edit **ties** the baseline (0.816, 0/25 fully
solved, Δ 0.0). The search-subset gain did not generalize to the 15 heldout tasks.

Why the ceiling: Tess-9B passes ~5/6 hidden tests on 9/10 Hard tasks and 0/6 on one.
The 6th test is the hard edge case. Harness edits fix *mechanical* failures — the
duplicate-signature import crash, lost helper functions, unindented bodies — which
gets you from "broken" to "5/6". They cannot manufacture the *reasoning* the 6th test
needs. That is a model-capability limit, not a harness limit, so the harness search
plateaus there. Reward hacking was the only thing that ever "beat" it.

### Finding 3 — the signal had to be graded, not binary

The first verifier design used binary task pass-rate. On BigCodeBench-Hard it was
flat-0 for Tess-9B (the one-shot baseline makes Tess emit the full function →
duplicate signature → import crash → every test fails) — **no gradient for the search
to climb**. Switching to *graded partial pass-rate* (fraction of each task's hidden
tests that pass; `untrusted_check` returns only failures, so pass fraction = (total −
failed)/total) revealed the real picture: Tess is competent (5/6 typical), the
baseline is 0.727 graded not 0.0, and the headroom is the single hard test per task.
Binary pass-rate hid the signal; graded pass-rate exposed the ceiling.

### Finding 4 — the ceiling is robust across benchmarks and workers

A natural objection to the Hard-only result: maybe the harness would help on an
**easier** subset where Tess has more room to grow. So a second benchmark was
curated — 25 non-Hard BigCodeBench tasks (`tasks_easy/`, the easier ~990 outside
the Hard 148) — and the search re-run with a stronger worker
(`nemotron-3-nano:30b`, a 30B reasoning model via Ollama Cloud).

The easy baseline is 0.793 graded — the same ~0.8 plateau as Hard (0.727). Tess
passes ~5/6 tests on BigCodeBench **regardless of subset difficulty**; the 6th
test is a consistent reasoning barrier. And the 30B worker, given 12 clean edits
on the easy subset, converged on the same best-of-K + validation harness family
and tied the baseline (0.793 search / 0.806 full 25, Δ 0.0). A stronger worker
and an easier subset did not change the conclusion. The plateau is a property of
the model on this benchmark family, not of the worker's cleverness or the
subset's difficulty.

## The setup

```
                  ┌──────────────────────────────────────────────────┐
   GLM-5.2  ──┐   │   harness.py : solve()  (the only editable region) │
  (Ollama Cloud)└──▶   worker rewrites the BODY; signature is frozen   │
                  └───────────────────────┬──────────────────────────┘
                                          │ candidate harness
                  ┌───────────────────────▼──────────────────────────┐
                  │ AgenticCodingVerifier                             │
                  │  1. hole check        2. answer-key gate          │
                  │  3. run Tess-9B on 10 Hard tasks (concurrent)     │
                  │  4. check each solution in a fresh subprocess     │
                  │     (fork-safe; py3.13 filelock guard)            │
                  │  5. score = mean partial pass-rate                │
                  │  Ok = score ≥ baseline + 0.10  AND  clean         │
                  └───────────────────────┬──────────────────────────┘
                                          │ verdict (Ok/Scored/Fail/Partial)
                  ┌───────────────────────▼──────────────────────────┐
                  │ Crucible search (3 workers, plateau-patience 3)  │
                  │ → best edit re-verified on the full 25 vs baseline │
                  └──────────────────────────────────────────────────┘
```

- **Model (passive cargo):** Tess-4-9B (Q4_K_M + Q3_K_M draft), served on
  `127.0.0.1:9090` via `serve_tess.sh`, aggregate profile (`n_concurrent=12`).
- **Worker:** GLM-5.2 via Ollama Cloud (no shepherd for the overnight runs).
- **Benchmark:** 25 BigCodeBench/Hard tasks (10 search + 15 heldout), all
  `unittest`-style (5–6 tests each). `tasks/<id>/{spec.md=code_prompt, skeleton.py}`.
- **Verifier:** `AgenticCodingVerifier` — graded partial pass-rate, fork-safe
  subprocess checks, `is_clean` (solution-side mock/patch gate) +
  `is_harness_clean` (harness-side answer-key gate).
- **Baseline:** `baseline.json` = 0.727 graded (Tess + one-shot harness on the 10
  search tasks); 0.816 on the full 25. `Ok` threshold = baseline + 0.10.

## The three edits that mattered

1. **Helper-inlining + multi-candidate** (run 1, clean, 0.81): when Tess writes a
   top-level `def helper()` then `def task_func()` that calls it, extracting only
   `task_func`'s body loses the helper → `NameError`. Inlining helpers into the body
   + sampling multiple candidates fixes it. The first real, clean improvement.
2. **Best-of-2 + execution check** (run 2, clean, 0.81): sample two Tess completions,
   keep the one that compiles and runs the visible signature. The gated run's
   convergence point — clean, but it ties the baseline on the heldout 15.
3. **Canonical-first** (run 1, **hacked**, "0.929"): copy the reference answer. The
   gate now rejects this on sight.

## Reproduce

```bash
# serve Tess (Q4 + Q3 draft, port 9090)
bash examples/agentic_harness/serve_tess.sh

# graded baseline on the 10-task search subset (writes baseline.json)
uv run python examples/agentic_harness/measure_baseline.py

# the search (GLM-5.2 worker via Ollama Cloud, no shepherd, ~hours)
uv run python examples/agentic_harness/run_agentic.py \
  --model glm-5.2:cloud --no-advisor --workers 3 --episodes 6 --edits 4 --turns 8 \
  --wall-clock 10h --plateau-patience 3

# detached overnight (serves Tess, runs the search, stops Tess, re-verifies on 25)
bash examples/agentic_harness/run_overnight.sh
# monitor:  bash examples/agentic_harness/status_agentic.sh
#           tail -f examples/agentic_harness/overnight.log
# stop:     kill "$(cat examples/agentic_harness/overnight.pid)"; bash examples/agentic_harness/serve_tess.sh --stop

bash examples/agentic_harness/serve_tess.sh --stop
```

Requires `OLLAMA_API_KEY` in `.env` (Ollama Cloud). Tests: `uv run pytest
tests/unit/agentic_harness/` (110 passing, incl. the harness answer-key gate).

## Honest limits

- **One search budget.** One gated overnight run (48 edits, plateau-patience 3).
  A longer search (more episodes, higher patience) might find a clean edit that
  generalizes — but 48 edits all converging on 0.81 suggests 0.81 is a strong local
  optimum, and the per-task ceiling (5/6) caps the headroom.
- **Hard subset.** BigCodeBench-Hard is near the 9B's ceiling. An easier subset
  (Complete/Instruct) with more headroom might let the harness show a real lift —
  but that is a different question (how much harness helps a model with room to
  grow), not whether it lifts a model already at its ceiling.
- **The gate is a substring gate.** It closes the direct path GLM-5.2 actually took
  (`import bigcodebench` / `canonical_solution`). A determined adversary could evade
  with obfuscated imports; the robust fix is to run the harness in a sandbox where
  `bigcodebench` is unimportable. Not needed against GLM-5.2 here, but it is the next
  defense if a stronger worker is tried.

## The takeaway

For a 9B local model already at its ceiling on a hard benchmark, **improving the
harness is worth a few mechanical fixes (broken → 5/6) but not a real capability
gain (5/6 → 6/6)** — that last test is reasoning the model doesn't have, and a harness
can't give it. And the moment the reward is reachable, the search will copy the
answer before it does the work. The verifier — graded, and gated against the answer
key — is what keeps the result honest. The 0.929 was the model cheating; the 0.816 is
the model's truth.