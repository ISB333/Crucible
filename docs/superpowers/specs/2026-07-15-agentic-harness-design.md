# Agentic Harness — verifier-grounded search over the harness of a fixed 9B

**Date:** 2026-07-15
**Status:** approved design (spec)
**Predecessor:** `2026-07-11-cpu-inference-speed-design.md` (same Crucible loop, different artifact + verifier)

## 1. Thesis

A fixed local 9B (Tess-4-9B, served) is **passive cargo**. The artifact being
searched is its **harness code** — the agentic loop that drives the model on coding
tasks (Weco AIDE²: the outer loop rewrites the inner-loop agent's harness code, on
fixed models, no weight updates). Crucible's loop is the outer loop. The verifier is
**BigCodeBench execution** (deterministic, hidden tests = the anti-reward-hacking
moat). This is the KERYKEION "long-term feedback loop" made literal: the search
improves the thing that improves the model.

> The verifier — not the model's confidence — decides what's real. Without a tight
> reward-hacking gate the worker "finds" a harness that passes 100% by cheating; the
> gate is the whole point (Weco's #1 lever, KERYKEION P4).

## 2. Measurable requirements

| id | requirement | measurable as |
|---|---|---|
| R1 | A baseline pass rate for Tess + a minimal harness on the subset | `baseline.json`: pass_rate on 25 tasks |
| R2 | The search improves pass rate over baseline | best harness pass_rate − baseline ≥ **+10 absolute** → `Ok` |
| R3 | Improvements are not reward-hacking | hidden-test gate + clean-env eval + no-test-edit gate; any trip → `Fail` |
| R4 | The verifier is deterministic | same harness → same verdict (modulo 9B nondeterminism: temperature=0 + greedy; documented) |
| R5 | The verifier is cheap enough to search | search-subset eval ≤ 30 min (10 tasks, capped, batched) |
| R6 | The result is independently re-verified | best harness re-run on the full 25-task subset (incl. the 15 held-out), reported |
| R7 | The harness is model-agnostic | `--agent-base-url` / `--agent-model` flag; default Tess@:9090 |

**Non-goals:** no fine-tuning / weight updates, no GPU, no SWE-bench (later tier), no
quality degradation of the *measurement* (reward-hacking) accepted.

## 3. The contract (data structures / frozen vs editable)

### 3.1 Frozen — `agent_contract.py`

```python
@dataclass(frozen=True)
class Task:
    id: str
    spec: str            # docstring + signature + visible example (what the agent sees)
    skeleton_path: Path  # workdir-relative file to complete
    eval_entry: str      # BigCodeBench task id for the official check

@dataclass(frozen=True)
class LLM:
    base_url: str        # http://127.0.0.1:9090/v1  (Tess)
    model: str
    # .chat(messages, max_tokens, temperature=0.0) -> str  (handles enable_thinking)

@dataclass(frozen=True)
class Tools:
    # bound to the per-task sandbox workdir
    read_file, write_file, list_dir, run_visible_tests

def solve(task: Task, workdir: Path, llm: LLM, tools: Tools) -> None:
    raise NotImplementedError  # the worker replaces this body
```

The primitives (LLM client, tool impls, task loader, test runner) and the `solve`
signature are **frozen** — the worker cannot change the tools, only orchestrate them.

### 3.2 Editable — `harness.py` region `solve`

The worker (Gemini) rewrites the **body of `solve`**: the agentic loop — prompt
strategy, reflection on test failures, retry/search policy, memory of prior failures,
plan-first vs write-then-test, max turns. This is the Weco "inner-loop harness code."

### 3.3 Frozen — `tasks/`

A curated **25-task subset** of BigCodeBench (mix of Complete + Instruct,
self-contained, cheap deps), split into:
- **search subset (10)** — fast ranking during the search,
- **held-out final (15)** — independent re-verification of the best harness.

Per task: `spec.md` (docstring+signature+visible example), `skeleton.py`, and the
BigCodeBench `eval_entry` id (hidden tests live in the BigCodeBench package, never in
the workdir the agent sees).

## 4. The verifier — `AgenticCodingVerifier`

`deterministic=True`, stateless. For a candidate harness:

1. **Per task (concurrent, batched against the served Tess at n_concurrent=12):**
   a. Materialize a **fresh sandbox workdir** (subprocess, clean copy of skeleton).
      Caps: `max_turns=8`, `max_tokens_per_turn=256`, `wall_s=180`.
   b. Call `harness.solve(task, workdir, llm, tools)`. Crash/timeout → task fails
      (partial credit preserved — not a global Fail).
2. **Execution gate:** copy the completed solution into BigCodeBench's **official
   `check`** in a clean env with **hidden tests**. Pass/fail, deterministic.
3. **Reward-hacking gate** (per task, any trip → task fails):
   - Hidden-test files are **outside** the sandbox workdir (agent cannot read them).
   - Solution scanned for test-framework monkeypatching / `sys.modules` tricks.
   - BigCodeBench's `check` guarantees the tests executed the genuine solution.
   - The frozen test/eval files are hash-compared after the run (no in-sandbox edit
     can reach them, but the check is defense-in-depth).
4. **Score:** `Scored(value = pass_rate on the search subset)`.
   `Ok` when `pass_rate ≥ baseline + 10` AND clean; `Fail` on harness
   crashes-everywhere or a global reward-hacking trip.

A module-level lock + the batched Tess server keep eval memory-bounded (one server,
n_concurrent=12). Gemini turns parallelize; the 9B is the serialized resource.

## 5. The feasibility strategy (the part that makes or breaks it)

A 9B agentic solve is expensive (~k tokens/task at ~7 tok/s). Naive full-eval = days
per harness. The verifier must be cheap, like the speed experiment:

- **Serve Tess with the best-aggregate config** (n_concurrent=12, ~10 tok/s —
  `serve_tess.sh`, adapted from `serve_best.sh`) and run the search subset's tasks
  **concurrently** → aggregate throughput.
- **Hard caps per task** (§4.1a) bound each task to a few k tokens ≈ minutes.
- **Tiered eval:** 10-task search subset (strict caps) for ranking; 25-task full
  subset for the independent final re-verification.
- **Honest cadence:** ~30 min/harness-eval (10 tasks × ~2k decode tokens, batched at
  the 9B's ~10 tok/s aggregate ≈ 34 min; BigCodeBench checks are negligible) →
  **~15-20 harness iterations overnight**. Not 100s. The Weco run was 100 steps / 8
  days on frontier models; ours is fewer steps on a 9B. The design is built around
  this budget — episode budgets are small (edits≈4) because each edit is a full eval.

## 6. The loop (orchestration)

```
run_agentic.py
  Task(root=examples/agentic_harness, editable=("solve",), network=True)
  verifier=AgenticCodingVerifier(baseline_path=.../baseline.json)
  model=gemini-3.1-flash-lite            # worker rewrites harness.py:solve
  advisor=glm-5.2:cloud + web_search     # shepherds on plateau, researches techniques
  advisor_factory=WebSearchAdvisor       # agentic (same as inference_speed)
  extra_tools=[web_search]               # worker can research too
  run_budget=RunBudget(episodes_per_worker=6, plateau_patience=3, wall_clock_s=36000)
```

**The recursion (KERYKEION two loops):**
- short-term (intra-task): the harness's own test→retry loop inside `solve`.
- long-term (inter-task): failed-task trajectories feed the shepherd's
  `advisor_consult` ("the harness failed these tasks like this") → GLM-5.2 researches
  a harness technique via web_search → the worker applies it. The search compounds.

## 7. Components — `examples/agentic_harness/`

```
agent_contract.py   # FROZEN: solve() signature, LLM(Tess), Tools, task loader
harness.py          # EDITABLE `solve` region (the worker's harness code)
tasks/              # FROZEN 25-task BigCodeBench subset (10 search + 15 held-out)
agentic_verifier.py # AgenticCodingVerifier: sandbox + run + BigCodeBench check + gate + score
sandbox.py          # per-task subprocess sandbox with caps (reuse SubprocessSandbox)
reward_hacking_gate.py # the anti-Goodhart checks (hidden-test isolation, monkeypatch scan)
run_agentic.py      # CLI: wires Crucible run_search + Gemini worker + GLM shepherd + web_search
baseline.json       # baseline pass rate: Tess + minimal harness
serve_tess.sh       # serve Tess, best-aggregate config, port 9090
status_agentic.sh   # live DB-backed monitor (Crucible prints nothing mid-run)
tests/              # contract tests + a reward-hacking regression test
```

## 8. Constraints

- The 9B served config: best-aggregate (n_concurrent=12, KV q8_0, Q3 draft, etc. from
  the speed experiment) — reuse the lossless runtime config so agentic eval is affordable.
- Tess is a reasoning model → `enable_thinking=false` in the LLM client (or enough
  max_tokens) so it emits answers, not thinking.
- Sandbox: subprocess (not docker) for per-task isolation, provided the eval env is
  clean (BigCodeBench deps installed). Docker is the fallback if isolation proves leaky.
- Reward-hacking gate is **non-optional** (Weco's #1 lever). A regression test must
  prove a cheating harness (edits the frozen test file / hardcodes the visible example
  into a monkeypatch) is rejected.
- Temperature=0 + greedy for determinism; document the residual 9B nondeterminism.
- Spend: Gemini worker + Ollama Cloud shepherd, gated at the point of action (the
  overnight launch). Same `.env` keys as inference_speed.

## 9. Task decomposition (waves)

- **Wave 0 — baseline + harness contract.** `agent_contract.py`, the minimal
  `harness.py` (write-once), `serve_tess.sh`, the 25-task subset curated, BigCodeBench
  check wired. Measure `baseline.json`. No Gemini spend.
- **Wave 1 — verifier + gate.** `AgenticCodingVerifier`, `sandbox.py`,
  `reward_hacking_gate.py`, the reward-hacking regression test. Deterministic unit
  tests with fakes (no 9B); one integration test on Tess.
- **Wave 2 — the search.** `run_agentic.py` (Crucible wiring + shepherd + web_search),
  `status_agentic.sh`. Dry-run (1 episode) to validate, then the overnight run.
- **Wave 3 — findings.** README + article, like inference_speed.

## 10. Honest expected outcome

On a 9B, harness techniques (test-feedback, reflection, failure-memory, better
prompts, best-of-K) typically lift coding pass rates ~10-30 absolute — **if**
reward-hacking is gated. The experiment's value is the **measured, lossless lift** +
the reasoning trail of which techniques actually helped a 9B, and whether the
Crucible loop (verifier-grounded, shepherded) can find it autonomously the way Weco
did on frontier models. Prior: a real lift, not a miracle; the gate keeps it honest.

## 11. Open defaults (resolved at approval)

- `Ok` threshold: +10 absolute over baseline.
- Search subset: 10 tasks (instruct-leaning); held-out final: 15; total 25.
- Sandbox: subprocess (docker fallback).
- Agent default: Tess-4-9B @ :9090; model-agnostic via flag.