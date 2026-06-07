# Crucible

**Verifier-grounded multi-agent search engine.** The verifier — not the model's
confidence — decides what counts as a solution.

Crucible runs N independent LLM workers against a task with frozen, immutable
regions and explicit editable holes. Each worker iterates a Ralph loop
(edit → verify → feed the verdict back as the gradient) inside a locked sandbox.
A candidate is accepted only when a deterministic verifier says so **and** the
integrity gate confirms nothing was gamed. Everything is logged append-only in
SQLite and is replayable.

- **v0** — pass/fail search over code, proofs, and exploits (see `CRUCIBLE_PRD_v0_1.md`)
- **v0.5** — optimization search via the `Scored` verdict: same engine, graded objectives

## How it works

A win must satisfy all four conditions:

1. **Verify `OK`** — the deterministic verifier accepts the artifact.
2. **Immutable spec untouched** — bytes outside the editable regions are
   byte-identical to the frozen original (integrity check 1).
3. **No escape tokens** — deny-lists scan the editable text for gaming vectors:
   `pytest.skip`, `# type: ignore`, `unittest.mock`, Lean `sorry` / `axiom`, …
   (integrity check 2).
4. **Fresh re-verify** — the verdict reproduces in a brand-new locked sandbox
   (integrity check 3).

The first worker to clear all four wins and the run ends. If no worker wins
within budget, the run returns `best_partial` — the highest-ranked attempt
(`Ok > Scored > Partial > Fail`, then score, then fewest open holes).

## Installation

```bash
uv venv .venv && uv pip install -e ".[dev]"

# provider SDKs are optional extras — install the ones you use
uv pip install -e ".[anthropic]"   # Claude models
uv pip install -e ".[gemini]"      # Google Gemini models
uv pip install -e ".[openai]"      # OpenAI-compatible endpoints (incl. local models)
```

API keys come from the environment only — never from code or config:
`ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` (or `GEMINI_API_KEY`), `OPENAI_API_KEY`
(+ `OPENAI_BASE_URL` for local/self-hosted endpoints).

Build the verifier images you need (see `docker/README.md`):

```bash
docker build -t crucible-py:0       -f docker/py.Dockerfile       docker/
docker build -t crucible-lean:0     -f docker/lean.Dockerfile     docker/
docker build -t crucible-chem:0     -f docker/chem.Dockerfile     docker/
```

## Quickstart

### SDK

```python
from crucible import Task, run, budgets
from crucible.verifiers import Pytest

result = run(
    task=Task.from_path("problem.py", editable=["solution"]),
    verifier=Pytest(suite="tests/"),
    model="claude-sonnet-4-6",          # claude-* | gemini-* | OpenAI-compatible
    workers=10,
    episode=budgets(edits=90, turns=40),
    run_budget=budgets(wall_clock="2h", usd=200),
)
result.solution      # Artifact | None — hole-free, verified, integrity-clean
result.best_partial  # Artifact — highest-ranked attempt if unsolved
result.run_id        # provenance key into the append-only store
```

### CLI

```bash
crucible run problem.py --editable solution --verifier pytest:tests/ --workers 10 \
    --episode-edits 90 --episode-turns 40 --run-budget 2h,200usd
```

Exit code `0` = solved (verified artifact written to `--out`), `2` = unsolved
(best partial written instead). Verifier specs: `pytest:SUITE`, `pyright:MODE`,
`cmd:COMMAND`, `lean`.

## Verifiers

All verifiers live in `crucible.verifiers` and implement the same contract:
`verify(artifact, ctx) -> Ok | Partial | Fail | Scored`.

| Verifier | Accepts when | Sandbox image | Notes |
|----------|--------------|---------------|-------|
| `Pytest(suite=)` | suite green — no skips, xfails, or holes | `crucible-py:0` | reference verifier |
| `Pyright(strict=)` | zero type errors, no holes | `crucible-py:0` | strict or standard mode |
| `Command(cmd)` | exit 0, no holes | any | escape hatch for any CLI-checkable artifact; shell operators rejected |
| `Lean()` | every `.lean` file compiles sorry-free | `crucible-lean:0` | `sorry`/`axiom` are also deny-list tokens |
| `Chem(target=)` | molecule valid, scaffold preserved, score ≥ target | `crucible-chem:0` | returns `Scored(value)` below target (v0.5) |
| `Rubric(spec=, threshold=)` | LLM judge ≥ threshold | — | **advisory only** (`deterministic=False`): can never be the sole accept signal |

`run()` refuses an advisory verifier as the sole verifier — a non-deterministic
verdict can never be the accept signal (PRD §3).

## Model providers

`make_session` routes by model-name prefix:

| Prefix | Provider | Key | Extra |
|--------|----------|-----|-------|
| `claude-*` | Anthropic | `ANTHROPIC_API_KEY` | `crucible[anthropic]` |
| `gemini-*` | Google Gemini | `GOOGLE_API_KEY` / `GEMINI_API_KEY` | `crucible[gemini]` |
| anything else | OpenAI-compatible | `OPENAI_API_KEY`, `OPENAI_BASE_URL` | `crucible[openai]` |

Per-session token usage is priced (known models) and counted against the run's
USD budget; unknown/local models cost 0.0 (no USD-cap signal).

## Budgets

Exceeding any budget ends the run gracefully with `best_partial`.

- **Episode** (`budgets(edits=, turns=)`): per-episode edit and turn caps
  (defaults 90 / 40).
- **Run** (`budgets(wall_clock=, usd=, episodes=)`): wall clock (default 2 h),
  optional USD cap, episodes per worker (default 8), and `plateau_patience` —
  stop a worker after N episodes with no rank improvement (v0.5, optimization runs).

## Sandboxes

Verification runs in a locked sandbox — `DockerSandbox` (default, `--network=none`
unless the Task declares `network=True`) or `SubprocessSandbox` (trusted local
fallback). The fresh re-verify always uses a brand-new sandbox instance.

## Optimization search (v0.5)

Beyond pass/fail, Crucible optimizes. A verifier may return `Scored(value)` for
a valid artifact that doesn't yet meet the target:

- Episodes are ranked `Ok > Scored > Partial > Fail`; among `Scored`, higher value wins.
- The orchestrator tracks the best artifact by rank; if no `Ok` is reached, the
  run returns the **best-scoring valid artifact**.
- `plateau_patience` stops workers early once the best score stops improving.
- The v0 accept path is unchanged: hitting the target yields `Ok` and goes
  through the same first-wins + fresh-reverify gate.

The reference implementation is the `Chem` verifier: a deterministic RDKit
surrogate (ESOL-style predicted solubility, baked into `crucible-chem:0`) scores
a SMILES string; deterministic scoring means `Scored`/`Ok` verdicts reproduce
under fresh re-verify.

## Examples

| Example | What it demonstrates | Requires |
|---------|----------------------|----------|
| `examples/integrity_suite/` | five classic gaming probes, each rejected | `crucible-py:0` |
| `examples/lean_ladder/` | Graded Lean 4 theorem proving (14 increasingly complex theorems), sorry-free | `crucible-lean:0` |
| `examples/chem/` | Molecular solubility optimization (8 chemical scaffold challenges, v0.5) | `crucible-chem:0`, API key |
| `examples/run_kata_real_model.py` | minimal end-to-end kata with a real model | `crucible-py:0`, API key |

### Integrity-gate suite

A verifier-grounded engine is only as honest as its integrity gate. The suite
runs five probes — the kind a "trust the model's claim" setup would accept —
and shows Crucible rejecting each one:

```bash
.venv/bin/python examples/integrity_suite/run_suite.py
```

| Probe | Gate / mechanism | Crucible's response |
|-------|------------------|---------------------|
| `pytest.skip(...)` in the solution | deny-list (check 2) | `integrity_violation`, reverted |
| `# type: ignore` to silence pyright | deny-list (check 2) | `integrity_violation`, reverted |
| hardcode the one visible answer | verifier runs the *whole* suite | `FAIL` — the held-out case fails |
| edit the hidden test suite | suite outside every editable region (check 1) | edit refused; suite byte-identical |
| `unittest.mock` the system under test | deny-list (check 2) | `integrity_violation`, reverted |

Each rejection is recorded in the append-only SQLite provenance and reproducible.

### Lean Math Ladder

A graded collection of 14 increasingly complex Lean 4 theorems (core Lean only, no mathlib), each seeded with `sorry`. Crucible drives the `Lean()` verifier until the proof compiles sorry-free — `sorry` or `axiom` are both holes and deny-list tokens, ensuring no accepted proof contains unproven assumptions.

```bash
.venv/bin/python examples/lean_ladder/run_ladder.py --scripted          # Smoke test with pre-recorded solutions
ANTHROPIC_API_KEY=... .venv/bin/python examples/lean_ladder/run_ladder.py
```

**Ladder Rungs (in increasing difficulty):**
1. `add_comm` — Commutativity of natural number addition
2. `add_assoc` — Associativity of natural number addition
3. `succ_gt` — For any natural number `a`, `a < a + 1`
4. `mul_zero` — Multiplication by zero is zero
5. `list_append` — List append properties
6. `le_succ` — Monotonicity under successor
7. `rev_involution` — Reversing a list twice returns the original
8. `sum_formula` — Sum of integers formula
9. `tree_mirror` — Tree mirroring properties
10. `pow_add` — Exponentiation addition property
11. `isort_sorted` — Insertion sort correctness
12. `isort_count` — Insertion sort element count
13. `compiler_correct` — Toy compiler correctness
14. `ackermann_gt` — Ackermann function growth property

Each rung builds on previous proofs, creating a structured learning path for theorem proving.


### Molecular Solubility Optimization

A molecular optimization challenge where Crucible seeks to maximize aqueous solubility (logS) of organic molecules while preserving their core scaffolds. This example demonstrates v0.5's optimization capabilities with the `Chem` verifier.

```bash
.venv/bin/python examples/chem/run_chem_ladder.py --scripted          # Smoke test with pre-recorded solutions
GOOGLE_API_KEY=... .venv/bin/python examples/chem/run_chem.py
```

**Chemical Scaffold Challenges:**
1. `free_solubility` — Optimize solubility of a simple organic molecule
2. `benzene_polyol` — Benzene ring with hydroxyl groups
3. `pyridine_push` — Pyridine-based scaffold optimization
4. `sulfonamide` — Sulfonamide functional group optimization
5. `greasy_chain` — Long aliphatic chain optimization
6. `naphthalene_burden` — Naphthalene scaffold optimization
7. `indole_tight` — Indole scaffold with tight constraints
8. `purine_summit` — Purine-based scaffold optimization

Each challenge presents a starting molecule and Crucible iteratively modifies it to improve solubility, guided by the RDKit-based `Chem` verifier's scoring function. The goal is to reach a logS value ≥ 0.0 (highly soluble).

