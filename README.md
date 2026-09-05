# Crucible

**Verifier-grounded multi-agent search engine.** The verifier — not the model's
confidence — decides what counts as a solution.

Crucible runs N independent LLM workers against a task with frozen, immutable
regions and explicit editable holes. Each worker iterates a Ralph loop
(edit → verify → feed the verdict back as the gradient) inside a locked sandbox.
A candidate is accepted only when a deterministic verifier says so **and** the
integrity gate confirms nothing was gamed. Everything is logged append-only in
SQLite and is replayable — including every model turn, so you can [inspect how each
worker reasoned](#inspecting-reasoning) after the fact.

- **v0** — pass/fail search over code, proofs, and exploits (see `CRUCIBLE_PRD_v0_1.md`)
- **v0.5** — optimization search via the `Scored` verdict: same engine, graded objectives

## What this is — and what it isn't

Crucible is **an experiment, not a product.** Its real question is:

> *How far does a verifier-grounded LLM loop get on a shoestring?*

The two papers below were run with serious infrastructure — reinforcement-learning provers on TPUs, frontier multi-agent platforms, evolutionary populations, thousands of episodes per problem. Crucible deliberately keeps **only the cheapest configuration**: no training, no RL, no fine-tuning, no evolutionary search — a single developer, one machine, a handful of parallel workers, small token budgets, and either a frontier API or a local model. It is a probe of how much of those papers' capability survives at the low-resource end. (One of the papers' own surprising results is that this cheapest configuration is far more competitive than expected — see [What I found](#what-I-found).)

It is **not** a reproduction of either paper's full system, and it is not a benchmark. Treat every number here as "what one person reached with very little."

## Inspired by two papers

Crucible is a small, hands-on synthesis of two 2026 papers that describe the same machine from opposite ends:

- **AlphaProof Nexus** — *Advancing Mathematics Research with AI-Driven Formal Proof Search* (Google DeepMind) — [arXiv:2605.22763](https://arxiv.org/abs/2605.22763). **The engine.** Crucible takes its core loop: an LLM edits an artifact-with-holes, a verifier (there, the Lean compiler) checks every edit, and the verdict is fed back as the search gradient. `examples/lean_ladder/` is a direct nod to its origin domain; the integrity gate generalizes that paper's `SafeVerify`; and the first-wins, no-shared-state orchestrator is that paper's **basic agent (Agent A)** — the configuration the authors found rivals the full evolutionary system as models improve.
- **CategoryScienceClaw** — *Self-Revising Discovery Systems for Science: A Categorical Framework for Agentic Artificial Intelligence* (MIT) — [arXiv:2606.01444](https://arxiv.org/abs/2606.01444). **The type-system.** Crucible takes its central thesis — **the gate (verifier) is what makes a result real**, not the model's confidence — and the progression it formalizes: *search* (satisfy a predicate, v0) → *optimization* (grade a `Scored` objective, v0.5) → *discovery* (extend the type schema, the v1 horizon). Typed artifacts and append-only provenance come from here too.

One paper is the engine, the other is the type-system; Crucible is the smallest thing that runs the loop they share.


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

## What I found

Four results, in increasing order of interest.

**1. Strong verifiers can't be gamed (`examples/integrity_suite/`).** The five probes — `pytest.skip`, `# type: ignore`, hardcoded outputs against a held-out suite, editing the hidden test, mocking the system under test — are all rejected, by the deny-list, the immutable-region check, or the verifier simply running the whole suite. A deterministic verifier plus the integrity gate is honest ground truth.

**2. The origin domain reproduces cheaply (`examples/lean_ladder/`).** Fourteen graded Lean 4 theorems (core Lean, no mathlib) are driven to sorry-free proofs by a single `Lean()` verifier in a small Docker image — the AlphaProof-Nexus domain at solo-dev scale, no RL and no proof-search infrastructure.

**3. A weak verifier gets gamed — and that is the sharpest result (`examples/chem/`).** The molecular ladder asks the model to maximize a molecule's predicted aqueous solubility (logS) while keeping a required scaffold. The scorer is a hand-rolled ESOL-style surrogate:

```
logS = 0.16 − 0.63·clogP − 0.0062·MW + 0.066·rotatable_bonds − 0.74·aromatic_fraction
```

Every rung "solves" — but look at *how*:

```
08_purine_summit:      SOLVED   OCC(O)C(O)C(O)C(O)Cn1cnc2cncnc21
06_naphthalene_burden: SOLVED   OC(CO)C(O)C(O)COc1ccc2cc(OCC(O)C(O)CO)ccc2c1
05_greasy_chain:       SOLVED   OCC(O)C(O)C(O)C(O)C(O)C(O)C(O)CO
```

The model did not find clever soluble molecules. It found the **degenerate optimum of the formula**: bolt a long polyhydroxyl (polyol) chain onto whatever scaffold is required. That single move pulls all three big levers at once — many −OH groups crush `clogP` (×−0.63), the chain adds `rotatable_bonds` (×+0.066), and the extra non-aromatic atoms shrink the `aromatic_fraction` penalty (×−0.74) — while `MW` is barely penalized (×−0.0062) and nothing scores synthesizability, stability, or drug-likeness. So *more polyol is always strictly better*, and the optimizer carpet-bombs hydroxyls. The molecules are valid and keep the scaffold, so they pass; they are not what "optimize solubility" meant.

This is **specification gaming / Goodhart's law**, and it is the two papers' thesis made concrete: **the verifier is the product.** With a strong verifier (the Lean compiler, pytest plus a held-out suite, the integrity gate) a win is real and ungameable. With a weak one (a logS formula with no cost term) the model optimizes the *measure* instead of the *intent*. AlphaProof Nexus needed end-to-end formal verification for exactly this reason; CategoryScienceClaw makes its gate pay for complexity (an MDL budget) for exactly this reason. Crucible's chem ladder is the failure they were guarding against, reproduced in miniature — and for a low-resource experiment, that negative result is the most useful thing in the repo.

**4. A strong verifier stays honest on a genuinely hard problem (`examples/sidon/`).** The chem ladder's mirror image, and the clearest single demonstration of the thesis. The task: return the largest possible **Sidon set** — distinct integers in `[1, 10000]` whose pairwise sums `a+b` are all distinct — with the `Ok` target set at 100 (≈ √10000, the asymptotic ceiling for a Sidon set in this range, so the target is deliberately near-optimal). The `SidonVerifier` is deterministic and unforgiving: it re-derives every pairwise sum and fails on the first collision, so there is no surrogate formula to Goodhart and nothing to mock. Running `gemini-3.1-flash-lite` (a cheap model) with 5 workers, the search climbs honestly: a plain greedy / Mian-Chowla sweep plateaus at 66, the Erdős–Turán construction `2pk + (k² mod p)` reaches ~71, and **seeding the greedy fill with that construction tops out at 74 / 100** (range `[71, 9941]`, 2775 distinct pairwise sums). It never reaches 100 — and that is the point. With a strong verifier the engine **cannot inflate the number**: 74 is a real lower bound that a tiny model rediscovered by reasoning its way to the right family of constructions, not a gamed one.

Read together, chem and sidon **bracket the entire thesis** — same engine, opposite verifiers:

| | Verifier | What the model does | What you get |
|---|----------|---------------------|--------------|
| **chem** (`examples/chem/`) | weak surrogate (logS, no cost term) | optimizes the *measure* — spams hydroxyls | a high score that doesn't mean what you asked |
| **sidon** (`examples/sidon/`) | strong, exact (recomputes every sum) | optimizes the *intent* — does real mathematics | an honest, ungameable lower bound (74/100) |

The verifier — not the model — decides whether *"what you got"* equals *"what you asked for."* And because every edit, tool call, and verdict is recorded, you can watch the Sidon climb happen turn by turn: see [Inspecting reasoning](#inspecting-reasoning).

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
| `Forge()` | `forge test` green, ≥ 1 test, no holes | Foundry image (e.g. `crucible-solidity:0`) | Solidity/Foundry; a 0-test run (delete/rename the test) is a gaming vector → `FAIL` |
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

A second reference is the `SidonVerifier` (`examples/sidon/`): it scores by **set
size** and recomputes every pairwise sum exactly, so the objective is strong and
ungameable — the same `Scored` machinery on a pure-math problem with no surrogate
to exploit. See [What I found](#what-I-found) for why the two make a matched pair.

## Examples

| Example | What it demonstrates | Requires |
|---------|----------------------|----------|
| `examples/integrity_suite/` | five classic gaming probes, each rejected | `crucible-py:0` |
| `examples/lean_ladder/` | Graded Lean 4 theorem proving (14 increasingly complex theorems), sorry-free | `crucible-lean:0` |
| `examples/chem/` | Molecular solubility optimization (8 chemical scaffold challenges, v0.5) | `crucible-chem:0`, API key |
| `examples/sidon/` | Sidon-set maximization (pure-math v0.5 stress test, strong/ungameable verifier) | API key (no Docker — runs in `SubprocessSandbox`) |
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

> **Heads-up:** this ladder is also Crucible's clearest *negative* result — see [What I found](#what-I-found). The model games the surrogate by spamming hydroxyl groups onto the scaffold rather than designing sensible molecules. That is the point: it shows what a weak verifier costs you.


### Sidon Set (Erdős stress test)

The chem ladder's positive mirror, and a good first run because it needs **no Docker image** — the candidate is stdlib-only Python, so it executes in the `SubprocessSandbox`. Crucible must return the largest Sidon set it can find in `[1, 10000]`; the `SidonVerifier` scores by set size and only emits `Ok` at the target (default 100 ≈ √10000, the asymptotic ceiling).

```bash
uv run python examples/sidon/run_sidon.py                                   # auto-detects key/model from .env
uv run python examples/sidon/run_sidon.py --model claude-sonnet-4-6 --workers 5
uv run python examples/sidon/run_sidon.py --target 100 --sandbox subprocess
```

Two things make this example worth studying:

- **The verifier can't be gamed.** It recomputes every pairwise sum `a+b` (including `a+a`) and fails on the first duplicate, checks range and uniqueness, and reproduces in a fresh sandbox. There is no formula to exploit — the only way to score higher is to actually build a bigger Sidon set.
- **The run re-verifies independently.** `run_sidon.py` re-executes the winning artifact *outside* the engine and recomputes the Sidon property from scratch before printing the size — never trust the model's self-report, and never trust a single verifier path either.

Representative result (`gemini-3.1-flash-lite`, 5 workers, target 100): best size **74**, reached by seeding a greedy fill with the Erdős–Turán construction `2pk + (k² mod p)`. The agents rediscover the construction family on their own, record lessons that carry across episodes, and plateau at an honest lower bound rather than reaching the near-optimal target. Inspect the full climb with:

```bash
uv run crucible reasoning --db examples/sidon/sidon.db
```


## LLM Shepherding (optional)

Crucible supports optional **LLM shepherding** — weaker coding agents can consult a stronger "advisor" model when stuck. This is useful when you want to use a fast, cheap model for the bulk of the search but escalate tough blockers to a stronger model.

**How it works:**

- **Self-trigger**: The worker explicitly calls `consult_advisor` when it recognizes it's stuck.
- **Engine-trigger**: The system automatically triggers the advisor after `fail_streak` non-improving episodes (plateau detection).
- **Caps**: Configurable limits on advisor calls per episode and per run prevent over-reliance.
- **Provenance**: All advisor consultations are recorded as `advisor_consult` events in SQLite.

### SDK usage

```python
from crucible import AdvisorPolicy, Task, run, budgets

# String shorthand (default policy: 1 call/episode, plateau_trigger=True, fail_streak=3)
result = run(
    task=Task.from_path("problem.py", editable=["solution"]),
    verifier=...,
    model="claude-sonnet-4-6",  # worker model
    advisor="claude-opus-4-8",  # advisor model
)

# Full policy control
result = run(
    task=Task.from_path("problem.py", editable=["solution"]),
    verifier=...,
    model="claude-sonnet-4-6",
    advisor=AdvisorPolicy(
        model="claude-opus-4-8",
        max_calls_per_episode=1,
        max_calls_per_run=5,
        plateau_trigger=True,
        fail_streak=3,  # trigger advisor after 3 non-improving episodes
        scope="suggestions",  # "suggestions" | "steering"
    ),
)
```

### CLI usage

```bash
# Enable advisor with default policy
crucible run problem.py --editable solution --verifier pytest:tests/ \
    --advisor claude-opus-4-8

# Full control over advisor policy
crucible run problem.py --editable solution --verifier pytest:tests/ \
    --advisor claude-opus-4-8 \
    --advisor-max-calls 5 \
    --advisor-fail-streak 4
```

### Example scripts

The example scripts support advisor passthrough:

```bash
# Sidon with advisor
uv run python examples/sidon/run_sidon.py --advisor claude-opus-4-8

# Chemistry with advisor
uv run python examples/chem/run_chem.py --advisor claude-opus-4-8
```

### Graceful degradation

If the advisor is unavailable (network error, missing API key, rate limit), the system degrades gracefully: the worker receives `"(advisor unavailable — proceed on your own)"` and continues. Advisor failures are logged but do not fail the run.

## Inspecting reasoning

Every episode's full conversation — the model's text reasoning, each tool call, and the verifier feedback returned after each edit — is captured automatically in the `reasoning_json` column of the `episodes` table (all providers: Anthropic, OpenAI, Gemini). It lives in the same append-only SQLite store as the rest of the provenance, so any run is replayable after the fact.

```bash
uv run crucible runs                                          # list runs in the default db
uv run crucible runs --db examples/sidon/sidon.db             # list runs in a specific db

uv run crucible reasoning --db examples/sidon/sidon.db        # latest run, all workers/episodes
uv run crucible reasoning 7  --db examples/sidon/sidon.db     # a specific run
uv run crucible reasoning --worker 0 --episode 0 --db examples/sidon/sidon.db
```

The output is the conversation in order:

- `[user]` — the initial problem (artifact + verdict) and tool results
- `[model]` — the model's reasoning and tool calls (`→ write_region(...)`, `→ record_lesson(...)`)
- `  ← ...` — the verifier verdict returned after each edit (the Ralph-loop gradient)

This is how you tell a real win from a gamed one, and how you watch a search like the [Sidon stress test](#sidon-set-erdős-stress-test) climb, plateau, and accumulate lessons across episodes.


## Citing the papers

If this experiment is useful, cite the work it is built on:

- G. Tsoukalas et al., *Advancing Mathematics Research with AI-Driven Formal Proof Search*, Google DeepMind, 2026. [arXiv:2605.22763](https://arxiv.org/abs/2605.22763)
- F. Y. Wang and M. J. Buehler, *Self-Revising Discovery Systems for Science: A Categorical Framework for Agentic Artificial Intelligence*, MIT, 2026. [arXiv:2606.01444](https://arxiv.org/abs/2606.01444)

Crucible is an independent experiment and is not affiliated with the authors of either paper.
