# LLM Shepherding (Frontier Advisor) — Design

**Date:** 2026-06-09
**Status:** Approved (brainstorming) — pending implementation plan
**Feature branch:** `feat/llm-shepherding`

## 1. Motivation

Crucible runs N independent workers driving a Ralph loop (edit → verify → feed
the verdict back) against a deterministic verifier. The interesting low-resource
question is *how far a tiny model gets*. The Fireworks "Frontier Advisors" result
([blog](https://fireworks.ai/blog/open-source-agents-frontier-advisors)) shows a
cheap open worker can **beat a frontier-only baseline at ~39% of the cost** by
consulting a strong "advisor" model *sparsely* (~0.83 calls/task): the open model
does the bulk of the work; the frontier model supplies steering/critique at
uncertain moments only.

This feature adds an **optional** advisor to Crucible so a small local worker
(e.g. `gemma4-12b-qat` via the OpenAI-compatible endpoint) can be "shepherded"
by a frontier advisor (`claude-*` / `gemini-*`). It is off by default and, when
off, behavior is **byte-identical to today**.

### Measurable outcomes

1. With `advisor=None`, the engine advertises no extra tool and performs zero
   consults — identical worker conversations and verdicts to current `main`
   (enforced by a regression test).
2. With an advisor configured, a worker can obtain advisor text via two paths
   (self-trigger tool + engine-trigger on plateau/fail-streak), bounded by
   per-episode and per-run caps.
3. The advisor **never** mutates the artifact or the store; it returns validated
   text only. The worker applies any suggestion through its existing edit tools.
4. Every consult is recorded in the append-only store and visible via
   `crucible reasoning`, and a per-run **consults-per-task** figure is reportable
   (the paper's headline metric).
5. Advisor token cost is counted against `run_budget.usd` and surfaced
   separately.

## 2. Constraints & invariants (from the Ananke constitution + project CLAUDE.md)

- **#9 — LLM never touches the datastore.** The advisor returns text; dedicated
  worker code records consults and applies nothing automatically.
- **#10 — Validate all model output before acting.** Advisor responses are
  validated (non-empty, truncated, token-sanitized) before being injected into
  the worker conversation.
- **#12 — Assume external dependencies fail.** Advisor calls are wrapped; on any
  error the run continues with a neutral "advisor unavailable" note.
- **Surgical edits / public interfaces sacred.** `run()`'s existing signature is
  extended with one optional keyword (`advisor=`); no existing parameter changes
  meaning. `make_session` gains one optional keyword (`extra_tools=()`) that
  defaults to today's behavior.
- **Token efficiency.** The advisor receives a *curated slice*, not the raw
  transcript.
- **New dependencies:** none. Advisor sessions reuse the already-optional
  provider SDKs (`anthropic` / `google-genai` / `openai`).

## 3. Architecture

### 3.1 New module `crucible/advisor.py`

```python
@dataclass(frozen=True)
class AdvisorPolicy:
    model: str                         # routes via provider prefix, like the worker
    max_calls_per_episode: int = 1
    max_calls_per_run: int | None = None
    plateau_trigger: bool = True       # engine consults on plateau/fail-streak
    fail_streak: int = 3               # consecutive non-improving verdicts → engine consult
    scope: str = "suggestions"         # "suggestions" | "steering"

@dataclass(frozen=True)
class AdvisorRequest:
    files: dict[str, str]              # current artifact files
    editable_regions: list[str]
    verdict_text: str                  # rendered current verdict
    recent_lessons: list[str]
    recent_turns: list[str]            # last k worker turns (curated, not full transcript)
    trigger: str                       # "self" | "engine"
    scope: str

@dataclass(frozen=True)
class AdvisorResponse:
    advice: str                        # validated, sanitized, truncated text
    cost_usd: float
    ok: bool                           # False ⇒ advisor unavailable (neutral note used)

class Advisor(Protocol):
    def consult(self, req: AdvisorRequest) -> AdvisorResponse: ...
    @property
    def cost_usd(self) -> float: ...   # cumulative for this advisor instance
```

- **`LLMAdvisor`** — builds a **fresh, stateless** completion per consult via
  `make_advisor_session(policy.model)`. Renders `AdvisorRequest` into a system
  prompt (advisor role; `scope` controls whether concrete suggestions are
  invited) + a user message with the curated slice. Validates the reply:
  non-empty, strip, truncate (~2000 chars), and run the existing
  `_sanitize_lesson`-style neutralization so advice can never smuggle
  `crucible:region` / `crucible:hole` / `NotImplementedError` markers when echoed
  back into the artifact via a lesson. Accumulates `cost_usd`. Any exception →
  `AdvisorResponse(advice="(advisor unavailable)", cost_usd=0.0, ok=False)`.
- **`ScriptedAdvisor`** — deterministic test double returning queued advice.

### 3.2 Tools-free completion: `make_advisor_session` (in `providers.py`)

A dedicated text-completion path (chosen over reusing `LLMSession`), ~3 thin
branches mirroring `make_session`, each returning `(text, cost_usd)`:

```python
def make_advisor_session(model: str, base_url: str | None = None) -> AdvisorSession: ...

class AdvisorSession(Protocol):
    def complete(self, system: str, user: str) -> str: ...
    @property
    def cost_usd(self) -> float: ...
```

Implementations call each provider with **no tools** and read assistant text.
This keeps the advisor fully decoupled from the tool-loop machinery and avoids
the fragility of a model emitting a tool call where prose is expected. Pricing
reuses `price_for`.

### 3.3 Worker integration (`worker.py`)

- `consult_advisor` tool schema (`CONSULT_ADVISOR_SCHEMA` in `llm.py`) advertised
  to the worker **only when an advisor is configured**, threaded through a new
  optional `extra_tools=()` on `make_session` and the provider classes (empty ⇒
  identical API call as today).
- **Self-trigger:** `_handle_call` routes `consult_advisor` to the advisor,
  builds an `AdvisorRequest(trigger="self")`, and returns the advice as the
  `ToolResult`. Counts against the caps.
- **Engine-trigger:** `run_episode` tracks consecutive non-improving verdicts
  (using the existing `outcome_rank`-style comparison on per-turn verdicts). On
  reaching `fail_streak` (and `plateau_trigger` enabled), it performs one consult
  and **appends** the advice to the next verdict feedback as
  `\n\n[ADVISOR]: …` — no `LLMSession` protocol change. Counts against the caps.
- Cap enforcement: a per-episode counter and a shared per-run counter
  (`max_calls_per_run`) gate both paths; once exhausted, `consult_advisor`
  returns "advisor budget exhausted" and the engine path is skipped.
- `EpisodeOutcome` gains `advisor_calls: int` and `advisor_cost_usd: float`;
  episode `cost_usd` includes advisor cost so `run_budget.usd` accounts for it.

### 3.4 Orchestrator & provenance (`orchestrator.py`, `store.py`)

- An `advisor_factory: Callable[[], Advisor] | None` is threaded to each worker
  (one instance per worker, mirroring the session factory, for clean per-worker
  cost isolation across threads).
- Each consult is recorded by worker code as an append-only event
  `advisor_consult` with `{worker, episode, trigger, advice, cost_usd}`. The
  advice is already in the worker's `messages` (tool result or verdict append),
  so it surfaces in `reasoning_json` / `crucible reasoning` as `[advisor] …`.
- `run` config dict records `{advisor: <model|null>, advisor_policy: {...}}`.
- Consults-per-task = count of `advisor_consult` events / tasks — reportable from
  the store (reproduces the paper's 0.83 metric).

### 3.5 Config surface

- **SDK:** `run(..., advisor: str | AdvisorPolicy | None = None)`. A string is
  sugar for `AdvisorPolicy(model=<string>)` with defaults; `None` ⇒ fully off.
- **CLI:** `--advisor MODEL`, `--advisor-max-calls N` (per-episode),
  `--advisor-fail-streak N`. Absent ⇒ off.
- **Examples:** `examples/sidon/run_sidon.py` and `examples/chem/run_chem.py`
  gain an optional `--advisor MODEL` passthrough (best showcase: tiny worker +
  frontier advisor on the Sidon climb).

## 4. Data flow (a consult)

```
worker turn → (self: calls consult_advisor)  ─┐
            → (engine: fail_streak reached)  ─┤
                                              ▼
              build AdvisorRequest (curated slice)
                                              ▼
        LLMAdvisor.consult → make_advisor_session.complete  (no tools)
                                              ▼
        validate + sanitize + truncate  →  AdvisorResponse
                                              ▼
   inject advice  (self: ToolResult │ engine: appended to verdict feedback)
                                              ▼
   record advisor_consult event (append-only)  +  add cost to episode
                                              ▼
   worker continues; applies suggestions via its OWN edit tools
```

## 5. Error handling & degradation

- Advisor API failure → `ok=False`, neutral note injected, `advisor_error` event
  recorded, run continues. Never raises into the loop.
- Validation failure (empty / non-string) → treated as unavailable.
- Caps exhausted → consult skipped with an explanatory note; never an error.

## 6. Testing strategy

- **Unit (`tests/unit/test_advisor.py`):** `AdvisorPolicy` defaults; `LLMAdvisor`
  validation/truncation/sanitization; graceful degradation on raised exception;
  `ScriptedAdvisor`.
- **Worker:** self-trigger path (tool → advice in `ToolResult`); engine-trigger
  path (fail-streak → advice appended to verdict) using `ScriptedSession` +
  `ScriptedAdvisor`.
- **Caps:** `max_calls_per_episode` and `max_calls_per_run` enforced across both
  paths.
- **Off-by-default regression:** with `advisor=None`, the worker's advertised
  tools and resulting conversation/verdicts are identical to current behavior
  (no `consult_advisor` schema present); zero `advisor_consult` events.
- **Provider:** `make_advisor_session` returns text + cost (live tests gated like
  existing `test_providers_live.py`).
- **Provenance:** `advisor_consult` events recorded and visible via the reasoning
  CLI.

## 7. Files

| File | Change |
|------|--------|
| `crucible/advisor.py` | **new** — `AdvisorPolicy`, `Advisor`, requests, `LLMAdvisor`, `ScriptedAdvisor` |
| `crucible/providers.py` | `extra_tools=()` on provider classes + `make_session`; `make_advisor_session` + `AdvisorSession` impls |
| `crucible/llm.py` | `CONSULT_ADVISOR_SCHEMA`; optional `extra_tools` plumbed to sessions |
| `crucible/worker.py` | consult handling (self + engine), cap counters, `EpisodeOutcome` fields |
| `crucible/orchestrator.py` | thread `advisor_factory`; record `advisor_consult`/`advisor_error`; config dict |
| `crucible/__init__.py` | `advisor` kwarg on `run()` |
| `crucible/cli.py` | `--advisor`, `--advisor-max-calls`, `--advisor-fail-streak` |
| `examples/sidon/run_sidon.py`, `examples/chem/run_chem.py` | optional `--advisor` passthrough |
| `README.md` | new "LLM shepherding (optional advisor)" section + consults-per-task note |
| `tests/unit/test_advisor.py` (+ worker/provider tests) | **new/extended** |

## 8. Out of scope (YAGNI)

- Multiple advisors / advisor voting.
- Advisor-initiated edits or direct artifact access.
- Persistent advisor memory across consults (each consult is stateless).
- Auto-tuning of trigger thresholds.
