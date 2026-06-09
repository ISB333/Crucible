# LLM Shepherding (Frontier Advisor) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional frontier "advisor" model that a small Crucible worker can consult sparsely (self-triggered tool + engine-triggered on plateau), returning text-only steering the worker applies through its own edit tools.

**Architecture:** A new `crucible/advisor.py` holds the policy, request/response types, an `LLMAdvisor` (stateless per consult via a tools-free completion), and a `ScriptedAdvisor` test double. The worker's Ralph loop gains two consult paths; both inject validated advisor text back into the conversation. When `advisor=None`, no extra tool is advertised and behavior is byte-identical to today.

**Tech Stack:** Python 3.12, pytest (markers: `unit`), pyright (strict via `pyrightconfig.json`), existing optional provider SDKs (`anthropic` / `google-genai` / `openai`). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-09-llm-shepherding-design.md`

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `crucible/advisor.py` | Advisor policy, request/response types, `LLMAdvisor`, `ScriptedAdvisor`, advice validation | **create** |
| `crucible/llm.py` | `CONSULT_ADVISOR_SCHEMA`; nothing else changes (sessions read tools from providers) | modify |
| `crucible/providers.py` | `extra_tools` on provider sessions + `make_session`; `AdvisorSession` + `make_advisor_session` | modify |
| `crucible/worker.py` | `verdict_rank`; `_handle_call` returns verdict; consult closure; self/engine triggers; `EpisodeOutcome` fields | modify |
| `crucible/orchestrator.py` | thread `advisor_factory`/`advisor_policy`; record `advisor_consult` events; config dict | modify |
| `crucible/__init__.py` | `advisor` kwarg on `run()`; wire `extra_tools` + advisor factory | modify |
| `crucible/cli.py` | `--advisor`, `--advisor-max-calls`, `--advisor-fail-streak` | modify |
| `examples/sidon/run_sidon.py`, `examples/chem/run_chem.py` | optional `--advisor` passthrough | modify |
| `README.md` | "LLM shepherding (optional advisor)" section | modify |
| `tests/unit/test_advisor.py` | advisor unit tests | **create** |
| `tests/unit/test_worker.py` | self/engine trigger + caps + off-by-default | modify |
| `tests/unit/test_providers.py` | `extra_tools` plumbing | modify |

**Key type contracts (used across tasks — names are fixed):**

```python
# crucible/advisor.py
AdvisorPolicy(model: str, max_calls_per_episode=1, max_calls_per_run: int|None=None,
              plateau_trigger=True, fail_streak=3, scope="suggestions")
AdvisorRequest(files: dict[str,str], editable_regions: list[str], verdict_text: str,
               recent_lessons: list[str], question: str, trigger: str, scope: str)
AdvisorResponse(advice: str, cost_usd: float, ok: bool)
class Advisor(Protocol): consult(req: AdvisorRequest) -> AdvisorResponse; cost_usd: float
```

---

## Task 1: Advisor policy, types, and ScriptedAdvisor

**Files:**
- Create: `crucible/advisor.py`
- Test: `tests/unit/test_advisor.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_advisor.py`:

```python
import pytest

from crucible.advisor import (
    AdvisorPolicy,
    AdvisorRequest,
    AdvisorResponse,
    ScriptedAdvisor,
    sanitize_advice,
)

pytestmark = pytest.mark.unit


def test_policy_defaults() -> None:
    p = AdvisorPolicy(model="claude-opus-4-8")
    assert p.max_calls_per_episode == 1
    assert p.max_calls_per_run is None
    assert p.plateau_trigger is True
    assert p.fail_streak == 3
    assert p.scope == "suggestions"


def test_sanitize_advice_neutralizes_markers() -> None:
    out = sanitize_advice("use crucible:region start and a crucible:hole with NotImplementedError")
    assert "crucible:region" not in out
    assert "crucible:hole" not in out
    assert "NotImplementedError" not in out


def test_sanitize_advice_truncates() -> None:
    out = sanitize_advice("x" * 5000, limit=100)
    assert len(out) <= 100


def test_scripted_advisor_pops_queued_advice_and_tracks_cost() -> None:
    adv = ScriptedAdvisor(advice=["first", "second"], cost_per_call=0.01)
    req = AdvisorRequest(
        files={"p.py": "..."}, editable_regions=["solution"], verdict_text="FAIL",
        recent_lessons=[], question="stuck", trigger="self", scope="suggestions",
    )
    r1 = adv.consult(req)
    assert isinstance(r1, AdvisorResponse) and r1.advice == "first" and r1.ok is True
    assert adv.consult(req).advice == "second"
    assert adv.consult(req).advice == "(no further advice)"  # queue drained
    assert adv.cost_usd == pytest.approx(0.02)  # only the two real calls billed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_advisor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'crucible.advisor'`

- [ ] **Step 3: Write minimal implementation**

Create `crucible/advisor.py`:

```python
"""Optional frontier advisor (LLM shepherding). The advisor returns TEXT only;
the worker applies any suggestion through its own validated edit tools. The advisor
never touches the artifact or the store (Ananke #9/#10) and never raises into the
Ralph loop (#12)."""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

ADVICE_CHAR_LIMIT = 2000


def sanitize_advice(text: str, limit: int = ADVICE_CHAR_LIMIT) -> str:
    """Validate+neutralize advisor text before it enters the worker conversation.
    Mirrors worker._sanitize_lesson so advice can never smuggle region/hole markers
    if the worker later echoes it into the artifact via record_lesson."""
    cleaned = (
        str(text)
        .replace("crucible:region", "crucible_region")
        .replace("crucible:hole", "crucible_hole")
        .replace("NotImplementedError", "NIE")
        .strip()
    )
    return cleaned[:limit]


@dataclass(frozen=True)
class AdvisorPolicy:
    model: str
    max_calls_per_episode: int = 1
    max_calls_per_run: int | None = None
    plateau_trigger: bool = True
    fail_streak: int = 3
    scope: str = "suggestions"  # "suggestions" | "steering"


@dataclass(frozen=True)
class AdvisorRequest:
    files: dict[str, str]
    editable_regions: list[str]
    verdict_text: str
    recent_lessons: list[str]
    question: str
    trigger: str  # "self" | "engine"
    scope: str


@dataclass(frozen=True)
class AdvisorResponse:
    advice: str
    cost_usd: float
    ok: bool


@runtime_checkable
class Advisor(Protocol):
    def consult(self, req: AdvisorRequest) -> AdvisorResponse: ...
    @property
    def cost_usd(self) -> float: ...


@dataclass
class ScriptedAdvisor:
    """Deterministic test double: pops one advice string per consult."""

    advice: Sequence[str] = field(default_factory=list)
    cost_per_call: float = 0.0

    def __post_init__(self) -> None:
        self._queue: deque[str] = deque(self.advice)
        self._cost = 0.0

    def consult(self, req: AdvisorRequest) -> AdvisorResponse:
        if not self._queue:
            return AdvisorResponse("(no further advice)", 0.0, True)
        self._cost += self.cost_per_call
        return AdvisorResponse(sanitize_advice(self._queue.popleft()), self.cost_per_call, True)

    @property
    def cost_usd(self) -> float:
        return self._cost
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_advisor.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add crucible/advisor.py tests/unit/test_advisor.py
git commit -m "feat: advisor policy, request/response types, ScriptedAdvisor"
```

---

## Task 2: consult_advisor schema + extra_tools threading

**Files:**
- Modify: `crucible/llm.py` (add schema)
- Modify: `crucible/providers.py:53-64` (`to_openai_tools`), provider `__init__`s and `_step`s, `make_session:337-342`
- Test: `tests/unit/test_providers.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_providers.py`:

```python
def test_to_openai_tools_includes_extra() -> None:
    from crucible.llm import CONSULT_ADVISOR_SCHEMA, TOOL_SCHEMAS
    from crucible.providers import to_openai_tools

    base = to_openai_tools()
    assert len(base) == len(TOOL_SCHEMAS)
    extended = to_openai_tools(TOOL_SCHEMAS + [CONSULT_ADVISOR_SCHEMA])
    names = [t["function"]["name"] for t in extended]
    assert "consult_advisor" in names
    assert len(extended) == len(TOOL_SCHEMAS) + 1


def test_consult_advisor_schema_shape() -> None:
    from crucible.llm import CONSULT_ADVISOR_SCHEMA

    assert CONSULT_ADVISOR_SCHEMA["name"] == "consult_advisor"
    assert "question" in CONSULT_ADVISOR_SCHEMA["input_schema"]["properties"]
    assert CONSULT_ADVISOR_SCHEMA["input_schema"]["required"] == ["question"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_providers.py -k "extra or consult_advisor_schema" -v`
Expected: FAIL with `ImportError: cannot import name 'CONSULT_ADVISOR_SCHEMA'`

- [ ] **Step 3a: Add the schema in `crucible/llm.py`**

After the `TOOL_SCHEMAS` list (ends at line 76), add:

```python
CONSULT_ADVISOR_SCHEMA: dict[str, Any] = {
    "name": "consult_advisor",
    "description": (
        "Ask a stronger advisor model for guidance when you are stuck or unsure."
        " Use it sparingly, only at genuinely hard steps. The advisor returns TEXT"
        " advice only; you must still make any change yourself with the edit tools."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "What you are stuck on."}
        },
        "required": ["question"],
    },
}
```

- [ ] **Step 3b: Parametrize `to_openai_tools` in `crucible/providers.py:53`**

Replace:

```python
def to_openai_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in TOOL_SCHEMAS
    ]
```

with:

```python
def to_openai_tools(tools: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in (tools if tools is not None else TOOL_SCHEMAS)
    ]
```

- [ ] **Step 3c: Thread `extra_tools` through the three provider sessions**

In `AnthropicSession.__init__` (line 68) add the parameter and tool list:

```python
    def __init__(self, model: str, max_tokens: int = 4096,
                 extra_tools: Sequence[dict[str, Any]] = ()) -> None:
        import anthropic  # optional extra: crucible[anthropic]

        self._client = anthropic.Anthropic()  # key from ANTHROPIC_API_KEY
        self._model_name = model
        self._model = model
        self._max_tokens = max_tokens
        self._tools = [*TOOL_SCHEMAS, *extra_tools]
        self._system = ""
        self._messages: list[dict[str, Any]] = []
        self._in_tokens = 0
        self._out_tokens = 0
```

In `AnthropicSession._step` (line 98) change `tools=TOOL_SCHEMAS` to `tools=self._tools`.

In `OpenAICompatSession.__init__` (line 137):

```python
    def __init__(self, model: str, base_url: str | None = None,
                 extra_tools: Sequence[dict[str, Any]] = ()) -> None:
        from openai import OpenAI  # optional extra: crucible[openai]

        self._client = OpenAI(base_url=base_url or os.environ.get("OPENAI_BASE_URL"))
        self._model_name = model
        self._model = model
        self._tools = [*TOOL_SCHEMAS, *extra_tools]
        self._messages: list[dict[str, Any]] = []
        self._in_tokens = 0
        self._out_tokens = 0
```

In `OpenAICompatSession._step` (line 159) change `tools=to_openai_tools()` to `tools=to_openai_tools(self._tools)`.

In `GeminiSession.__init__` (line 186) add `extra_tools: Sequence[dict[str, Any]] = ()` to the signature and add `self._tools = [*TOOL_SCHEMAS, *extra_tools]` next to the other assignments. In `GeminiSession._step` (line 244) change the loop `for tool in TOOL_SCHEMAS:` to `for tool in self._tools:`.

(`Sequence` is already imported on line 10.)

- [ ] **Step 3d: Thread `extra_tools` through `make_session` (line 337)**

```python
def make_session(
    model: str, base_url: str | None = None, extra_tools: Sequence[dict[str, Any]] = ()
) -> LLMSession:
    if model.startswith("claude"):
        return AnthropicSession(model, extra_tools=extra_tools)
    if model.startswith("gemini"):
        return GeminiSession(model, extra_tools=extra_tools)
    return OpenAICompatSession(model, base_url=base_url, extra_tools=extra_tools)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_providers.py -k "extra or consult_advisor_schema" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crucible/llm.py crucible/providers.py tests/unit/test_providers.py
git commit -m "feat: consult_advisor schema and extra_tools threading"
```

---

## Task 3: AdvisorSession completion + LLMAdvisor

**Files:**
- Modify: `crucible/providers.py` (add `AdvisorSession` protocol + 3 impls + `make_advisor_session`)
- Modify: `crucible/advisor.py` (add `LLMAdvisor`, `render_request`)
- Test: `tests/unit/test_advisor.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_advisor.py`:

```python
def test_llm_advisor_validates_and_bills(monkeypatch) -> None:
    from crucible import advisor as adv_mod

    class FakeSession:
        def __init__(self) -> None:
            self.seen: tuple[str, str] | None = None
        def complete(self, system: str, user: str) -> str:
            self.seen = (system, user)
            return "  use a crucible:hole trick  "  # whitespace + marker to sanitize
        @property
        def cost_usd(self) -> float:
            return 0.05

    fake = FakeSession()
    monkeypatch.setattr(adv_mod, "make_advisor_session", lambda model, base_url=None: fake)

    a = adv_mod.LLMAdvisor(adv_mod.AdvisorPolicy(model="claude-opus-4-8"))
    req = adv_mod.AdvisorRequest(
        files={"p.py": "code"}, editable_regions=["solution"], verdict_text="FAIL: boom",
        recent_lessons=["tried X"], question="why fail?", trigger="self", scope="suggestions",
    )
    resp = a.consult(req)
    assert resp.ok is True
    assert resp.advice == "use a crucible_hole trick"  # sanitized + stripped
    assert resp.cost_usd == pytest.approx(0.05)
    assert a.cost_usd == pytest.approx(0.05)
    assert "FAIL: boom" in fake.seen[1] and "why fail?" in fake.seen[1]


def test_llm_advisor_degrades_on_error(monkeypatch) -> None:
    from crucible import advisor as adv_mod

    class BoomSession:
        def complete(self, system: str, user: str) -> str:
            raise RuntimeError("network down")
        @property
        def cost_usd(self) -> float:
            return 0.0

    monkeypatch.setattr(adv_mod, "make_advisor_session", lambda model, base_url=None: BoomSession())
    a = adv_mod.LLMAdvisor(adv_mod.AdvisorPolicy(model="gpt-4o"))
    req = adv_mod.AdvisorRequest(
        files={}, editable_regions=[], verdict_text="", recent_lessons=[],
        question="", trigger="engine", scope="steering",
    )
    resp = a.consult(req)
    assert resp.ok is False
    assert "unavailable" in resp.advice
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_advisor.py -k "llm_advisor" -v`
Expected: FAIL with `AttributeError: module 'crucible.advisor' has no attribute 'LLMAdvisor'`

- [ ] **Step 3a: Add the completion sessions in `crucible/providers.py`**

After `make_session` (end of file), add:

```python
from typing import Protocol  # add to the existing typing import at top, or here


@runtime_checkable
class AdvisorSession(Protocol):
    def complete(self, system: str, user: str) -> str: ...
    @property
    def cost_usd(self) -> float: ...
```

> Note: move `Protocol`/`runtime_checkable` into the top-level `from typing import Any` line — make it `from typing import Any, Protocol, runtime_checkable`.

Then the three implementations (tools-free, single round-trip):

```python
class _AnthropicAdvisor:
    def __init__(self, model: str, max_tokens: int = 1024) -> None:
        import anthropic

        self._client = anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens
        self._in = 0
        self._out = 0

    def complete(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self._model, max_tokens=self._max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        self._in += resp.usage.input_tokens
        self._out += resp.usage.output_tokens
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    @property
    def cost_usd(self) -> float:
        pin, pout = price_for(self._model)
        return (self._in * pin + self._out * pout) / 1_000_000


class _OpenAIAdvisor:
    def __init__(self, model: str, base_url: str | None = None) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url or os.environ.get("OPENAI_BASE_URL"))
        self._model = model
        self._in = 0
        self._out = 0

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        if resp.usage is not None:
            self._in += resp.usage.prompt_tokens
            self._out += resp.usage.completion_tokens
        return resp.choices[0].message.content or ""

    @property
    def cost_usd(self) -> float:
        pin, pout = price_for(self._model)
        return (self._in * pin + self._out * pout) / 1_000_000


class _GeminiAdvisor:
    def __init__(self, model: str, max_tokens: int = 1024) -> None:
        import google.genai as genai
        import google.genai.types as types  # type: ignore[reportMissingImports]

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY must be set in environment")
        self._client = genai.Client(api_key=api_key)  # type: ignore[reportPrivateImportUsage]
        self._model = model
        self._max_tokens = max_tokens
        self._types = types
        self._in = 0
        self._out = 0

    def complete(self, system: str, user: str) -> str:
        resp = self._client.models.generate_content(
            model=self._model,
            contents=[self._types.Content(role="user", parts=[self._types.Part(text=user)])],
            config={"max_output_tokens": self._max_tokens, "system_instruction": system},
        )
        return getattr(resp, "text", "") or ""

    @property
    def cost_usd(self) -> float:
        pin, pout = price_for(self._model)
        return (self._in * pin + self._out * pout) / 1_000_000


def make_advisor_session(model: str, base_url: str | None = None) -> AdvisorSession:
    if model.startswith("claude"):
        return _AnthropicAdvisor(model)
    if model.startswith("gemini"):
        return _GeminiAdvisor(model)
    return _OpenAIAdvisor(model, base_url=base_url)
```

- [ ] **Step 3b: Add `LLMAdvisor` and `render_request` in `crucible/advisor.py`**

At the top of `crucible/advisor.py`, the imports stay; add `LLMAdvisor` and a renderer at the end of the file:

```python
def render_request(req: AdvisorRequest) -> tuple[str, str]:
    """Build (system, user) for a single stateless advisor completion."""
    if req.scope == "steering":
        role = (
            "You are an expert advisor to a weaker coding agent. Give high-level steering"
            " and critique ONLY — do not write the solution. 3-6 sentences."
        )
    else:
        role = (
            "You are an expert advisor to a weaker coding agent. Diagnose the blocker and"
            " give a concrete, minimal suggestion (a snippet, value, or tactic). Be brief."
            " The agent must apply it itself with its own edit tools."
        )
    files = "\n\n".join(f"=== {p} ===\n{req.files[p]}" for p in sorted(req.files))
    lessons = "\n".join(f"- {l}" for l in req.recent_lessons) or "(none)"
    question = req.question or "(the agent is stuck and not making progress)"
    user = (
        f"Editable regions: {', '.join(req.editable_regions) or '(none)'}\n\n"
        f"{files}\n\nCurrent verdict:\n{req.verdict_text}\n\n"
        f"Lessons so far:\n{lessons}\n\nAgent's question: {question}"
    )
    return role, user


@dataclass
class LLMAdvisor:
    policy: AdvisorPolicy
    base_url: str | None = None

    def __post_init__(self) -> None:
        self._session = None
        self._cost = 0.0

    def _ensure(self):
        if self._session is None:
            self._session = make_advisor_session(self.policy.model, base_url=self.base_url)
        return self._session

    def consult(self, req: AdvisorRequest) -> AdvisorResponse:
        system, user = render_request(req)
        try:
            session = self._ensure()
            before = session.cost_usd
            text = session.complete(system, user)
            cost = max(0.0, session.cost_usd - before)
        except Exception:
            return AdvisorResponse("(advisor unavailable — proceed on your own)", 0.0, False)
        self._cost += cost
        advice = sanitize_advice(text)
        if not advice:
            return AdvisorResponse("(advisor returned nothing — proceed on your own)", cost, False)
        return AdvisorResponse(advice, cost, True)

    @property
    def cost_usd(self) -> float:
        return self._cost
```

Add the import at the top of `crucible/advisor.py`:

```python
from crucible.providers import make_advisor_session
```

> Note: `crucible.providers` imports lazily inside classes, so importing `make_advisor_session` at module top is safe (no SDK import happens until a session is constructed).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_advisor.py -v`
Expected: PASS (all tests, including the two new `llm_advisor` ones)

- [ ] **Step 5: Commit**

```bash
git add crucible/providers.py crucible/advisor.py tests/unit/test_advisor.py
git commit -m "feat: tools-free advisor sessions and LLMAdvisor with graceful degradation"
```

---

## Task 4: Worker consult paths (self-trigger + engine-trigger + caps)

**Files:**
- Modify: `crucible/worker.py` (`verdict_rank`, `_handle_call`, `run_episode`, `EpisodeOutcome`)
- Test: `tests/unit/test_worker.py`

- [ ] **Step 1: Write the failing tests**

Add to the top imports of `tests/unit/test_worker.py`:

```python
from crucible.advisor import AdvisorPolicy, ScriptedAdvisor
from crucible.worker import run_episode
from crucible.llm import ScriptedSession
```

Add helper + tests at the end of `tests/unit/test_worker.py`:

```python
def consult(question: str) -> ToolCall:
    return ToolCall(id="9", name="consult_advisor", args={"question": question})


def _episode(make_ctx, script, advisor=None, policy=None, cap=0):
    from crucible.budgets import EpisodeBudget
    from crucible.integrity import Composite, DenyTokens, ImmutableRegions
    from crucible.verify import RunContext
    a = Artifact.from_files({"problem.py": PROBLEM})
    integrity = Composite(checks=(ImmutableRegions.freeze(a), DenyTokens()))
    out, _ = run_episode(
        a, StubVerifier(), make_ctx(), ScriptedSession(script),
        EpisodeBudget(), integrity, advisor=advisor, policy=policy, advisor_cap=cap,
    )
    return out


def test_self_trigger_returns_advice_to_worker(tmp_path, make_ctx) -> None:
    adv = ScriptedAdvisor(advice=["try returning 42"], cost_per_call=0.01)
    pol = AdvisorPolicy(model="x", max_calls_per_episode=1)
    # turn 1: ask advisor; turn 2: apply; then stop
    out = _episode(make_ctx, [[consult("stuck")], [wr(SOLUTION)]], advisor=adv, policy=pol, cap=1)
    assert out.advisor_calls == 1
    assert out.advisor_cost_usd == pytest.approx(0.01)
    assert any(r["trigger"] == "self" for r in out.advisor_records)
    assert out.solved is True


def test_advisor_cap_blocks_extra_self_calls(make_ctx) -> None:
    adv = ScriptedAdvisor(advice=["a", "b"], cost_per_call=0.01)
    pol = AdvisorPolicy(model="x", max_calls_per_episode=1)
    out = _episode(make_ctx, [[consult("q1")], [consult("q2")], []], advisor=adv, policy=pol, cap=1)
    assert out.advisor_calls == 1  # second consult refused by the cap


def test_engine_trigger_on_fail_streak(make_ctx) -> None:
    # StubVerifier returns FAIL until SOLUTION; three non-improving edits trip fail_streak=2
    adv = ScriptedAdvisor(advice=["hint"], cost_per_call=0.02)
    pol = AdvisorPolicy(model="x", max_calls_per_episode=1, fail_streak=2)
    bad = ToolCall(id="3", name="write_region", args={"name": "solution", "content": "def solve() -> int:\n    return 0  # still wrong"})
    out = _episode(make_ctx, [[bad], [bad], []], advisor=adv, policy=pol, cap=1)
    assert out.advisor_calls == 1
    assert any(r["trigger"] == "engine" for r in out.advisor_records)


def test_off_by_default_no_consult(make_ctx) -> None:
    out = _episode(make_ctx, [[consult("stuck")], []])  # advisor=None, cap=0
    assert out.advisor_calls == 0
    assert out.advisor_records == ()
```

> The `StubVerifier` in `tests/unit/conftest.py` returns `Ok` only for the exact `SOLUTION` text and `Fail`/`Partial` otherwise — confirm this when wiring the engine-trigger test; if it returns `Partial` for holes, the `bad` edit (no hole) yields `Fail`, which is non-improving, so the streak logic holds.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/unit/test_worker.py -k "self_trigger or cap_blocks or engine_trigger or off_by_default" -v`
Expected: FAIL — `run_episode() got an unexpected keyword argument 'advisor'`

- [ ] **Step 3a: Add `verdict_rank` and refactor `outcome_rank` in `crucible/worker.py`**

Add above `outcome_rank` (line 136):

```python
def verdict_rank(verdict: Verdict) -> tuple[int, float]:
    """Comparable progress key for a single verdict. Higher is better:
    Ok > Scored(value) > Partial(-holes) > Fail."""
    match verdict:
        case Ok():
            return (3, 0.0)
        case Scored(value=value):
            return (2, value)
        case Partial(open_holes=holes):
            return (1, -float(len(holes)))
        case Fail():
            return (0, 0.0)
```

Replace the body of `outcome_rank` (lines 140-150) with:

```python
    if outcome.end_reason == "integrity_violation":
        return (0, 0.0)
    return verdict_rank(outcome.final_verdict)
```

- [ ] **Step 3b: Make `_handle_call` return the verdict + handle `consult_advisor`**

Replace `_handle_call` (lines 55-82) with:

```python
def _handle_call(
    call: ToolCall,
    artifact: Artifact,
    verifier: Verifier,
    ctx: RunContext,
    lessons: list[str],
    consult: "Callable[[str, str], str] | None",
) -> tuple[ToolResult, Artifact, bool, Verdict | None]:
    # Validate model output before acting on it: coerce args defensively.
    if call.name == "record_lesson":
        lessons.append(str(call.args.get("text", "")).strip())
        return ToolResult(call.id, "lesson recorded"), artifact, False, None
    if call.name == "consult_advisor":
        if consult is None:
            return ToolResult(call.id, "advisor not enabled"), artifact, False, None
        advice = consult(str(call.args.get("question", "")), "self")
        return ToolResult(call.id, f"[ADVISOR]: {advice}"), artifact, False, None
    if call.name == "search_replace":
        er = search_replace(
            artifact,
            str(call.args.get("file", "")),
            str(call.args.get("old", "")),
            str(call.args.get("new", "")),
        )
    elif call.name == "write_region":
        er = write_region(
            artifact, str(call.args.get("name", "")), str(call.args.get("content", ""))
        )
    else:
        return ToolResult(call.id, f"unknown tool {call.name!r}"), artifact, False, None
    if not er.applied:
        return ToolResult(call.id, f"EDIT REJECTED: {er.error}"), artifact, False, None
    verdict = verifier.verify(er.artifact, ctx)  # verify after every edit (PRD §5)
    return ToolResult(call.id, render(verdict)), er.artifact, True, verdict
```

- [ ] **Step 3c: Extend `EpisodeOutcome` (lines 43-52)**

Add three fields at the end of the dataclass:

```python
    advisor_calls: int = 0
    advisor_cost_usd: float = 0.0
    advisor_records: tuple[dict, ...] = ()
```

- [ ] **Step 3d: Rewrite `run_episode` (lines 85-133)**

```python
def run_episode(
    artifact: Artifact,
    verifier: Verifier,
    ctx: RunContext,
    session: LLMSession,
    budget: EpisodeBudget,
    integrity: IntegrityCheck,
    advisor: "Advisor | None" = None,
    policy: "AdvisorPolicy | None" = None,
    advisor_cap: int = 0,
) -> tuple[EpisodeOutcome, LLMSession]:
    start_artifact = artifact
    lessons: list[str] = []
    edits = 0
    end_reason = "model_stopped"
    last_verdict = verifier.verify(artifact, ctx)
    best_rank = verdict_rank(last_verdict)
    streak = 0
    advisor_records: list[dict] = []
    consult_used = 0

    def consult(question: str, trigger: str) -> str:
        nonlocal consult_used
        if advisor is None or policy is None or consult_used >= advisor_cap:
            return "(advisor budget exhausted)"
        req = AdvisorRequest(
            files=dict(artifact.files),
            editable_regions=[r.name for r in artifact.regions],
            verdict_text=render(last_verdict),
            recent_lessons=list(lessons),
            question=question,
            trigger=trigger,
            scope=policy.scope,
        )
        resp = advisor.consult(req)
        consult_used += 1
        advisor_records.append(
            {"trigger": trigger, "advice": resp.advice, "cost_usd": resp.cost_usd, "ok": resp.ok}
        )
        return resp.advice

    use_consult = consult if (advisor is not None and policy is not None) else None
    calls = session.start(SYSTEM_PROMPT, initial_prompt(artifact, last_verdict))
    turns = 1
    while calls:
        if turns >= budget.turns:
            end_reason = "turn_budget"
            break
        results: list[ToolResult] = []
        last_applied_idx: int | None = None
        for call in calls:
            result, artifact, applied, verdict = _handle_call(
                call, artifact, verifier, ctx, lessons, use_consult
            )
            results.append(result)
            if applied and verdict is not None:
                last_applied_idx = len(results) - 1
                last_verdict = verdict
                rank = verdict_rank(verdict)
                if rank > best_rank:
                    best_rank, streak = rank, 0
                else:
                    streak += 1
            edits += int(applied)
            if edits >= budget.edits:
                break
        # engine-trigger: nudge once when the worker is plateauing
        if (
            policy is not None
            and policy.plateau_trigger
            and streak >= policy.fail_streak
            and consult_used < advisor_cap
            and last_applied_idx is not None
        ):
            advice = consult("", "engine")
            r = results[last_applied_idx]
            results[last_applied_idx] = ToolResult(r.call_id, f"{r.content}\n\n[ADVISOR]: {advice}")
            streak = 0
        if edits >= budget.edits:
            end_reason = "edit_budget"
            break
        calls = session.reply(results)
        turns += 1
    final = verifier.verify(artifact, ctx)
    clean = integrity.check(artifact, ctx)
    solved = clean and isinstance(final, Ok) and not scan_holes(artifact) and verifier.deterministic
    if not clean:
        artifact = start_artifact  # revert to last good (PRD §5/§7)
        end_reason = "integrity_violation"
    if solved:
        end_reason = "solved"
    advisor_cost = sum(r["cost_usd"] for r in advisor_records)
    outcome = EpisodeOutcome(
        artifact=artifact,
        solved=solved,
        turns=turns,
        edits=edits,
        end_reason=end_reason,
        lessons="\n".join(line for line in lessons if line),
        cost_usd=session.cost_usd + advisor_cost,
        final_verdict=final,
        advisor_calls=consult_used,
        advisor_cost_usd=advisor_cost,
        advisor_records=tuple(advisor_records),
    )
    return outcome, session
```

Add the imports near the top of `crucible/worker.py` (after the existing `from crucible.budgets ...` block):

```python
from crucible.advisor import Advisor, AdvisorPolicy, AdvisorRequest
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_worker.py -v`
Expected: PASS (existing tests still pass — `run_episode`'s advisor args default to off; new tests pass)

- [ ] **Step 5: Commit**

```bash
git add crucible/worker.py tests/unit/test_worker.py
git commit -m "feat: worker self-trigger and engine-trigger advisor consults with caps"
```

---

## Task 5: Orchestrator + run_worker wiring and provenance

**Files:**
- Modify: `crucible/worker.py` (`run_worker` signature + per-run cap + episode call + event records)
- Modify: `crucible/orchestrator.py` (`search` params + config dict + pass-through)
- Test: `tests/unit/test_worker.py` (provenance event)

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_worker.py` (uses the existing `harness`, extended to pass advisor):

```python
def test_advisor_consult_event_recorded(tmp_path, make_ctx) -> None:
    from crucible.advisor import AdvisorPolicy, ScriptedAdvisor

    a = Artifact.from_files({"problem.py": PROBLEM})
    integrity = Composite(checks=(ImmutableRegions.freeze(a), DenyTokens()))
    store = Store(tmp_path / "t.db")
    run_id = store.start_run(task_root="/t", verifier_id="stub", model="scripted", config={})
    adv = ScriptedAdvisor(advice=["hint"], cost_per_call=0.01)
    ep1 = [[consult("stuck")], [wr(SOLUTION)]]
    result = run_worker(
        initial=a, verifier=StubVerifier(), ctx=make_ctx(),
        new_session=lambda ordinal: ScriptedSession(ep1),
        store=store, run_id=run_id, index=0,
        episode_budget=EpisodeBudget(),
        run_budget=RunBudget(episodes_per_worker=1),
        integrity=integrity, cancel=threading.Event(), started_at=time.time(),
        advisor_factory=lambda: adv, advisor_policy=AdvisorPolicy(model="x", max_calls_per_episode=1),
    )
    assert result.solution is not None
    rows = store._conn.execute(
        "SELECT payload_json FROM events WHERE run_id=? AND kind='advisor_consult'", (run_id,)
    ).fetchall()
    assert len(rows) == 1
    import json as _json
    assert _json.loads(rows[0][0])["trigger"] == "self"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_worker.py -k advisor_consult_event -v`
Expected: FAIL — `run_worker() got an unexpected keyword argument 'advisor_factory'`

- [ ] **Step 3a: Extend `run_worker` in `crucible/worker.py`**

Add to the `run_worker` signature (after `started_at: float,`):

```python
    advisor_factory: "Callable[[], Advisor] | None" = None,
    advisor_policy: "AdvisorPolicy | None" = None,
```

Inside `run_worker`, before the `for ordinal` loop (after `episodes = 0`), add:

```python
    advisor = advisor_factory() if advisor_factory is not None else None
    run_calls_left = (
        advisor_policy.max_calls_per_run
        if advisor_policy is not None and advisor_policy.max_calls_per_run is not None
        else None
    )
```

Replace the `out, session = run_episode(...)` call (line 223) with:

```python
        if advisor is not None and advisor_policy is not None:
            cap = advisor_policy.max_calls_per_episode
            if run_calls_left is not None:
                cap = min(cap, run_calls_left)
        else:
            cap = 0
        out, session = run_episode(
            artifact, verifier, ctx, new_session(ordinal), episode_budget, integrity,
            advisor=advisor if cap > 0 else None, policy=advisor_policy, advisor_cap=cap,
        )
        if run_calls_left is not None:
            run_calls_left -= out.advisor_calls
```

After `ep_id = store.add_episode(...)` (line 229-236), add the event writes:

```python
        for rec in out.advisor_records:
            store.add_event(
                run_id,
                "advisor_consult",
                {"worker": index, "episode": ordinal, "trigger": rec["trigger"],
                 "advice": rec["advice"], "cost_usd": rec["cost_usd"], "ok": rec["ok"]},
            )
```

- [ ] **Step 3b: Thread advisor through `search` in `crucible/orchestrator.py`**

Add imports near the top:

```python
from crucible.advisor import Advisor, AdvisorPolicy
```

Add two params to `search` (after `run_budget: RunBudget | None = None,`):

```python
    advisor_factory: Callable[[], Advisor] | None = None,
    advisor_policy: AdvisorPolicy | None = None,
```

In the `config={...}` dict passed to `store.start_run` (lines 64-73), add a key:

```python
            "advisor": (
                None if advisor_policy is None else {
                    "model": advisor_policy.model,
                    "max_calls_per_episode": advisor_policy.max_calls_per_episode,
                    "max_calls_per_run": advisor_policy.max_calls_per_run,
                    "plateau_trigger": advisor_policy.plateau_trigger,
                    "fail_streak": advisor_policy.fail_streak,
                    "scope": advisor_policy.scope,
                }
            ),
```

In the `one(i)` closure, pass the advisor through to `run_worker`:

```python
        return run_worker(
            ...,  # existing kwargs unchanged
            advisor_factory=advisor_factory,
            advisor_policy=advisor_policy,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_worker.py -k advisor_consult_event -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crucible/worker.py crucible/orchestrator.py tests/unit/test_worker.py
git commit -m "feat: thread advisor through orchestrator and record consult events"
```

---

## Task 6: SDK `run()` advisor kwarg

**Files:**
- Modify: `crucible/__init__.py` (`run` signature + normalization + factory + extra_tools)
- Test: `tests/unit/test_sdk.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_sdk.py` (uses the existing `session_factory` test seam pattern; confirm how that file builds sessions and mirror it):

```python
def test_run_advisor_string_is_normalized(monkeypatch, tmp_path) -> None:
    import crucible
    captured = {}

    def fake_run_search(**kwargs):
        captured.update(kwargs)
        from crucible.orchestrator import SearchResult
        from crucible.artifact import Artifact
        a = Artifact.from_files({"problem.py": "x"})
        return SearchResult(solution=None, best_partial=a, run_id=1, winner=None, cost_usd=0.0)

    monkeypatch.setattr(crucible, "run_search", fake_run_search)
    crucible.run(
        task=_make_task(tmp_path),  # reuse this file's existing task helper
        verifier=_StubDet(),        # reuse this file's deterministic stub verifier
        model="gpt-4o",
        advisor="claude-opus-4-8",
        sandbox="subprocess",
        db=str(tmp_path / "x.db"),
    )
    assert captured["advisor_policy"] is not None
    assert captured["advisor_policy"].model == "claude-opus-4-8"
    assert captured["advisor_factory"] is not None
```

> If `test_sdk.py` lacks `_make_task` / `_StubDet`, reuse whatever task + deterministic verifier helpers it already defines (the file already calls `run()` with a `session_factory` seam — copy that setup).

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_sdk.py -k advisor_string -v`
Expected: FAIL — `run() got an unexpected keyword argument 'advisor'`

- [ ] **Step 3: Implement in `crucible/__init__.py`**

Add to imports:

```python
from crucible.advisor import Advisor, AdvisorPolicy
```

Add params to `run` (after `base_url` on line 58):

```python
    advisor: str | AdvisorPolicy | None = None,
    advisor_factory: Callable[[], Advisor] | None = None,  # test seam
```

After the sandbox-factory block and before building `sf` (around line 90), add policy normalization:

```python
    policy: AdvisorPolicy | None
    if advisor is None:
        policy = None
    elif isinstance(advisor, str):
        policy = AdvisorPolicy(model=advisor)
    else:
        policy = advisor

    af = advisor_factory
    if policy is not None and af is None:
        _policy = policy
        _adv_base = base_url

        def af() -> Advisor:  # noqa: E306
            from crucible.advisor import LLMAdvisor
            return LLMAdvisor(_policy, base_url=_adv_base)
```

Update the default session factory branch (lines 93-102) so worker sessions advertise the consult tool when an advisor is configured:

```python
    else:
        from crucible.llm import CONSULT_ADVISOR_SCHEMA
        from crucible.providers import make_session  # optional extras imported lazily

        _base_url = base_url
        _model = model
        _extra = [CONSULT_ADVISOR_SCHEMA] if policy is not None else []

        def _make(*_: int) -> LLMSession:  # type: ignore[unused-argument]
            return make_session(_model, base_url=_base_url, extra_tools=_extra)

        sf = _make
```

Add the two kwargs to the `run_search(...)` call (line 104):

```python
        advisor_factory=af,
        advisor_policy=policy,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_sdk.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crucible/__init__.py tests/unit/test_sdk.py
git commit -m "feat: advisor kwarg on run() with policy normalization and tool wiring"
```

---

## Task 7: CLI flags

**Files:**
- Modify: `crucible/cli.py` (`build_parser` + `main` run branch)
- Test: `tests/unit/test_cli.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_cli.py` (mirror the existing `run_fn` injection pattern this file already uses):

```python
def test_cli_passes_advisor(monkeypatch) -> None:
    from crucible.cli import main

    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        from crucible.orchestrator import SearchResult
        from crucible.artifact import Artifact
        return SearchResult(
            solution=None, best_partial=Artifact.from_files({"problem.py": "x"}),
            run_id=1, winner=None, cost_usd=0.0,
        )

    code = main(
        ["run", "problem.py", "--editable", "solution", "--verifier", "pytest:tests/",
         "--advisor", "claude-opus-4-8", "--advisor-max-calls", "2", "--sandbox", "subprocess"],
        run_fn=fake_run,
    )
    assert code in (0, 2)
    pol = captured["advisor"]
    assert pol is not None and pol.model == "claude-opus-4-8" and pol.max_calls_per_episode == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_cli.py -k advisor -v`
Expected: FAIL — `unrecognized arguments: --advisor`

- [ ] **Step 3a: Add flags in `build_parser` (after line 73, in the `run` subparser)**

```python
    r.add_argument("--advisor", default=None, help="advisor model name (enables shepherding)")
    r.add_argument("--advisor-max-calls", type=int, default=1, help="max advisor calls per episode")
    r.add_argument("--advisor-fail-streak", type=int, default=3, help="non-improving edits before an engine consult")
```

- [ ] **Step 3b: Build the policy in `main` and pass it (in the `run` branch, lines 227-239)**

Before the `fn(...)` call:

```python
        from crucible.advisor import AdvisorPolicy
        advisor_policy = (
            None if args.advisor is None
            else AdvisorPolicy(
                model=args.advisor,
                max_calls_per_episode=args.advisor_max_calls,
                fail_streak=args.advisor_fail_streak,
            )
        )
```

Add `advisor=advisor_policy,` to the `fn(...)` kwargs.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crucible/cli.py tests/unit/test_cli.py
git commit -m "feat: CLI --advisor / --advisor-max-calls / --advisor-fail-streak"
```

---

## Task 8: Example passthroughs + README

**Files:**
- Modify: `examples/sidon/run_sidon.py`, `examples/chem/run_chem.py`
- Modify: `README.md`

- [ ] **Step 1: Add `--advisor` to `examples/sidon/run_sidon.py`**

After the `--sandbox` argument (line 60), add:

```python
parser.add_argument("--advisor", default=None, help="frontier advisor model (LLM shepherding)")
```

Build a policy after `args = parser.parse_args()` (line 61):

```python
from crucible.advisor import AdvisorPolicy  # noqa: E402
advisor = AdvisorPolicy(model=args.advisor) if args.advisor else None
```

Add `advisor=advisor,` to the `run(...)` call (line 66-75), and update the print on line 64:

```python
print(f"Model: {args.model}  |  Workers: {args.workers}  |  Target: {args.target}"
      f"  |  Advisor: {args.advisor or 'off'}")
```

- [ ] **Step 2: Add the same `--advisor` passthrough to `examples/chem/run_chem.py`**

Read `examples/chem/run_chem.py` first, then mirror Step 1 exactly: add the `--advisor` argparse argument, construct `advisor = AdvisorPolicy(model=args.advisor) if args.advisor else None`, and pass `advisor=advisor` into that file's `run(...)` call.

- [ ] **Step 3: Smoke-check the examples parse (no API call)**

Run: `.venv/bin/python examples/sidon/run_sidon.py --help`
Expected: help text lists `--advisor`.

- [ ] **Step 4: Add a README section**

Insert a new section after "## Model providers" (after line 176) in `README.md`:

```markdown
## LLM shepherding (optional advisor)

A small local worker can be *shepherded* by a stronger frontier "advisor" model,
following [Fireworks' Frontier Advisors](https://fireworks.ai/blog/open-source-agents-frontier-advisors)
result (a cheap worker beating a frontier-only baseline at a fraction of the cost
by consulting sparsely, ~0.83×/task). The advisor returns **text steering only** —
it never edits the artifact or touches the store; the worker applies suggestions
through its own validated edit tools. It is **off by default**; when unset, behavior
is byte-identical to a plain run.

Two trigger paths, both capped:
- **Self-trigger** — the worker calls a `consult_advisor` tool when it judges itself stuck.
- **Engine-trigger** — the engine force-consults once after `fail_streak` non-improving edits.

```python
from crucible.advisor import AdvisorPolicy

result = run(
    task=task, verifier=verifier,
    model="gemma-3-12b-qat",                 # tiny local worker (OpenAI-compatible)
    base_url="http://localhost:8000/v1",
    advisor="claude-opus-4-8",               # frontier advisor (or an AdvisorPolicy)
    workers=5,
)
```

```bash
crucible run problem.py --editable solution --verifier pytest:tests/ \
    --model gemma-3-12b-qat --advisor claude-opus-4-8 --advisor-max-calls 1
```

Every consult is recorded append-only as an `advisor_consult` event and shows up in
`crucible reasoning` as an `[ADVISOR]: …` turn — so you can count consults-per-task
and see exactly where the advisor changed the worker's trajectory.
```

- [ ] **Step 5: Commit**

```bash
git add examples/sidon/run_sidon.py examples/chem/run_chem.py README.md
git commit -m "docs: --advisor passthrough in examples and README shepherding section"
```

---

## Task 9: Full verification gate

**Files:** none (verification only)

- [ ] **Step 1: Run the full unit suite**

Run: `.venv/bin/pytest tests/unit -q`
Expected: all pass (no regressions; new advisor/worker/provider/sdk/cli tests green).

- [ ] **Step 2: Run the type gate**

Run: `.venv/bin/pyright`
Expected: 0 errors. Fix any type issues surfaced by the new code (notably: the forward-ref string annotations in `worker.py` resolve because `Advisor`/`AdvisorPolicy`/`AdvisorRequest` are imported at module top; the `match` in `verdict_rank` is exhaustive over the `Verdict` union).

- [ ] **Step 3: Confirm off-by-default invariance**

Run: `.venv/bin/pytest tests/unit/test_worker.py -k off_by_default -v`
Expected: PASS — proves `advisor=None` performs zero consults and records nothing.

- [ ] **Step 4: Final commit (if any fixes were needed)**

```bash
git add -A
git commit -m "chore: pass full unit + pyright gates for LLM shepherding"
```

---

## Self-Review

**Spec coverage:**
- §3.1 advisor module → Task 1 (+ `LLMAdvisor` in Task 3). ✓
- §3.2 `make_advisor_session` → Task 3. ✓
- §3.3 worker integration (self + engine + caps + outcome fields) → Task 4. ✓
- §3.4 orchestrator + provenance events + config → Task 5. ✓
- §3.5 config surface (SDK kwarg, CLI flags, examples) → Tasks 6, 7, 8. ✓
- §2 invariants: no-datastore (advisor returns text; events written by worker code) ✓; validate output (`sanitize_advice`, non-empty check) ✓; graceful degradation (`LLMAdvisor.consult` try/except) Task 3 ✓; off-by-default byte-identical (Task 4 test) ✓; no new deps ✓.
- §1 outcome #4 (consults-per-task reportable) → `advisor_consult` events, Task 5. ✓
- §6 testing → tests embedded in Tasks 1,3,4,5,6,7 + Task 9 gate. ✓

**Deviations from spec (intentional, noted here):**
- Spec §3.1 listed `recent_turns` in `AdvisorRequest`; replaced with `question` + `recent_lessons` to avoid provider-specific transcript parsing (simpler, token-cheaper). Same intent (curated slice).

**Placeholder scan:** none — every code step contains complete code. Two steps reference reusing existing test helpers (`test_sdk.py`, `test_cli.py`); those are flagged with explicit "mirror the existing pattern" instructions because the helper names live in files not yet read.

**Type consistency:** `AdvisorPolicy`, `AdvisorRequest`, `AdvisorResponse`, `Advisor`, `LLMAdvisor`, `ScriptedAdvisor`, `make_advisor_session`, `sanitize_advice`, `render_request`, `verdict_rank`, `CONSULT_ADVISOR_SCHEMA`, and the `run_episode(..., advisor=, policy=, advisor_cap=)` / `run_worker(..., advisor_factory=, advisor_policy=)` / `search(..., advisor_factory=, advisor_policy=)` signatures are used identically across Tasks 1–8. ✓
