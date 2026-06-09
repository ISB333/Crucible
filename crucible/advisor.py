"""Optional frontier advisor (LLM shepherding). The advisor returns TEXT only;
the worker applies any suggestion through its own validated edit tools. The advisor
never touches the artifact or the store (Ananke #9/#10) and never raises into the
Ralph loop (#12)."""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from crucible.providers import make_advisor_session

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
