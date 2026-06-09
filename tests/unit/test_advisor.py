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


def test_llm_advisor_validates_and_bills(monkeypatch) -> None:
    from crucible import advisor as adv_mod

    class FakeSession:
        def __init__(self) -> None:
            self.seen: tuple[str, str] | None = None
            self._cost = 0.0
        def complete(self, system: str, user: str) -> str:
            self.seen = (system, user)
            self._cost = 0.05
            return "  use a crucible:hole trick  "  # whitespace + marker to sanitize
        @property
        def cost_usd(self) -> float:
            return self._cost

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
