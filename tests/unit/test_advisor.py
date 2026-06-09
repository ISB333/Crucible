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
