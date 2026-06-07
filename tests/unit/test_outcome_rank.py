import pytest

from crucible.artifact import Artifact, Hole
from crucible.verify import Fail, Ok, Partial, Scored
from crucible.worker import EpisodeOutcome, outcome_rank

pytestmark = pytest.mark.unit


def _outcome(verdict, end_reason="model_stopped") -> EpisodeOutcome:
    a = Artifact.from_files({"m.smi": "CO\n"})
    return EpisodeOutcome(
        artifact=a,
        solved=False,
        turns=1,
        edits=1,
        end_reason=end_reason,
        lessons="",
        cost_usd=0.0,
        final_verdict=verdict,
    )


def test_tiers_are_ordered_ok_gt_scored_gt_partial_gt_fail() -> None:
    a = Artifact.from_files({"m.smi": "CO\n"})
    ok = outcome_rank(_outcome(Ok(produced=a)))
    scored = outcome_rank(_outcome(Scored(produced=a, value=5.0)))
    partial = outcome_rank(_outcome(Partial(open_holes=(), feedback="")))
    fail = outcome_rank(_outcome(Fail(feedback="x")))
    assert ok > scored > partial > fail


def test_higher_score_outranks_lower() -> None:
    a = Artifact.from_files({"m.smi": "CO\n"})
    assert outcome_rank(_outcome(Scored(produced=a, value=9.0))) > outcome_rank(
        _outcome(Scored(produced=a, value=1.0))
    )


def test_fewer_holes_outrank_more_within_partial() -> None:
    h = Hole(file="m.smi", line=0, kind="sentinel", text="x")
    assert outcome_rank(_outcome(Partial(open_holes=(), feedback=""))) > outcome_rank(
        _outcome(Partial(open_holes=(h, h), feedback=""))
    )


def test_integrity_violation_ranks_at_bottom() -> None:
    a = Artifact.from_files({"m.smi": "CO\n"})
    violated = outcome_rank(
        _outcome(Scored(produced=a, value=9.0), end_reason="integrity_violation")
    )
    assert violated == (0, 0.0)
