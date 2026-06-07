import pytest

from crucible.artifact import Artifact
from crucible.verify import Fail, Ok, Partial, Scored, Verdict, render

pytestmark = pytest.mark.unit


def _art() -> Artifact:
    return Artifact.from_files({"m.smi": "CCO\n"})


def test_scored_carries_value_and_artifact() -> None:
    s = Scored(produced=_art(), value=1.5, feedback="below target")
    assert s.value == 1.5
    assert s.produced.files["m.smi"] == "CCO\n"


def test_scored_is_a_verdict() -> None:
    s: Verdict = Scored(produced=_art(), value=0.0)
    assert isinstance(s, Scored)


def test_render_scored_shows_value() -> None:
    out = render(Scored(produced=_art(), value=2.3456, feedback="keep going"))
    assert "SCORED" in out and "2.3456" in out and "keep going" in out


def test_render_still_handles_the_other_three() -> None:
    assert "OK" in render(Ok(produced=_art()))
    assert "PARTIAL" in render(Partial(open_holes=(), feedback=""))
    assert "FAIL" in render(Fail(feedback="boom"))
