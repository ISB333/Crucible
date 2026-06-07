import pytest

from crucible.artifact import Artifact
from crucible.verifiers.rubric import Rubric
from crucible.verify import Fail, Ok, Verifier

pytestmark = pytest.mark.unit

CONTENT = "# crucible:region start name=s\nprint('hi')\n# crucible:region end\n"


def make() -> Artifact:
    return Artifact.from_files({"f.py": CONTENT})


def test_rubric_is_advisory() -> None:
    r = Rubric(
        spec="clear and idiomatic",
        judge=lambda prompt: '{"score": 1.0, "feedback": "fine"}',
    )
    assert r.deterministic is False
    assert isinstance(r, Verifier)


def test_score_above_threshold_is_ok(make_ctx) -> None:
    r = Rubric(spec="x", threshold=0.8, judge=lambda p: '{"score": 0.9, "feedback": "good"}')
    assert isinstance(r.verify(make(), make_ctx()), Ok)


def test_score_below_threshold_is_fail_with_feedback(make_ctx) -> None:
    r = Rubric(
        spec="x",
        threshold=0.8,
        judge=lambda p: '{"score": 0.3, "feedback": "unclear naming"}',
    )
    v = r.verify(make(), make_ctx())
    assert isinstance(v, Fail) and "unclear naming" in v.feedback


def test_malformed_judge_output_is_fail_not_crash(make_ctx) -> None:
    r = Rubric(spec="x", judge=lambda p: "I think it's great!")
    v = r.verify(make(), make_ctx())
    assert isinstance(v, Fail) and "judge" in v.feedback


def test_judge_prompt_contains_spec_and_files() -> None:
    seen: list[str] = []

    def judge(prompt: str) -> str:
        seen.append(prompt)
        return '{"score": 1.0, "feedback": ""}'

    Rubric(spec="must be idiomatic", judge=judge).verify(make(), None)  # type: ignore[arg-type]
    assert "must be idiomatic" in seen[0] and "print('hi')" in seen[0]


# Finding 2 — REJECTED: substituted values in .format are not re-scanned, so
# braces inside spec cannot cause a KeyError.  Confirmed by:
#   python3 -c 'print("{s}".format(s="x {y} z"))'  # → "x {y} z", no error
def test_spec_with_braces_does_not_crash(make_ctx) -> None:
    r = Rubric(
        spec="must handle {config} and {{escaped}}",
        judge=lambda p: '{"score": 1.0, "feedback": "ok"}',
    )
    assert isinstance(r.verify(make(), make_ctx()), Ok)


# Finding 3 — boundary and regression tests


def test_score_at_threshold_is_ok(make_ctx) -> None:
    r = Rubric(spec="x", threshold=0.8, judge=lambda p: '{"score": 0.8, "feedback": ""}')
    assert isinstance(r.verify(make(), make_ctx()), Ok)


def test_bool_score_is_rejected(make_ctx) -> None:
    # Old code: float(True) == 1.0 >= 0.8 → Ok (wrong).
    # New code: bool guard fires before float() → Fail.
    r = Rubric(spec="x", threshold=0.8, judge=lambda p: '{"score": true, "feedback": ""}')
    v = r.verify(make(), make_ctx())
    assert isinstance(v, Fail) and "non-numeric" in v.feedback


def test_out_of_range_score_is_rejected(make_ctx) -> None:
    r = Rubric(spec="x", threshold=0.8, judge=lambda p: '{"score": 1.5, "feedback": ""}')
    v = r.verify(make(), make_ctx())
    assert isinstance(v, Fail) and "out of range" in v.feedback
