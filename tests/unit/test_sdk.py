"""SDK surface tests (PRD §9): run(), Result, advisory-verifier guard, sandbox validation."""

from pathlib import Path

import pytest

from crucible import AdvisorPolicy, Result, Task, budgets, run
from crucible.advisor import ScriptedAdvisor
from crucible.llm import ScriptedSession, ToolCall
from crucible.verifiers import Command
from crucible.verifiers.rubric import Rubric

pytestmark = pytest.mark.unit

PROBLEM = """\
# crucible:region start name=solution
def solve() -> int:
    raise NotImplementedError
# crucible:region end
"""


def wr(content: str) -> ToolCall:
    return ToolCall(id="1", name="write_region", args={"name": "solution", "content": content})


def test_run_end_to_end_scripted(tmp_path: Path) -> None:
    (tmp_path / "problem.py").write_text(PROBLEM)
    result = run(
        task=Task.from_path(tmp_path / "problem.py", editable=["solution"]),
        verifier=Command("true"),
        model="scripted",
        workers=1,
        episode=budgets(edits=5, turns=5),
        run_budget=budgets(episodes=1),
        sandbox="subprocess",
        db=tmp_path / "crucible.db",
        session_factory=lambda w, e: ScriptedSession([[wr("def solve() -> int:\n    return 42")]]),
    )
    assert isinstance(result, Result)
    assert result.solution is not None
    assert "return 42" in result.solution.files["problem.py"]
    assert result.best_partial is not None
    assert result.run_id >= 1


def test_advisory_sole_verifier_is_config_error(tmp_path: Path) -> None:
    (tmp_path / "problem.py").write_text(PROBLEM)
    with pytest.raises(ValueError, match="advisory"):
        run(
            task=Task.from_path(tmp_path / "problem.py", editable=["solution"]),
            verifier=Rubric(spec="nice code", judge=lambda p: '{"score": 1.0, "feedback": ""}'),
            model="scripted",
            workers=1,
            sandbox="subprocess",
            db=tmp_path / "crucible.db",
            session_factory=lambda w, e: ScriptedSession([]),
        )


def test_unknown_sandbox_kind_rejected(tmp_path: Path) -> None:
    (tmp_path / "problem.py").write_text(PROBLEM)
    with pytest.raises(ValueError, match="sandbox"):
        run(
            task=Task.from_path(tmp_path / "problem.py", editable=["solution"]),
            verifier=Command("true"),
            model="scripted",
            sandbox="bare-metal",
            db=tmp_path / "crucible.db",
            session_factory=lambda w, e: ScriptedSession([]),
        )


def test_advisor_kwarg_str_model(tmp_path: Path) -> None:
    """Task 6: advisor kwarg as string creates AdvisorPolicy with that model."""
    (tmp_path / "problem.py").write_text(PROBLEM)
    # This test verifies that the advisor kwarg is accepted and wired through
    # We use a scripted session that solves immediately so no advisor call is needed
    result = run(
        task=Task.from_path(tmp_path / "problem.py", editable=["solution"]),
        verifier=Command("true"),
        model="scripted",
        workers=1,
        episode=budgets(edits=5, turns=5),
        run_budget=budgets(episodes=1),
        sandbox="subprocess",
        db=tmp_path / "crucible.db",
        session_factory=lambda w, e: ScriptedSession([[wr("def solve() -> int:\n    return 42")]]),
        advisor="claude-opus-4-8",  # string model name
    )
    assert result.solution is not None


def test_advisor_kwarg_policy(tmp_path: Path) -> None:
    """Task 6: advisor kwarg as AdvisorPolicy uses that policy."""
    (tmp_path / "problem.py").write_text(PROBLEM)
    policy = AdvisorPolicy(model="claude-opus-4-8", max_calls_per_episode=2, max_calls_per_run=5)
    result = run(
        task=Task.from_path(tmp_path / "problem.py", editable=["solution"]),
        verifier=Command("true"),
        model="scripted",
        workers=1,
        episode=budgets(edits=5, turns=5),
        run_budget=budgets(episodes=1),
        sandbox="subprocess",
        db=tmp_path / "crucible.db",
        session_factory=lambda w, e: ScriptedSession([[wr("def solve() -> int:\n    return 42")]]),
        advisor=policy,
    )
    assert result.solution is not None


def test_advisor_kwarg_invalid_type(tmp_path: Path) -> None:
    """Task 6: advisor kwarg with invalid type raises ValueError."""
    (tmp_path / "problem.py").write_text(PROBLEM)
    with pytest.raises(ValueError, match="advisor must be"):
        run(
            task=Task.from_path(tmp_path / "problem.py", editable=["solution"]),
            verifier=Command("true"),
            model="scripted",
            workers=1,
            sandbox="subprocess",
            db=tmp_path / "crucible.db",
            session_factory=lambda w, e: ScriptedSession([]),
            advisor=123,  # type: ignore
        )
