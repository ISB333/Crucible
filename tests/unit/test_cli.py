from pathlib import Path

import pytest

from crucible import AdvisorPolicy
from crucible.artifact import Artifact
from crucible.budgets import RunBudget
from crucible.cli import main, parse_run_budget, parse_verifier
from crucible.orchestrator import SearchResult
from crucible.verifiers import Command, Lean, Pyright, Pytest

pytestmark = pytest.mark.unit

PROBLEM = """\
# crucible:region start name=solution
def solve() -> int:
    raise NotImplementedError
# crucible:region end
"""


def test_parse_verifier_specs() -> None:
    assert parse_verifier("pytest:tests/") == Pytest(suite="tests/")
    assert parse_verifier("pyright:strict") == Pyright(strict=True)
    assert parse_verifier("pyright:standard") == Pyright(strict=False)
    assert parse_verifier("cmd:make check") == Command("make check")
    assert parse_verifier("lean") == Lean()
    with pytest.raises(ValueError):
        parse_verifier("voodoo:chicken")


def test_parse_run_budget() -> None:
    assert parse_run_budget("2h,200usd") == RunBudget(wall_clock_s=7200.0, usd=200.0)
    assert parse_run_budget("30m") == RunBudget(wall_clock_s=1800.0)
    with pytest.raises(ValueError):
        parse_run_budget("200kg")
    with pytest.raises(ValueError, match="usd"):
        parse_run_budget("usd")


def solved_result(tmp_path: Path) -> SearchResult:
    solved_text = PROBLEM.replace("raise NotImplementedError", "return 42")
    a = Artifact.from_files({"problem.py": solved_text})
    return SearchResult(solution=a, best_partial=a, run_id=1, winner=0, cost_usd=0.0)


def test_main_solved_writes_out_and_exits_zero(tmp_path: Path, capsys) -> None:
    (tmp_path / "problem.py").write_text(PROBLEM)
    out = tmp_path / "out"
    code = main(
        [
            "run",
            str(tmp_path / "problem.py"),
            "--editable",
            "solution",
            "--verifier",
            "cmd:true",
            "--workers",
            "1",
            "--out",
            str(out),
            "--db",
            str(tmp_path / "c.db"),
        ],
        run_fn=lambda **kw: solved_result(tmp_path),
    )
    assert code == 0
    assert "return 42" in (out / "problem.py").read_text()
    assert "SOLVED" in capsys.readouterr().out


def test_main_unsolved_exits_two(tmp_path: Path, capsys) -> None:
    (tmp_path / "problem.py").write_text(PROBLEM)
    a = Artifact.from_files({"problem.py": PROBLEM})
    unsolved = SearchResult(solution=None, best_partial=a, run_id=1, winner=None, cost_usd=0.0)
    code = main(
        [
            "run",
            str(tmp_path / "problem.py"),
            "--editable",
            "solution",
            "--verifier",
            "cmd:true",
            "--out",
            str(tmp_path / "out"),
            "--db",
            str(tmp_path / "c.db"),
        ],
        run_fn=lambda **kw: unsolved,
    )
    assert code == 2
    assert "best partial" in capsys.readouterr().out.lower()


def test_main_advisor_flags_build_policy(tmp_path: Path, capsys) -> None:
    """Task 7: CLI --advisor flags build AdvisorPolicy and pass to run()."""
    (tmp_path / "problem.py").write_text(PROBLEM)
    out = tmp_path / "out"
    captured_kwargs = {}

    def capture_run(**kw):
        captured_kwargs.update(kw)
        return solved_result(tmp_path)

    code = main(
        [
            "run",
            str(tmp_path / "problem.py"),
            "--editable",
            "solution",
            "--verifier",
            "cmd:true",
            "--workers",
            "1",
            "--out",
            str(out),
            "--db",
            str(tmp_path / "c.db"),
            "--advisor",
            "claude-opus-4-8",
            "--advisor-max-calls",
            "5",
            "--advisor-fail-streak",
            "4",
        ],
        run_fn=capture_run,
    )
    assert code == 0
    assert "advisor" in captured_kwargs
    policy = captured_kwargs["advisor"]
    assert isinstance(policy, AdvisorPolicy)
    assert policy.model == "claude-opus-4-8"
    assert policy.max_calls_per_run == 5
    assert policy.fail_streak == 4


def test_main_advisor_defaults_when_not_specified(tmp_path: Path, capsys) -> None:
    """Task 7: when --advisor is not specified, advisor=None."""
    (tmp_path / "problem.py").write_text(PROBLEM)
    captured_kwargs = {}

    def capture_run(**kw):
        captured_kwargs.update(kw)
        return solved_result(tmp_path)

    main(
        [
            "run",
            str(tmp_path / "problem.py"),
            "--editable",
            "solution",
            "--verifier",
            "cmd:true",
            "--workers",
            "1",
            "--out",
            str(tmp_path / "out"),
            "--db",
            str(tmp_path / "c.db"),
        ],
        run_fn=capture_run,
    )
    assert captured_kwargs.get("advisor") is None
