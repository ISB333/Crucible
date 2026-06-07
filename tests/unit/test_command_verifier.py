import pytest

from crucible.artifact import Artifact
from crucible.sandbox import SandboxResult
from crucible.verifiers.command import Command
from crucible.verify import Fail, Ok, Partial
from tests.unit.conftest import FakeSandbox

pytestmark = pytest.mark.unit

SOLVED = "# crucible:region start name=s\nx = 1\n# crucible:region end\n"
HOLEY = "# crucible:region start name=s\nraise NotImplementedError\n# crucible:region end\n"


def test_exit_zero_no_holes_is_ok(make_ctx) -> None:
    ctx = make_ctx()
    a = Artifact.from_files({"f.py": SOLVED})
    v = Command("true").verify(a, ctx)
    assert isinstance(v, Ok) and v.produced is a


def test_exit_zero_with_holes_is_partial(make_ctx) -> None:
    a = Artifact.from_files({"f.py": HOLEY})
    v = Command("true").verify(a, make_ctx())
    assert isinstance(v, Partial)
    assert v.open_holes[0].kind == "not_implemented"


def test_nonzero_exit_is_fail_with_feedback(make_ctx) -> None:
    sandbox = FakeSandbox(SandboxResult(exit_code=1, stdout="", stderr="AssertionError: boom"))
    a = Artifact.from_files({"f.py": SOLVED})
    v = Command("make check").verify(a, make_ctx(sandbox))
    assert isinstance(v, Fail) and "AssertionError: boom" in v.feedback


def test_timeout_is_fail_timeout(make_ctx) -> None:
    sandbox = FakeSandbox(SandboxResult(exit_code=124, stdout="", stderr="", timed_out=True))
    a = Artifact.from_files({"f.py": SOLVED})
    v = Command("slow", timeout_s=1).verify(a, make_ctx(sandbox))
    assert isinstance(v, Fail) and "timeout" in v.feedback


def test_command_runs_in_materialized_workspace(make_ctx) -> None:
    sandbox = FakeSandbox()
    ctx = make_ctx(sandbox)
    a = Artifact.from_files({"f.py": SOLVED})
    Command("make check").verify(a, ctx)
    argv, cwd = sandbox.calls[0]
    assert argv == ("make", "check")
    assert cwd == ctx.materialize(a)


def test_verifier_id() -> None:
    assert Command("make check").verifier_id == "cmd:make check"


def test_timeout_feedback_keeps_partial_output(make_ctx) -> None:
    sandbox = FakeSandbox(
        SandboxResult(exit_code=124, stdout="ran 3 tests so far", stderr="", timed_out=True)
    )
    a = Artifact.from_files({"f.py": SOLVED})
    v = Command("slow", timeout_s=1).verify(a, make_ctx(sandbox))
    assert isinstance(v, Fail) and "timeout" in v.feedback
    assert "ran 3 tests so far" in v.feedback


def test_nonzero_exit_with_holes_is_partial(make_ctx) -> None:
    sandbox = FakeSandbox(SandboxResult(exit_code=1, stdout="", stderr="boom"))
    a = Artifact.from_files({"f.py": HOLEY})
    v = Command("make check").verify(a, make_ctx(sandbox))
    assert isinstance(v, Partial)  # holes dominate even over failure
    assert "boom" in v.feedback


def test_unquoted_shell_metacharacters_rejected() -> None:
    with pytest.raises(ValueError):
        Command("make check && echo done")
    Command("sh -c 'make check && echo done'")  # quoted single token — fine
