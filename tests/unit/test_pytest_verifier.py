import pytest

from crucible.artifact import Artifact
from crucible.sandbox import SandboxResult
from crucible.verifiers.pytest_verifier import Pytest, summary_counts
from crucible.verify import Fail, Ok, Partial
from tests.unit.conftest import FakeSandbox

pytestmark = pytest.mark.unit

SOLVED = "# crucible:region start name=s\ndef f():\n    return 42\n# crucible:region end\n"
HOLEY = (
    "# crucible:region start name=s\ndef f():\n"
    "    raise NotImplementedError\n# crucible:region end\n"
)


def test_summary_counts_parses_pytest_tail() -> None:
    out = "....\n4 passed, 1 skipped, 2 xfailed in 0.12s\n"
    assert summary_counts(out) == {"passed": 4, "skipped": 1, "xfailed": 2}


def test_green_suite_is_ok(make_ctx) -> None:
    sandbox = FakeSandbox(SandboxResult(0, "3 passed in 0.05s\n", ""))
    a = Artifact.from_files({"solution.py": SOLVED})
    assert isinstance(Pytest(suite="tests/").verify(a, make_ctx(sandbox)), Ok)


def test_green_with_skips_is_fail(make_ctx) -> None:
    sandbox = FakeSandbox(SandboxResult(0, "2 passed, 1 skipped in 0.05s\n", ""))
    a = Artifact.from_files({"solution.py": SOLVED})
    v = Pytest(suite="tests/").verify(a, make_ctx(sandbox))
    assert isinstance(v, Fail) and "skip" in v.feedback


def test_red_suite_is_fail_with_output(make_ctx) -> None:
    out = "FAILED test_x.py::test_a - AssertionError\n1 failed in 0.1s\n"
    sandbox = FakeSandbox(SandboxResult(1, out, ""))
    a = Artifact.from_files({"solution.py": SOLVED})
    v = Pytest(suite="tests/").verify(a, make_ctx(sandbox))
    assert isinstance(v, Fail) and "AssertionError" in v.feedback


def test_holes_make_it_partial(make_ctx) -> None:
    sandbox = FakeSandbox(SandboxResult(1, "1 failed in 0.1s\n", ""))
    a = Artifact.from_files({"solution.py": HOLEY})
    v = Pytest(suite="tests/").verify(a, make_ctx(sandbox))
    assert isinstance(v, Partial) and v.open_holes[0].kind == "not_implemented"


def test_timeout_is_fail(make_ctx) -> None:
    sandbox = FakeSandbox(SandboxResult(124, "", "", timed_out=True))
    a = Artifact.from_files({"solution.py": SOLVED})
    v = Pytest(suite="tests/").verify(a, make_ctx(sandbox))
    assert isinstance(v, Fail) and "timeout" in v.feedback


def test_invokes_pytest_with_suite(make_ctx) -> None:
    sandbox = FakeSandbox(SandboxResult(0, "1 passed in 0.01s\n", ""))
    ctx = make_ctx(sandbox)
    a = Artifact.from_files({"solution.py": SOLVED})
    Pytest(suite="tests/").verify(a, ctx)
    argv, _ = sandbox.calls[0]
    assert "pytest" in argv and "tests/" in argv
    assert "-p" in argv and "no:cacheprovider" in argv and "-rA" in argv


def test_spoof_count_before_real_summary_is_gamed(make_ctx) -> None:
    # A malicious test prints a fake clean count line; the real pytest summary
    # (which follows) contains xfailed — only the last matching line must be used.
    stdout = "0 xfailed, 5 passed in 0.01s\n5 passed, 2 xfailed in 0.12s\n"
    sandbox = FakeSandbox(SandboxResult(0, stdout, ""))
    a = Artifact.from_files({"solution.py": SOLVED})
    v = Pytest(suite="tests/").verify(a, make_ctx(sandbox))
    assert isinstance(v, Fail) and "xfailed" in v.feedback


def test_count_line_after_summary_without_timing_shape_is_ignored(make_ctx) -> None:
    # A malicious test can append text after the real summary (e.g. atexit).
    # Old code took counts from the last count-bearing line; the anchored parse
    # only trusts summary-shaped lines (timing suffix), so the real one wins.
    stdout = "5 passed, 2 xfailed in 0.12s\nnote: 0 xfailed remaining, 5 passed\n"
    sandbox = FakeSandbox(SandboxResult(0, stdout, ""))
    a = Artifact.from_files({"solution.py": SOLVED})
    v = Pytest(suite="tests/").verify(a, make_ctx(sandbox))
    assert isinstance(v, Fail)


def test_exit5_no_tests_collected_is_fail(make_ctx) -> None:
    # pytest exits with code 5 when no tests are collected — treat as failure.
    sandbox = FakeSandbox(SandboxResult(5, "", ""))
    a = Artifact.from_files({"solution.py": SOLVED})
    v = Pytest(suite="tests/").verify(a, make_ctx(sandbox))
    assert isinstance(v, Fail)


def test_exit0_empty_stdout_is_ok(make_ctx) -> None:
    # pytest never exits 0 having printed no summary in practice; exit code 5
    # (no tests collected) is the real no-tests signal.  With exit 0, no counts,
    # and no gaming flags the verifier currently returns Ok — document that.
    sandbox = FakeSandbox(SandboxResult(0, "", ""))
    a = Artifact.from_files({"solution.py": SOLVED})
    v = Pytest(suite="tests/").verify(a, make_ctx(sandbox))
    assert isinstance(v, Ok)
