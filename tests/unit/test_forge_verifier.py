import pytest

from crucible.artifact import Artifact
from crucible.sandbox import SandboxResult
from crucible.verifiers.forge_verifier import Forge, forge_counts
from crucible.verify import Fail, Ok, Partial
from tests.unit.conftest import FakeSandbox

pytestmark = pytest.mark.unit

# A minimal attacker artifact: the editable region is filled (no hole).
SOLVED = (
    "// crucible:region start name=attack\n"
    "vault.deposit{value: 1 ether}();\n"
    "// crucible:region end\n"
)
HOLEY = "// crucible:region start name=attack\n// crucible:hole\n// crucible:region end\n"

PASS_OUT = "Ran 1 test suite in 1.2s: 1 tests passed, 0 failed, 0 skipped (1 total tests)\n"
FAIL_OUT = (
    "[FAIL: vault not drained] testDrainsVault() (gas: 1)\n"
    "Ran 1 test suite in 1.2s: 0 tests passed, 1 failed, 0 skipped (1 total tests)\n"
)
ZERO_OUT = (
    "No tests to run\nRan 0 test suites: 0 tests passed, 0 failed, 0 skipped (0 total tests)\n"
)


def test_forge_counts_parses_summary() -> None:
    assert forge_counts(PASS_OUT) == {"passed": 1, "failed": 0, "total": 1}


def test_passing_exploit_is_ok(make_ctx) -> None:
    sandbox = FakeSandbox(SandboxResult(0, PASS_OUT, ""))
    a = Artifact.from_files({"Attack.sol": SOLVED})
    assert isinstance(Forge().verify(a, make_ctx(sandbox)), Ok)


def test_failing_exploit_is_fail_with_output(make_ctx) -> None:
    sandbox = FakeSandbox(SandboxResult(1, FAIL_OUT, ""))
    a = Artifact.from_files({"Attack.sol": SOLVED})
    v = Forge().verify(a, make_ctx(sandbox))
    assert isinstance(v, Fail) and "not drained" in v.feedback


def test_zero_tests_is_fail(make_ctx) -> None:
    sandbox = FakeSandbox(SandboxResult(0, ZERO_OUT, ""))
    a = Artifact.from_files({"Attack.sol": SOLVED})
    v = Forge().verify(a, make_ctx(sandbox))
    assert isinstance(v, Fail) and "no tests" in v.feedback.lower()


def test_holes_make_it_partial(make_ctx) -> None:
    sandbox = FakeSandbox(SandboxResult(1, FAIL_OUT, ""))
    a = Artifact.from_files({"Attack.sol": HOLEY})
    assert isinstance(Forge().verify(a, make_ctx(sandbox)), Partial)


def test_timeout_is_fail(make_ctx) -> None:
    sandbox = FakeSandbox(SandboxResult(124, "", "", timed_out=True))
    a = Artifact.from_files({"Attack.sol": SOLVED})
    v = Forge().verify(a, make_ctx(sandbox))
    assert isinstance(v, Fail) and "timeout" in v.feedback
