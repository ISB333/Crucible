import pytest

from crucible.artifact import Artifact
from crucible.sandbox import SandboxResult
from crucible.verifiers.lean import Lean
from crucible.verify import Fail, Ok, Partial
from tests.unit.conftest import FakeSandbox

pytestmark = pytest.mark.unit

PROVED = """\
theorem add_comm_toy (a b : Nat) : a + b = b + a := by
  -- crucible:region start name=proof
  exact Nat.add_comm a b
  -- crucible:region end
"""

SORRIED = """\
theorem add_comm_toy (a b : Nat) : a + b = b + a := by
  -- crucible:region start name=proof
  sorry
  -- crucible:region end
"""


def test_clean_compile_is_ok(make_ctx) -> None:
    sandbox = FakeSandbox(SandboxResult(0, "", ""))
    a = Artifact.from_files({"toy.lean": PROVED})
    assert isinstance(Lean().verify(a, make_ctx(sandbox)), Ok)


def test_sorry_is_partial_with_sorry_hole(make_ctx) -> None:
    out = "toy.lean:3:2: warning: declaration uses 'sorry'\n"
    sandbox = FakeSandbox(SandboxResult(0, out, ""))
    a = Artifact.from_files({"toy.lean": SORRIED})
    v = Lean().verify(a, make_ctx(sandbox))
    assert isinstance(v, Partial)
    assert v.open_holes[0].kind == "sorry"
    assert v.open_holes[0].line == 2  # 0-based line of the sorry


def test_compile_error_is_fail_with_message(make_ctx) -> None:
    out = "toy.lean:3:8: error: unknown identifier 'Nat.add_com'\n"
    sandbox = FakeSandbox(SandboxResult(1, "", out))
    a = Artifact.from_files({"toy.lean": PROVED})
    v = Lean().verify(a, make_ctx(sandbox))
    assert isinstance(v, Fail) and "unknown identifier" in v.feedback


def test_no_lean_files_is_fail(make_ctx) -> None:
    src = "# crucible:region start name=s\nx=1\n# crucible:region end\n"
    a = Artifact.from_files({"f.py": src})
    assert isinstance(Lean().verify(a, make_ctx()), Fail)


def test_sorry_backtick_quote_style_is_detected(make_ctx) -> None:
    # Real Lean 4.30 output: backtick-quoted `sorry` (observed from crucible-lean:0)
    out = "toy.lean:1:8: warning: declaration uses `sorry`\n"
    sandbox = FakeSandbox(SandboxResult(0, out, ""))
    a = Artifact.from_files({"toy.lean": SORRIED})
    v = Lean().verify(a, make_ctx(sandbox))
    assert isinstance(v, Partial)
    assert v.open_holes[0].kind == "sorry"


def test_timeout_is_fail(make_ctx) -> None:
    sandbox = FakeSandbox(SandboxResult(124, "", "", timed_out=True))
    a = Artifact.from_files({"toy.lean": PROVED})
    v = Lean().verify(a, make_ctx(sandbox))
    assert isinstance(v, Fail) and "timeout" in v.feedback
