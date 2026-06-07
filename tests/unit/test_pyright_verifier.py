import json

import pytest

from crucible.artifact import Artifact
from crucible.sandbox import SandboxResult
from crucible.verifiers.pyright_verifier import Pyright
from crucible.verify import Fail, Ok, Partial
from tests.unit.conftest import FakeSandbox

pytestmark = pytest.mark.unit

SOLVED = "# crucible:region start name=s\nx: int = 1\n# crucible:region end\n"
HOLEY = "# crucible:region start name=s\nraise NotImplementedError\n# crucible:region end\n"


def pyright_json(error_count: int, diagnostics: list | None = None) -> str:
    return json.dumps(
        {
            "generalDiagnostics": diagnostics or [],
            "summary": {"errorCount": error_count, "warningCount": 0},
        }
    )


def test_zero_errors_is_ok(make_ctx) -> None:
    sandbox = FakeSandbox(SandboxResult(0, pyright_json(0), ""))
    a = Artifact.from_files({"f.py": SOLVED})
    assert isinstance(Pyright().verify(a, make_ctx(sandbox)), Ok)


def test_errors_are_fail_with_locus(make_ctx) -> None:
    diag = [
        {
            "file": "/ws/f.py",
            "severity": "error",
            "message": 'Operator "+" not supported',
            "range": {"start": {"line": 1, "character": 0}, "end": {"line": 1, "character": 5}},
        }
    ]
    sandbox = FakeSandbox(SandboxResult(1, pyright_json(1, diag), ""))
    a = Artifact.from_files({"f.py": SOLVED})
    v = Pyright().verify(a, make_ctx(sandbox))
    assert isinstance(v, Fail)
    assert "not supported" in v.feedback
    assert v.locus is not None and v.locus.line_start == 1


def test_holes_make_it_partial(make_ctx) -> None:
    sandbox = FakeSandbox(SandboxResult(0, pyright_json(0), ""))
    a = Artifact.from_files({"f.py": HOLEY})
    assert isinstance(Pyright().verify(a, make_ctx(sandbox)), Partial)


def test_unparsable_output_is_fail_not_crash(make_ctx) -> None:
    sandbox = FakeSandbox(SandboxResult(2, "node panic", ""))
    a = Artifact.from_files({"f.py": SOLVED})
    v = Pyright().verify(a, make_ctx(sandbox))
    assert isinstance(v, Fail)


def test_verifier_id_reflects_strictness() -> None:
    assert Pyright(strict=True).verifier_id == "pyright:strict"
    assert Pyright(strict=False).verifier_id == "pyright:standard"


def test_summary_zero_but_error_diagnostic_is_fail(make_ctx) -> None:
    """Summary errorCount=0 but a real error-severity diagnostic → must be Fail.

    This divergence case would produce Ok under the old summary-gated implementation.
    """
    diag = [
        {
            "file": "/ws/f.py",
            "severity": "error",
            "message": "Divergence: summary lied",
            "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}},
        }
    ]
    sandbox = FakeSandbox(SandboxResult(0, pyright_json(0, diag), ""))
    a = Artifact.from_files({"f.py": SOLVED})
    v = Pyright().verify(a, make_ctx(sandbox))
    assert isinstance(v, Fail), (
        "Expected Fail when diagnostics list has errors despite errorCount=0"
    )
    assert "Divergence: summary lied" in v.feedback


def test_locus_line_is_stored_zero_based_displayed_one_based(make_ctx) -> None:
    """Diagnostic with line=2 → locus.line_start==2 (0-based), feedback contains 'f.py:3:'."""
    diag = [
        {
            "file": "/ws/f.py",
            "severity": "error",
            "message": "Bad type",
            "range": {"start": {"line": 2, "character": 0}, "end": {"line": 2, "character": 4}},
        }
    ]
    sandbox = FakeSandbox(SandboxResult(1, pyright_json(1, diag), ""))
    a = Artifact.from_files({"f.py": SOLVED})
    v = Pyright().verify(a, make_ctx(sandbox))
    assert isinstance(v, Fail)
    assert v.locus is not None
    assert v.locus.line_start == 2
    assert "f.py:3:" in v.feedback


def test_timeout_is_fail(make_ctx) -> None:
    sandbox = FakeSandbox(SandboxResult(124, "", "", timed_out=True))
    a = Artifact.from_files({"f.py": SOLVED})
    v = Pyright().verify(a, make_ctx(sandbox))
    assert isinstance(v, Fail)
    assert "timeout" in v.feedback
