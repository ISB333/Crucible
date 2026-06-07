import json

import pytest

from crucible.artifact import Artifact
from crucible.sandbox import SandboxResult
from crucible.verifiers.chem_verifier import Chem
from crucible.verify import Fail, Ok, Partial, Scored
from tests.unit.conftest import FakeSandbox

pytestmark = pytest.mark.unit

MOL = "# crucible:region start name=smiles\nCO\n# crucible:region end\n"
HOLEY = "# crucible:region start name=smiles\n# crucible:hole\n# crucible:region end\n"


def out(valid: bool, score: float, has_scaffold: bool = True, error: str = "") -> str:
    return json.dumps(
        {"valid": valid, "score": score, "has_scaffold": has_scaffold, "error": error}
    )


def test_score_at_or_above_target_is_ok(make_ctx) -> None:
    sandbox = FakeSandbox(SandboxResult(0, out(True, 1.0), ""))
    a = Artifact.from_files({"molecule.smi": MOL})
    assert isinstance(Chem(target=0.5).verify(a, make_ctx(sandbox)), Ok)


def test_valid_below_target_is_scored(make_ctx) -> None:
    sandbox = FakeSandbox(SandboxResult(0, out(True, -1.0), ""))
    a = Artifact.from_files({"molecule.smi": MOL})
    v = Chem(target=0.5).verify(a, make_ctx(sandbox))
    assert isinstance(v, Scored) and v.value == pytest.approx(-1.0)


def test_invalid_molecule_is_fail(make_ctx) -> None:
    sandbox = FakeSandbox(SandboxResult(0, out(False, 0.0, error="invalid SMILES: 'xqz'"), ""))
    a = Artifact.from_files({"molecule.smi": MOL})
    v = Chem(target=0.5).verify(a, make_ctx(sandbox))
    assert isinstance(v, Fail) and "invalid" in v.feedback.lower()


def test_missing_scaffold_is_fail(make_ctx) -> None:
    sandbox = FakeSandbox(SandboxResult(0, out(True, 9.0, has_scaffold=False), ""))
    a = Artifact.from_files({"molecule.smi": MOL})
    v = Chem(target=0.5, scaffold="c1ccccc1").verify(a, make_ctx(sandbox))
    assert isinstance(v, Fail) and "scaffold" in v.feedback.lower()


def test_holes_make_it_partial(make_ctx) -> None:
    sandbox = FakeSandbox(SandboxResult(0, out(True, 9.0), ""))
    a = Artifact.from_files({"molecule.smi": HOLEY})
    assert isinstance(Chem(target=0.5).verify(a, make_ctx(sandbox)), Partial)


def test_unparsable_scorer_output_is_fail(make_ctx) -> None:
    sandbox = FakeSandbox(SandboxResult(0, "segfault", ""))
    a = Artifact.from_files({"molecule.smi": MOL})
    assert isinstance(Chem(target=0.5).verify(a, make_ctx(sandbox)), Fail)


def test_timeout_is_fail(make_ctx) -> None:
    sandbox = FakeSandbox(SandboxResult(124, "", "", timed_out=True))
    a = Artifact.from_files({"molecule.smi": MOL})
    v = Chem(target=0.5).verify(a, make_ctx(sandbox))
    assert isinstance(v, Fail) and "timeout" in v.feedback


def test_verifier_id_reflects_target() -> None:
    assert Chem(target=0.5).verifier_id == "chem:0.5"
