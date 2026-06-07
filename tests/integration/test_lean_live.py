from pathlib import Path

import pytest

from crucible.artifact import Artifact
from crucible.sandbox import DockerSandbox
from crucible.task import Task
from crucible.verifiers.lean import Lean
from crucible.verify import Ok, Partial, RunContext

pytestmark = pytest.mark.integration  # needs crucible-lean:0 built (Task 4)

PROVED = """\
theorem add_comm_toy (a b : Nat) : a + b = b + a := by
  -- crucible:region start name=proof
  exact Nat.add_comm a b
  -- crucible:region end
"""


def make_ctx(tmp_path: Path) -> RunContext:
    task = Task(root=tmp_path, files=("toy.lean",), editable=("proof",))
    return RunContext(
        task=task, sandbox=DockerSandbox(image="crucible-lean:0"), scratch=tmp_path / "s"
    )


def test_real_lean_accepts_valid_proof(tmp_path: Path) -> None:
    a = Artifact.from_files({"toy.lean": PROVED})
    assert isinstance(Lean().verify(a, make_ctx(tmp_path)), Ok)


def test_real_lean_reports_sorry_as_partial(tmp_path: Path) -> None:
    a = Artifact.from_files({"toy.lean": PROVED.replace("exact Nat.add_comm a b", "sorry")})
    v = Lean().verify(a, make_ctx(tmp_path))
    assert isinstance(v, Partial) and v.open_holes[0].kind == "sorry"
