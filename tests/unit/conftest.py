from collections.abc import Sequence
from pathlib import Path

import pytest

from crucible.artifact import Artifact, scan_holes
from crucible.sandbox import SandboxResult
from crucible.task import Task
from crucible.verify import Ok, Partial, RunContext, Verdict


class FakeSandbox:
    """Returns a canned result; records every call for assertions."""

    def __init__(self, result: SandboxResult | None = None) -> None:
        self.result = result or SandboxResult(exit_code=0, stdout="", stderr="")
        self.calls: list[tuple[tuple[str, ...], Path]] = []

    def run(
        self, argv: Sequence[str], *, cwd: Path, timeout_s: float, network: bool = False
    ) -> SandboxResult:
        self.calls.append((tuple(argv), cwd))
        return self.result


class StubVerifier:
    """OK iff no generic holes remain. Counts invocations."""

    def __init__(self, deterministic: bool = True) -> None:
        self.deterministic = deterministic
        self.calls = 0

    def verify(self, artifact: Artifact, ctx: RunContext) -> Verdict:
        self.calls += 1
        holes = scan_holes(artifact)
        if holes:
            return Partial(open_holes=holes, feedback="obligations remain")
        return Ok(produced=artifact)


@pytest.fixture
def make_ctx(tmp_path: Path):
    def _make(sandbox: FakeSandbox | None = None, network: bool = False) -> RunContext:
        task = Task(root=tmp_path, files=(), editable=("solution",), network=network)
        return RunContext(task=task, sandbox=sandbox or FakeSandbox(), scratch=tmp_path / "scratch")

    return _make
