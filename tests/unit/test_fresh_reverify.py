from pathlib import Path

import pytest

from crucible.artifact import Artifact
from crucible.integrity import fresh_reverify
from crucible.task import Task
from crucible.verify import Fail, Ok, RunContext
from tests.unit.conftest import FakeSandbox

pytestmark = pytest.mark.unit

CONTENT = "# crucible:region start name=s\nx = 1\n# crucible:region end\n"


def make() -> tuple[Artifact, Task]:
    a = Artifact.from_files({"f.py": CONTENT})
    return a, Task(root=Path("/t"), files=("f.py",), editable=("s",))


class RecordingVerifier:
    deterministic = True

    def __init__(self, verdicts: list) -> None:
        self._verdicts = verdicts
        self.contexts: list[RunContext] = []

    def verify(self, artifact: Artifact, ctx: RunContext):
        self.contexts.append(ctx)
        return self._verdicts.pop(0)


def test_reproducing_pass_is_accepted() -> None:
    a, task = make()
    v = RecordingVerifier([Ok(produced=a)])
    assert fresh_reverify(a, v, task, sandbox_factory=FakeSandbox)


def test_non_reproducing_pass_is_rejected() -> None:
    a, task = make()
    v = RecordingVerifier([Fail(feedback="flaky — does not reproduce in isolation")])
    assert not fresh_reverify(a, v, task, sandbox_factory=FakeSandbox)


def test_reverify_uses_fresh_scratch_and_fresh_sandbox() -> None:
    a, task = make()
    v = RecordingVerifier([Ok(produced=a)])
    sandboxes: list[FakeSandbox] = []

    def factory() -> FakeSandbox:
        sb = FakeSandbox()
        sandboxes.append(sb)
        return sb

    fresh_reverify(a, v, task, sandbox_factory=factory)
    ctx = v.contexts[0]
    assert ctx.sandbox is sandboxes[0]  # freshly constructed, not reused
    assert "crucible-reverify-" in str(ctx.scratch)
