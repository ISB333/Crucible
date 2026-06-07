"""The engine returns the best-scoring artifact when no Ok is reached (v0.5 substrate)."""

from pathlib import Path

import pytest

from crucible.artifact import Artifact
from crucible.budgets import EpisodeBudget, RunBudget
from crucible.llm import ScriptedSession, ToolCall
from crucible.orchestrator import search
from crucible.sandbox import SubprocessSandbox
from crucible.store import Store
from crucible.task import Task
from crucible.verify import RunContext, Scored, Verdict

pytestmark = pytest.mark.unit

SEED = "# crucible:region start name=smiles\n0.0\n# crucible:region end\n"


class StubScored:
    """Reads a float from the editable region and returns it as the score. Never Ok
    (no target), so the run exhausts its budget and returns the best-scoring artifact."""

    deterministic = True
    verifier_id = "stub-scored"

    def verify(self, artifact: Artifact, ctx: RunContext) -> Verdict:
        body = artifact.region_text(artifact.region("smiles")).strip()
        return Scored(produced=artifact, value=float(body or "0"))


def wr(value: str) -> ToolCall:
    return ToolCall(id="1", name="write_region", args={"name": "smiles", "content": value})


async def test_search_returns_highest_scoring_artifact(tmp_path: Path) -> None:
    (tmp_path / "m.smi").write_text(SEED)
    # Worker 0 ends at 3.0; worker 1 ends at 9.0. Both have 0 holes, so the old
    # min-by-holes code picks worker 0 (first in sequence) and returns "3.0".
    # The rank-based selection (max by best_rank) must return worker 1's "9.0".
    scripts = {
        0: [[wr("3.0")]],  # one episode, final score 3.0
        1: [[wr("9.0")]],  # one episode, final score 9.0
    }

    def factory(worker: int, episode: int) -> ScriptedSession:
        return ScriptedSession([scripts[worker][episode]])

    result = await search(
        task=Task.from_path(tmp_path / "m.smi", editable=["smiles"]),
        verifier=StubScored(),
        session_factory=factory,
        store=Store(tmp_path / "c.db"),
        sandbox_factory=SubprocessSandbox,
        model="scripted",
        workers=2,
        episode_budget=EpisodeBudget(edits=5, turns=5),
        run_budget=RunBudget(episodes_per_worker=1),
    )
    assert result.solution is None  # Scored never accepts on its own
    assert result.best_partial.region_text(result.best_partial.region("smiles")).strip() == "9.0"
