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
    deterministic = True
    verifier_id = "stub-scored"

    def verify(self, artifact: Artifact, ctx: RunContext) -> Verdict:
        body = artifact.region_text(artifact.region("smiles")).strip()
        return Scored(produced=artifact, value=float(body or "0"))


def wr(value: str) -> ToolCall:
    return ToolCall(id="1", name="write_region", args={"name": "smiles", "content": value})


async def test_plateau_stops_early(tmp_path: Path) -> None:
    (tmp_path / "m.smi").write_text(SEED)
    # best is reached at episode 0 (value 5.0); the next two do not improve.
    scripts = [[wr("5.0")], [wr("1.0")], [wr("2.0")], [wr("3.0")], [wr("4.0")]]

    def factory(worker: int, episode: int) -> ScriptedSession:
        return ScriptedSession([scripts[episode]])

    store = Store(tmp_path / "c.db")
    result = await search(
        task=Task.from_path(tmp_path / "m.smi", editable=["smiles"]),
        verifier=StubScored(),
        session_factory=factory,
        store=store,
        sandbox_factory=SubprocessSandbox,
        model="scripted",
        workers=1,
        episode_budget=EpisodeBudget(edits=5, turns=5),
        run_budget=RunBudget(episodes_per_worker=5, plateau_patience=2),
    )
    # episode 0 sets best=5.0; episodes 1,2 do not improve -> patience 2 exhausted -> stop.
    n_episodes = store._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    assert n_episodes == 3  # 0 (improve) + 1,2 (no improve) then break
    assert result.best_partial.region_text(result.best_partial.region("smiles")).strip() == "5.0"
