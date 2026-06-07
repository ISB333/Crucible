"""Molecular optimization end-to-end with the real RDKit surrogate (needs crucible-chem:0)."""

import shutil
from pathlib import Path

import pytest

from crucible.budgets import EpisodeBudget, RunBudget
from crucible.llm import ScriptedSession, ToolCall
from crucible.orchestrator import search
from crucible.sandbox import DockerSandbox
from crucible.store import Store
from crucible.task import Task
from crucible.verifiers import Chem

pytestmark = pytest.mark.integration

EXAMPLE = Path(__file__).parent.parent.parent / "examples" / "chem"
TARGET = 0.4  # calibrated: between the CO (0.208) and O (0.5679) logS scores


def wr(smiles: str) -> ToolCall:
    return ToolCall(id="1", name="write_region", args={"name": "smiles", "content": smiles})


def _task(tmp_path: Path) -> Task:
    root = tmp_path / "chem"
    shutil.copytree(EXAMPLE, root)
    return Task.from_path(root / "molecule.smi", editable=["smiles"])


async def test_above_target_molecule_is_accepted(tmp_path: Path) -> None:
    result = await search(
        task=_task(tmp_path),
        verifier=Chem(target=TARGET),
        session_factory=lambda _w, _e: ScriptedSession([[wr("O")]]),  # water: very soluble
        store=Store(tmp_path / "c.db"),
        sandbox_factory=lambda: DockerSandbox(image="crucible-chem:0"),
        model="scripted",
        workers=1,
        episode_budget=EpisodeBudget(edits=5, turns=5),
        run_budget=RunBudget(episodes_per_worker=1),
    )
    assert result.solution is not None
    assert result.solution.region_text(result.solution.region("smiles")).strip() == "O"


async def test_below_target_run_returns_best_scoring(tmp_path: Path) -> None:
    # All below TARGET by score: octane (-2.34) < propanol (-0.39) < ethanol (-0.12).
    # best_partial must be ethanol (CCO).
    mols = ["CCCCCCCC", "CCO", "CCCO"]

    def factory(_worker: int, episode: int) -> ScriptedSession:
        return ScriptedSession([[wr(mols[episode])]])

    result = await search(
        task=_task(tmp_path),
        verifier=Chem(target=99.0),  # unreachable: force the optimization path
        session_factory=factory,
        store=Store(tmp_path / "c.db"),
        sandbox_factory=lambda: DockerSandbox(image="crucible-chem:0"),
        model="scripted",
        workers=1,
        episode_budget=EpisodeBudget(edits=5, turns=5),
        run_budget=RunBudget(episodes_per_worker=3),
    )
    assert result.solution is None
    best = result.best_partial.region_text(result.best_partial.region("smiles")).strip()
    assert best == "CCO"  # ethanol (-0.1247) is the most soluble of octane/ethanol/propanol
