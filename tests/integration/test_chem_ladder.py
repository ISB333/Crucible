"""Chem ladder — every rung's golden molecule hits its calibrated target, and the
bare-scaffold start never does (needs crucible-chem:0)."""

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

LADDER = Path(__file__).parent.parent.parent / "examples" / "chem"

REGION_START = "# crucible:region start name=smiles"
REGION_END = "# crucible:region end"

# rung -> (scaffold SMARTS, logS target). Single source: examples/chem/run_chem_ladder.py.
RUNGS: dict[str, tuple[str, float]] = {
    "01_free_solubility": ("", 0.4),
    "02_benzene_polyol": ("c1ccccc1", -0.35),
    "03_pyridine_push": ("c1ccncc1", 0.0),
    "04_sulfonamide": ("NS(=O)(=O)c1ccccc1", 0.1),
    "05_greasy_chain": ("CCCCCCCC", 2.0),
    "06_naphthalene_burden": ("c1ccc2ccccc2c1", -0.9),
    "07_indole_tight": ("c1ccc2c(c1)cc[nH]2", -0.95),
    "08_purine_summit": ("c1ncc2ncnc2n1", 0.1),
}


def region_content(path: Path) -> str:
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if REGION_START in line)
    end = next(i for i, line in enumerate(lines) if REGION_END in line)
    return "\n".join(lines[start + 1 : end])


def wr(smiles: str) -> ToolCall:
    return ToolCall(id="1", name="write_region", args={"name": "smiles", "content": smiles})


async def _run(rung: str, smiles: str, tmp_path: Path):
    scaffold, target = RUNGS[rung]
    root = tmp_path / rung
    shutil.copytree(LADDER / "cases" / rung, root)
    return await search(
        task=Task.from_path(root / "molecule.smi", editable=["smiles"]),
        verifier=Chem(target=target, scaffold=scaffold),
        session_factory=lambda _w, _e: ScriptedSession([[wr(smiles)]]),
        store=Store(tmp_path / "c.db"),
        sandbox_factory=lambda: DockerSandbox(image="crucible-chem:0"),
        model="scripted",
        workers=1,
        episode_budget=EpisodeBudget(edits=5, turns=5),
        run_budget=RunBudget(episodes_per_worker=1),
    )


@pytest.mark.parametrize("rung", sorted(RUNGS), ids=sorted(RUNGS))
async def test_golden_molecule_hits_target(rung: str, tmp_path: Path) -> None:
    golden = region_content(LADDER / "golden" / f"{rung}.smi").strip()
    result = await _run(rung, golden, tmp_path)
    assert result.solution is not None, f"{rung}: golden molecule below target"


@pytest.mark.parametrize("rung", sorted(RUNGS), ids=sorted(RUNGS))
async def test_start_molecule_is_below_target(rung: str, tmp_path: Path) -> None:
    start = region_content(LADDER / "cases" / rung / "molecule.smi").strip()
    result = await _run(rung, start, tmp_path)
    assert result.solution is None, f"{rung}: start molecule already hits target"
    assert result.best_partial is not None  # valid molecule => Scored, ranked as best
