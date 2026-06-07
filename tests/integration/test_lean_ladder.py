"""Lean ladder — each rung is solvable end-to-end, sorry-free (needs crucible-lean:0)."""

import shutil
from pathlib import Path

import pytest

from crucible.budgets import EpisodeBudget, RunBudget
from crucible.llm import ScriptedSession, ToolCall
from crucible.orchestrator import search
from crucible.sandbox import DockerSandbox
from crucible.store import Store
from crucible.task import Task
from crucible.verifiers import Lean

pytestmark = pytest.mark.integration

LADDER = Path(__file__).parent.parent.parent / "examples" / "lean_ladder"
CASES = LADDER / "cases"
GOLDEN_DIR = LADDER / "golden"

REGION_START = "-- crucible:region start name=proof"
REGION_END = "-- crucible:region end"

# Golden proofs — the editable `proof` region body, extracted from the full
# compile-verified solutions in examples/lean_ladder/golden/<rung>.lean.
RUNGS = sorted(p.stem for p in GOLDEN_DIR.glob("*.lean"))


def golden_proof(rung: str) -> str:
    lines = (GOLDEN_DIR / f"{rung}.lean").read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if REGION_START in line)
    end = next(i for i, line in enumerate(lines) if REGION_END in line)
    return "\n".join(lines[start + 1 : end])


def prove(content: str) -> ToolCall:
    return ToolCall(id="1", name="write_region", args={"name": "proof", "content": content})


@pytest.mark.parametrize("rung", RUNGS, ids=RUNGS)
async def test_rung_is_solved_sorry_free(rung: str, tmp_path: Path) -> None:
    root = tmp_path / rung
    shutil.copytree(CASES / rung, root)
    store = Store(tmp_path / "c.db")
    result = await search(
        task=Task.from_path(root, editable=["proof"]),
        verifier=Lean(),
        session_factory=lambda _w, _e: ScriptedSession([[prove(golden_proof(rung))]]),
        store=store,
        sandbox_factory=lambda: DockerSandbox(image="crucible-lean:0"),
        model="scripted",
        workers=1,
        episode_budget=EpisodeBudget(edits=5, turns=5),
        run_budget=RunBudget(episodes_per_worker=1),
    )
    assert result.solution is not None, f"{rung} not solved"
    assert "sorry" not in result.solution.files["thm.lean"]
