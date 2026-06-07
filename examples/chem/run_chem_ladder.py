"""Run the molecular-optimization ladder and report solved rungs (needs crucible-chem:0).

    GOOGLE_API_KEY=... .venv/bin/python examples/chem/run_chem_ladder.py
    # or, no key: falls back to the golden molecules to smoke the pipeline
    .venv/bin/python examples/chem/run_chem_ladder.py --scripted

Each rung locks a scaffold (SMARTS) that fights solubility — aromatic atoms,
hydrophobic chains — and demands a logS target only reachable with deliberate
polar decoration. Targets are calibrated against the deterministic surrogate
in crucible-chem:0; golden solutions in golden/<rung>.smi are score-verified
(see the per-rung naive-attempt scores recorded there).
"""

import os
import sys
from collections.abc import Callable
from pathlib import Path

from crucible import Task, run
from crucible.budgets import RunBudget, budgets
from crucible.llm import LLMSession, ScriptedSession, ToolCall
from crucible.verifiers import Chem

CASES = Path(__file__).parent / "cases"
GOLDEN = Path(__file__).parent / "golden"

REGION_START = "# crucible:region start name=smiles"
REGION_END = "# crucible:region end"

# rung -> (scaffold SMARTS, logS target). Calibrated 2026-06-07 vs crucible-chem:0;
# the bare scaffold start always scores well below target (e.g. naphthalene -3.16
# vs -0.9) and the golden clears it with a tight margin (0.05-0.47).
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


def golden_molecule(rung: str) -> str:
    """SMILES-region content from the score-verified golden solution."""
    lines = (GOLDEN / f"{rung}.smi").read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if REGION_START in line)
    end = next(i for i, line in enumerate(lines) if REGION_END in line)
    return "\n".join(lines[start + 1 : end])


def scripted_factory(content: str) -> Callable[[int, int], LLMSession]:
    def factory(_worker: int, _episode: int) -> LLMSession:
        call = ToolCall(id="1", name="write_region", args={"name": "smiles", "content": content})
        return ScriptedSession([[call]])

    return factory


def main(argv: list[str]) -> int:
    has_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    scripted = "--scripted" in argv or not has_key
    solved: list[tuple[str, bool]] = []
    for rung, (scaffold, target) in sorted(RUNGS.items()):
        task = Task.from_path(CASES / rung / "molecule.smi", editable=["smiles"])
        if scripted:
            result = run(
                task=task,
                verifier=Chem(target=target, scaffold=scaffold),
                model="scripted",  # required by run(); session_factory plays the golden molecule
                workers=1,
                episode=budgets(edits=5, turns=5),
                run_budget=budgets(episodes=1),
                sandbox="docker",
                image="crucible-chem:0",
                db=f"chem-ladder-{rung}.db",
                session_factory=scripted_factory(golden_molecule(rung)),
            )
        else:
            result = run(
                task=task,
                verifier=Chem(target=target, scaffold=scaffold),
                model="gemini-3.5-flash",
                workers=3,
                episode=None,
                run_budget=RunBudget(episodes_per_worker=20, plateau_patience=5),
                sandbox="docker",
                image="crucible-chem:0",
                db=f"chem-ladder-{rung}.db",
            )
        ok = result.solution is not None
        solved.append((rung, ok))
        best = result.solution or result.best_partial
        mol = best.region_text(best.region("smiles")).strip() if best else "<none>"
        print(f"{rung}: {'SOLVED' if ok else 'unsolved'} ({mol})")
    return 0 if all(ok for _, ok in solved) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
