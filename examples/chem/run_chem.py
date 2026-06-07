"""Optimize a molecule's predicted solubility with a real model (needs crucible-chem:0).

ANTHROPIC_API_KEY=... .venv/bin/python examples/chem/run_chem.py
"""

from pathlib import Path

from crucible import Task, run
from crucible.budgets import RunBudget
from crucible.verifiers import Chem

result = run(
    task=Task.from_path(Path("examples/chem/molecule.smi"), editable=["smiles"]),
    verifier=Chem(target=0.4),  # calibrated target (matches the integration test)
    model="gemini-3.5-flash",
    workers=3,
    episode=None,
    run_budget=RunBudget(episodes_per_worker=20, plateau_patience=5),
    sandbox="docker",
    image="crucible-chem:0",
    db="crucible-chem.db",
)
artifact = result.solution or result.best_partial
print("SOLVED (hit target)" if result.solution else "best molecule found:")
print(artifact.region_text(artifact.region("smiles")).strip())
