"""Optimize a molecule's predicted solubility with a real model (needs crucible-chem:0).

ANTHROPIC_API_KEY=... .venv/bin/python examples/chem/run_chem.py
GOOGLE_API_KEY=... .venv/bin/python examples/chem/run_chem.py --model gemini-flash

With LLM shepherding (optional):
  ANTHROPIC_API_KEY=... .venv/bin/python examples/chem/run_chem.py --advisor claude-opus-4-8
"""

from pathlib import Path
import argparse

from crucible import AdvisorPolicy, Task, run
from crucible.budgets import RunBudget
from crucible.verifiers import Chem

# Get the directory of the current script
SCRIPT_DIR = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="claude-haiku-4-5")
parser.add_argument("--advisor", default=None, help="advisor model for LLM shepherding (e.g. claude-opus-4-8)")
parser.add_argument("--advisor-max-calls", type=int, default=None, help="max advisor calls per run")
parser.add_argument("--advisor-fail-streak", type=int, default=3, help="trigger advisor after N non-improving episodes")
args = parser.parse_args()

# Build advisor policy if --advisor is specified
advisor = None
if args.advisor is not None:
    advisor = AdvisorPolicy(
        model=args.advisor,
        max_calls_per_run=args.advisor_max_calls,
        fail_streak=args.advisor_fail_streak,
    )

result = run(
    task=Task.from_path(SCRIPT_DIR / "molecule.smi", editable=["smiles"]),
    verifier=Chem(target=0.4),  # calibrated target (matches the integration test)
    model=args.model,
    workers=3,
    episode=None,
    run_budget=RunBudget(episodes_per_worker=20, plateau_patience=5),
    sandbox="docker",
    image="crucible-chem:0",
    db="crucible-chem.db",
    advisor=advisor,
)
artifact = result.solution or result.best_partial
print("SOLVED (hit target)" if result.solution else "best molecule found:")
print(artifact.region_text(artifact.region("smiles")).strip())
