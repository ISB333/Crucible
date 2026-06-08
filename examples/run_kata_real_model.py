"""Manual §13 demonstration with a real model. Needs ANTHROPIC_API_KEY and crucible-py:0.

Run: .venv/bin/python examples/run_kata_real_model.py --model gemini-3.5-flash
"""

from pathlib import Path
import argparse

from crucible import Task, budgets, run
from crucible.verifiers import Pytest

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="claude-sonnet-4-6")
args = parser.parse_args()

result = run(
    task=Task.from_path(Path("tests/fixtures/kata"), editable=["solution"]),
    verifier=Pytest(suite="tests/"),
    model=args.model,
    workers=3,
    episode=budgets(edits=20, turns=15),
    run_budget=budgets(wall_clock="10m", usd=2.0),
    db="crucible-example.db",
)
print("SOLVED" if result.solution else "UNSOLVED — best partial:")
artifact = result.solution or result.best_partial
print(artifact.files["problem.py"])
