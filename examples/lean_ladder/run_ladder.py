"""Run the Lean ladder with a real model and report solved rungs.

    ANTHROPIC_API_KEY=... .venv/bin/python examples/lean_ladder/run_ladder.py
    # or, no key: falls back to the golden scripted proofs to smoke the pipeline
    .venv/bin/python examples/lean_ladder/run_ladder.py --scripted

Rungs 01-06 are warm-ups (one-line proofs). Rungs 07-14 are deep: custom
definitions with no applicable library lemmas, requiring multi-lemma
induction proofs built inside the editable region — they get larger budgets.

Golden solutions live in golden/<rung>.lean (full files, compile-verified
against crucible-lean:0); scripted mode extracts the proof-region content
from them.

Needs crucible-lean:0.
"""

import os
import sys
from collections.abc import Callable
from pathlib import Path

from crucible import Task, budgets, run
from crucible.llm import LLMSession, ScriptedSession, ToolCall
from crucible.verifiers import Lean

CASES = Path(__file__).parent / "cases"
GOLDEN = Path(__file__).parent / "golden"

REGION_START = "-- crucible:region start name=proof"
REGION_END = "-- crucible:region end"

# Rungs >= DEEP_RUNG carry multi-lemma induction proofs; give the model room.
DEEP_RUNG = 7


def golden_proof(rung: str) -> str:
    """Proof-region content from the compile-verified golden solution."""
    lines = (GOLDEN / f"{rung}.lean").read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if REGION_START in line)
    end = next(i for i, line in enumerate(lines) if REGION_END in line)
    return "\n".join(lines[start + 1 : end])


def scripted_factory(content: str) -> Callable[[int, int], LLMSession]:
    def factory(_worker: int, _episode: int) -> LLMSession:
        call = ToolCall(id="1", name="write_region", args={"name": "proof", "content": content})
        return ScriptedSession([[call]])

    return factory


def main(argv: list[str]) -> int:
    scripted = "--scripted" in argv or not os.environ.get("ANTHROPIC_API_KEY")
    rungs = sorted(p.name for p in CASES.iterdir() if p.is_dir())
    solved: list[tuple[str, bool]] = []
    for rung in rungs:
        task = Task.from_path(CASES / rung, editable=["proof"])
        deep = int(rung.split("_")[0]) >= DEEP_RUNG
        if scripted:
            result = run(
                task=task,
                verifier=Lean(),
                model="scripted",  # required by run(); session_factory plays the golden proof
                workers=1,
                episode=budgets(edits=10, turns=10),
                run_budget=budgets(episodes=1),
                sandbox="docker",
                image="crucible-lean:0",
                db=f"lean-ladder-{rung}.db",
                session_factory=scripted_factory(golden_proof(rung)),
            )
        else:
            result = run(
                task=task,
                verifier=Lean(),
                model="claude-sonnet-4-6",
                workers=1,
                episode=budgets(edits=30, turns=40) if deep else budgets(edits=10, turns=10),
                run_budget=budgets(wall_clock="20m", usd=4.0)
                if deep
                else budgets(wall_clock="5m", usd=1.0),
                sandbox="docker",
                image="crucible-lean:0",
                db=f"lean-ladder-{rung}.db",
            )
        ok = result.solution is not None
        solved.append((rung, ok))
        print(f"{rung}: {'SOLVED' if ok else 'unsolved'}")
    return 0 if all(ok for _, ok in solved) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
