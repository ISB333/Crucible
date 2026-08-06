"""One-time: curate a 25-task EASIER BigCodeBench subset (non-Hard) for the
second benchmark. The Hard 148 are at Tess-9B's ceiling (5/6, no headroom); the
other ~990 should have room to grow, which is what tests whether harness edits
can show a REAL generalizing lift (vs just fixing mechanics at the ceiling).

Run once, commit the output:
    uv run python examples/agentic_harness/build_subset_easy.py

Picks 25 non-Hard tasks (seeded) split 10 search + 15 heldout. spec.md =
code_prompt (signature+docstring); skeleton.py = "pass" (body placeholder, the
agent writes the body; check_solution_graded prepends code_prompt internally).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from bcb_wrapper import load_tasks  # type: ignore[import-not-found]


def main() -> None:
    full = load_tasks(subset="full")
    hard = load_tasks(subset="hard")
    easy_ids = [tid for tid in full if tid not in set(hard)]
    random.seed(42)
    pick = random.sample(easy_ids, 25)
    root = Path(__file__).parent / "tasks_easy"
    root.mkdir(parents=True, exist_ok=True)
    subset = []
    for i, tid in enumerate(pick):
        t = full[tid]
        d = root / tid
        d.mkdir(parents=True, exist_ok=True)
        (d / "spec.md").write_text(t["code_prompt"])
        (d / "skeleton.py").write_text("pass\n")
        subset.append({"task_id": tid, "split": "search" if i < 10 else "heldout"})
    (root / "subset.json").write_text(json.dumps(subset, indent=2) + "\n")
    print(f"wrote {len(subset)} non-hard tasks to {root} (10 search + 15 heldout)")


if __name__ == "__main__":
    main()