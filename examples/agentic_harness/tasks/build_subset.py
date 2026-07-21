"""One-time: pick 25 BigCodeBench/Hard tasks with light deps, write spec.md + skeleton.py
per task, and produce subset.json with search/heldout split.

Contract:
  - spec.md     <- task["code_prompt"] (imports + function signature — what the agent SEES)
  - skeleton.py <- "pass\n" (placeholder body — what the agent WRITES)
  - check_solution(task_id, sol) prepends code_prompt internally, so skeleton must be
    the body only, NOT the full function.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bcb_wrapper import load_tasks

# Safe libs: stdlib + numpy/pandas/scipy (already available in the eval env)
SAFE_LIBS = {
    "numpy", "pandas", "scipy",
    "itertools", "math", "re", "collections", "datetime", "json", "os", "sys",
    "random", "statistics", "functools", "string", "typing", "hashlib", "base64",
    "io", "copy", "enum", "decimal", "fractions", "textwrap", "bisect", "heapq",
    "operator", "pathlib", "unittest", "csv", "urllib", "html", "xml",
    "configparser", "tempfile", "shutil", "glob", "zipfile", "gzip", "struct",
    "array", "queue", "threading", "multiprocessing", "socket", "sqlite3", "logging",
}


def _parse_libs(libs_str: str) -> list[str]:
    """Parse the libs field (a string repr of a Python list) into a list of lib names."""
    if not libs_str or libs_str == "[]":
        return []
    try:
        return ast.literal_eval(libs_str)
    except (ValueError, SyntaxError):
        # Fallback: comma-split
        return [l.strip().strip("'\"") for l in libs_str.split(",") if l.strip()]


def main() -> None:
    tasks = load_tasks(subset="hard")
    root = Path(__file__).parent

    # Filter to tasks with only safe libs
    safe_ids: list[str] = []
    for tid, t in tasks.items():
        libs = _parse_libs(t["libs"])
        if all(lib.lower() in {s.lower() for s in SAFE_LIBS} for lib in libs):
            safe_ids.append(tid)

    if len(safe_ids) < 25:
        raise RuntimeError(f"Only {len(safe_ids)} safe tasks, need 25")

    # Take first 25 safe tasks (deterministic, BigCodeBench ordering)
    selected = safe_ids[:25]

    subset: list[dict] = []
    for i, tid in enumerate(selected):
        t = tasks[tid]
        split = "search" if i < 10 else "heldout"

        d = root / tid
        d.mkdir(parents=True, exist_ok=True)

        # spec.md = the code_prompt (imports + signature + docstring — what agent sees)
        (d / "spec.md").write_text(t["code_prompt"])

        # skeleton.py = placeholder body only (check_solution prepends code_prompt)
        (d / "skeleton.py").write_text("pass\n")

        subset.append({"task_id": tid, "split": split})

    (root / "subset.json").write_text(json.dumps(subset, indent=2) + "\n")
    print(f"Wrote {len(subset)} tasks to {root / 'subset.json'}")
    for entry in subset:
        print(f"  {entry['split']:8s} {entry['task_id']}")


if __name__ == "__main__":
    main()