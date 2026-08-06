"""Wave 0: measure Tess + the minimal harness on the 10-task SEARCH subset.

We measure on the search subset only (not the full 25) because the verifier
(Task 7) ranks candidates on the same 10-task search subset — the baseline
must be comparable for the +10 threshold to be meaningful.

Signal is GRADED: pass_rate is the mean partial pass fraction (fraction of each
task's hidden tests that pass), not the binary task-pass count. The binary
signal is flat-0 on BigCodeBench-Hard for Tess-9B (duplicate-signature import
crashes with the one-shot baseline harness), so it gives the search no gradient.
The graded signal does: a harness that fixes the import (AST body extraction)
lifts a task from 0/6 to 5/6 → 0.833, a real climb the search can reward.
"""
from __future__ import annotations

import json
from pathlib import Path

from agent_contract import LLM, load_subset
from bcb_wrapper import check_solution_graded
from harness import solve

ROOT = Path(__file__).parent

# benchmark name -> (tasks dir, baseline file). "hard" = BigCodeBench-Hard (Tess
# at its ceiling, no headroom); "easy" = non-Hard BigCodeBench (headroom test:
# does the harness lift a model NOT already at its ceiling?).
_BENCHMARKS = {
    "hard": ("tasks", "baseline.json"),
    "easy": ("tasks_easy", "baseline_easy.json"),
}


def main() -> None:
    import argparse

    from reward_hacking_gate import is_clean
    from sandbox import run_solve_capped, sandbox_fresh_workdir, set_tasks_root

    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=sorted(_BENCHMARKS), default="hard")
    args = ap.parse_args()
    tasks_dir, baseline_file = _BENCHMARKS[args.benchmark]
    set_tasks_root(ROOT / tasks_dir)  # copy skeletons from the right benchmark dir

    # Search subset only — same 10 tasks the verifier ranks on.
    all_tasks = load_subset(ROOT / tasks_dir / "subset.json")
    tasks = [t for t in all_tasks if t.id.startswith("search/")]
    print(
        f"Measuring baseline [{args.benchmark}] on {len(tasks)} search tasks "
        f"(filtered from {len(all_tasks)} total)"
    )

    llm = LLM(base_url="http://127.0.0.1:9090/v1", model="tess")

    rates: list[float] = []
    for i, t in enumerate(tasks):
        print(f"[{i+1}/{len(tasks)}] {t.id} ...", flush=True)
        wd = sandbox_fresh_workdir(t)
        run_solve_capped(solve, t, wd, llm, max_turns=8, max_tokens=256, wall_s=180)
        sol = (wd / t.skeleton_path).read_text()
        clean = is_clean(sol)
        rate = check_solution_graded(t.eval_task_id, sol) if clean else 0.0
        rates.append(rate)
        print(f"  clean={clean} partial_rate={rate:.3f}", flush=True)

    mean_rate = sum(rates) / len(rates) if rates else 0.0
    (ROOT / baseline_file).write_text(
        json.dumps({"pass_rate": mean_rate, "n": len(tasks)}, indent=2) + "\n"
    )
    print(f"baseline pass_rate (graded)={mean_rate:.3f} on {len(tasks)} tasks")


if __name__ == "__main__":
    main()