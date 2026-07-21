"""Wave 0: measure Tess + the minimal harness on the 25-task subset.

Run AFTER Tasks 5+6 land (sandbox.run_solve_capped + reward_hacking_gate.is_clean).
"""
from __future__ import annotations

import json
from pathlib import Path

from agent_contract import LLM, load_subset
from bcb_wrapper import check_solution
from harness import solve

# Task 5 will provide sandbox.run_solve_capped and sandbox_fresh_workdir
# Task 6 will provide reward_hacking_gate.is_clean

ROOT = Path(__file__).parent


def main() -> None:
    from sandbox import run_solve_capped, sandbox_fresh_workdir  # noqa: F401 – Task 5
    from reward_hacking_gate import is_clean  # noqa: F401 – Task 6

    tasks = load_subset(ROOT / "tasks" / "subset.json")
    llm = LLM(base_url="http://127.0.0.1:9090/v1", model="tess")

    passed = 0
    for t in tasks:
        wd = sandbox_fresh_workdir(t)
        run_solve_capped(solve, t, wd, llm, max_turns=8, max_tokens=256, wall_s=180)
        sol = (wd / t.skeleton_path).read_text()
        if is_clean(sol) and check_solution(t.eval_task_id, sol):
            passed += 1

    rate = passed / len(tasks)
    (ROOT / "baseline.json").write_text(
        json.dumps({"pass_rate": rate, "n": len(tasks)}, indent=2) + "\n"
    )
    print(f"baseline pass_rate={rate:.3f} ({passed}/{len(tasks)})")


if __name__ == "__main__":
    main()