"""Wave 0: measure Tess + the minimal harness on the 10-task SEARCH subset.

We measure on the search subset only (not the full 25) because the verifier
(Task 7) ranks candidates on the same 10-task search subset — the baseline
must be comparable for the +10 threshold to be meaningful.
"""
from __future__ import annotations

import json
from pathlib import Path

from agent_contract import LLM, load_subset
from bcb_wrapper import check_solution
from harness import solve

ROOT = Path(__file__).parent


def main() -> None:
    from sandbox import run_solve_capped, sandbox_fresh_workdir
    from reward_hacking_gate import is_clean

    # Search subset only — same 10 tasks the verifier ranks on.
    all_tasks = load_subset(ROOT / "tasks" / "subset.json")
    tasks = [t for t in all_tasks if t.id.startswith("search/")]
    print(f"Measuring baseline on {len(tasks)} search tasks (filtered from {len(all_tasks)} total)")

    llm = LLM(base_url="http://127.0.0.1:9090/v1", model="tess")

    passed = 0
    for i, t in enumerate(tasks):
        print(f"[{i+1}/{len(tasks)}] {t.id} ...", flush=True)
        wd = sandbox_fresh_workdir(t)
        run_solve_capped(solve, t, wd, llm, max_turns=8, max_tokens=256, wall_s=180)
        sol = (wd / t.skeleton_path).read_text()
        clean = is_clean(sol)
        result = check_solution(t.eval_task_id, sol) if clean else False
        status = "PASS" if result else "FAIL"
        if result:
            passed += 1
        print(f"  clean={clean} check={result} -> {status}", flush=True)

    rate = passed / len(tasks)
    (ROOT / "baseline.json").write_text(
        json.dumps({"pass_rate": rate, "n": len(tasks)}, indent=2) + "\n"
    )
    print(f"baseline pass_rate={rate:.3f} ({passed}/{len(tasks)})")


if __name__ == "__main__":
    main()