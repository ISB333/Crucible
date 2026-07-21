"""Tests for sandbox: fresh workdir isolation + wall-clock watchdog."""
from __future__ import annotations

import time
from pathlib import Path

from sandbox import run_solve_capped, sandbox_fresh_workdir, sandbox_fresh_workdir_to  # type: ignore[import-not-found]
from agent_contract import Task, LLM, Tools  # type: ignore[import-not-found]


def _make_task_with_skeleton(tmp_path, skeleton_content="def solve():\n    pass\n"):
    """Create a source skeleton and return (Task, tasks_root).

    Mimics the production layout: tasks_root/<eval_task_id>/skeleton.py
    """
    tasks_root = tmp_path / "source"
    (tasks_root / "t").mkdir(parents=True)
    (tasks_root / "t" / "skeleton.py").write_text(skeleton_content)
    # skeleton_path matches load_subset convention: Path(eval_task_id) / "skeleton.py"
    task = Task(id="t", spec="do nothing", skeleton_path=Path("t") / "skeleton.py", eval_task_id="t")
    return task, tasks_root


def test_fresh_workdir_is_isolated(tmp_path):
    """The fresh workdir contains a copy of the skeleton at task.skeleton_path."""
    task, tasks_root = _make_task_with_skeleton(tmp_path)
    wd = sandbox_fresh_workdir_to(tmp_path / "wd", task, _tasks_root=tasks_root)
    # The skeleton must be reachable at workdir / task.skeleton_path
    assert (wd / task.skeleton_path).read_text() == "def solve():\n    pass\n"


def test_fresh_workdir_is_a_copy_not_original(tmp_path):
    """Modifying the workdir skeleton must not affect the source."""
    task, tasks_root = _make_task_with_skeleton(tmp_path, "def solve():\n    pass\n")
    wd = sandbox_fresh_workdir_to(tmp_path / "wd", task, _tasks_root=tasks_root)
    # Overwrite in workdir
    (wd / task.skeleton_path).write_text("def solve():\n    return 42\n")
    # Source must be unchanged
    assert (tasks_root / "t" / "skeleton.py").read_text() == "def solve():\n    pass\n"


def test_run_solve_capped_timeout(tmp_path):
    """The watchdog must return within wall_s even if solve sleeps forever."""
    def slow_solve(task, workdir, llm, tools):
        time.sleep(10)

    task, tasks_root = _make_task_with_skeleton(tmp_path)
    wd = sandbox_fresh_workdir_to(tmp_path / "wd", task, _tasks_root=tasks_root)
    start = time.monotonic()
    # wall_s=0.5 -> must cut off the sleeping solve
    run_solve_capped(slow_solve, task, wd, LLM("http://x", "t"), max_turns=1, max_tokens=8, wall_s=0.5)
    elapsed = time.monotonic() - start
    # Should return in ~0.5s, not 10s
    assert elapsed < 5, f"watchdog took {elapsed:.1f}s, expected < 5s"


def test_run_solve_capped_normal_return(tmp_path):
    """When solve finishes before the timeout, run_solve_capped returns normally."""
    def fast_solve(task, workdir, llm, tools):
        pass

    task, tasks_root = _make_task_with_skeleton(tmp_path)
    wd = sandbox_fresh_workdir_to(tmp_path / "wd", task, _tasks_root=tasks_root)
    start = time.monotonic()
    run_solve_capped(fast_solve, task, wd, LLM("http://x", "t"), max_turns=1, max_tokens=8, wall_s=5)
    elapsed = time.monotonic() - start
    # Should return quickly, not wait the full wall_s
    assert elapsed < 2, f"took {elapsed:.1f}s, expected fast return"


def test_run_solve_capped_crash_doesnt_propagate(tmp_path):
    """If solve raises an exception, run_solve_capped still returns cleanly."""
    def crashing_solve(task, workdir, llm, tools):
        raise RuntimeError("harness crash")

    task, tasks_root = _make_task_with_skeleton(tmp_path)
    wd = sandbox_fresh_workdir_to(tmp_path / "wd", task, _tasks_root=tasks_root)
    # Should NOT raise
    run_solve_capped(crashing_solve, task, wd, LLM("http://x", "t"), max_turns=1, max_tokens=8, wall_s=5)