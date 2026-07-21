"""Per-task sandbox: fresh isolated workdir + wall-clock watchdog on the harness solve.

The harness runs in-process (it's trusted Python the worker wrote, evaluated in the
verifier); the SANDBOX guarantees (a) a fresh workdir copy so harness file actions
can't reach frozen eval files, and (b) a hard wall-clock cap so a looping harness
can't hang the verifier.
"""
from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Callable

from agent_contract import Task, LLM, Tools


def sandbox_fresh_workdir(task: Task, base: Path | None = None, *, _tasks_root: Path | None = None) -> Path:
    """Create a fresh isolated workdir for a task, with the skeleton copied in.

    The workdir is structured so that ``workdir / task.skeleton_path`` resolves
    to the skeleton file (matching ``load_subset`` conventions).

    Args:
        task: The task to create a workdir for.
        base: Optional base directory (for testing). If None, a temp dir is created.
        _tasks_root: Optional override for the tasks source directory (test seam).

    Returns:
        The workdir path (``base`` if provided, otherwise a new temp dir).
    """
    import tempfile

    root = Path(base) if base else Path(tempfile.mkdtemp(prefix="agentic_"))
    tasks_root = _tasks_root or Path(__file__).parent / "tasks"
    src = tasks_root / task.eval_task_id / "skeleton.py"
    dst = root / task.skeleton_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return root


# test seam: allow tests to pass a base dir and tasks root
def sandbox_fresh_workdir_to(base: Path, task: Task, *, _tasks_root: Path | None = None) -> Path:
    return sandbox_fresh_workdir(task, base, _tasks_root=_tasks_root)


class _Timeout(Exception):
    pass


def run_solve_capped(solve_fn: Callable, task: Task, workdir: Path, llm: LLM,
                     max_turns: int, max_tokens: int, wall_s: float) -> None:
    """Run solve_fn with a wall-clock watchdog. Returns cleanly on timeout."""
    tools = Tools(workdir)
    result: dict = {"done": False}

    def _run():
        try:
            solve_fn(task, workdir, llm, tools)
        except Exception:
            pass  # harness crash -> task fails upstream (no solution written)
        finally:
            result["done"] = True

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    th.join(wall_s)
    if th.is_alive():
        # watchdog: the daemon thread is abandoned; the task will fail (no/empty solution)
        return