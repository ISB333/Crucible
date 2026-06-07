"""Orchestrator (PRD §6): N independent workers, first integrity-clean OK wins.

Cancellation is cooperative: workers check the flag between episodes, so after a win
the run drains within at most one in-flight episode per worker. v0 accepts that.
"""

import asyncio
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from crucible.artifact import Artifact
from crucible.budgets import EpisodeBudget, RunBudget
from crucible.integrity import Composite, DenyTokens, ImmutableRegions, fresh_reverify
from crucible.llm import LLMSession
from crucible.sandbox import Sandbox
from crucible.store import Store
from crucible.task import Task
from crucible.verify import RunContext, Verifier
from crucible.worker import WorkerResult, run_worker


@dataclass(frozen=True)
class SearchResult:
    solution: Artifact | None  # hole-free, verified, integrity-clean — or None
    best_partial: Artifact  # highest-progress attempt
    run_id: int
    winner: int | None
    cost_usd: float


def verifier_id_of(verifier: Verifier) -> str:
    return getattr(verifier, "verifier_id", type(verifier).__name__.lower())


async def search(
    *,
    task: Task,
    verifier: Verifier,
    session_factory: Callable[[int, int], LLMSession],  # (worker, episode) -> session
    store: Store,
    sandbox_factory: Callable[[], Sandbox],
    model: str,
    workers: int = 10,
    episode_budget: EpisodeBudget | None = None,
    run_budget: RunBudget | None = None,
) -> SearchResult:
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    eb = episode_budget or EpisodeBudget()
    rb = run_budget or RunBudget()
    initial = Artifact.from_files(task.load_files())
    integrity = Composite(
        checks=(ImmutableRegions.freeze(initial), DenyTokens(extra=task.deny_tokens))
    )
    run_id = store.start_run(
        task_root=str(task.root),
        verifier_id=verifier_id_of(verifier),
        model=model,
        config={
            "workers": workers,
            "episode_budget": {"edits": eb.edits, "turns": eb.turns},
            "run_budget": {
                "wall_clock_s": rb.wall_clock_s,
                "usd": rb.usd,
                "episodes_per_worker": rb.episodes_per_worker,
                "plateau_patience": rb.plateau_patience,
            },
        },
    )
    cancel = threading.Event()
    started_at = time.time()
    scratch_root = Path(tempfile.mkdtemp(prefix="crucible-run-"))

    def one(i: int) -> WorkerResult:
        ctx = RunContext(task=task, sandbox=sandbox_factory(), scratch=scratch_root / f"w{i}")
        return run_worker(
            initial=initial,
            verifier=verifier,
            ctx=ctx,
            new_session=lambda ordinal: session_factory(i, ordinal),
            store=store,
            run_id=run_id,
            index=i,
            episode_budget=eb,
            run_budget=rb,
            integrity=integrity,
            cancel=cancel,
            started_at=started_at,
        )

    tasks = [asyncio.create_task(asyncio.to_thread(one, i)) for i in range(workers)]
    solution: Artifact | None = None
    winner: int | None = None
    try:
        for fut in asyncio.as_completed(list(tasks)):
            r = await fut
            if solution is None and r.solution is not None:
                accepted = await asyncio.to_thread(
                    fresh_reverify, r.solution, verifier, task, sandbox_factory
                )
                if accepted:
                    solution, winner = r.solution, r.index
                    cancel.set()  # remaining workers stop at their next episode boundary
                    store.add_event(run_id, "worker_won", {"worker": r.index})
                else:
                    store.add_event(run_id, "reverify_failed", {"worker": r.index})
        results = [t.result() for t in tasks]
    except BaseException:
        store.add_event(run_id, "run_finished", {"status": "error"})
        raise
    cost = sum(r.cost_usd for r in results)
    best_partial = solution or max(results, key=lambda r: r.best_rank).best_partial
    store.add_event(
        run_id,
        "run_finished",
        {"status": "solved" if solution else "exhausted", "winner": winner, "cost_usd": cost},
    )
    shutil.rmtree(scratch_root, ignore_errors=True)  # workers all done on this path
    return SearchResult(
        solution=solution, best_partial=best_partial, run_id=run_id, winner=winner, cost_usd=cost
    )


def run_search(**kwargs) -> SearchResult:
    """Synchronous wrapper for the SDK/CLI."""
    return asyncio.run(search(**kwargs))
