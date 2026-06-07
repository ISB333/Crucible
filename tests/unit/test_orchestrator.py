from pathlib import Path

import pytest

from crucible.artifact import Artifact
from crucible.budgets import EpisodeBudget, RunBudget
from crucible.llm import LLMSession, ScriptedSession, ToolCall
from crucible.orchestrator import search
from crucible.store import Store
from crucible.task import Task
from crucible.verify import Fail, Ok, Partial, RunContext
from tests.unit.conftest import FakeSandbox

pytestmark = pytest.mark.unit

PROBLEM = """\
# crucible:region start name=solution
def solve() -> int:
    raise NotImplementedError
# crucible:region end
"""

SOLUTION = "def solve() -> int:\n    return 42"


def wr(content: str) -> ToolCall:
    return ToolCall(id="1", name="write_region", args={"name": "solution", "content": content})


def make_task(tmp_path: Path) -> Task:
    (tmp_path / "problem.py").write_text(PROBLEM)
    return Task.from_path(tmp_path / "problem.py", editable=["solution"])


class HoleVerifier:
    """Deterministic: OK iff no generic holes (same semantics as StubVerifier)."""

    deterministic = True
    verifier_id = "stub:holes"

    def verify(self, artifact: Artifact, ctx: RunContext):
        from crucible.artifact import scan_holes

        holes = scan_holes(artifact)
        return Partial(open_holes=holes, feedback="open") if holes else Ok(produced=artifact)


def scripted_factory(scripts_by_worker: dict[int, list[list[ToolCall]]]):
    def factory(worker: int, episode: int) -> LLMSession:
        return ScriptedSession(scripts_by_worker.get(worker, [[]]))

    return factory


async def test_first_clean_ok_wins(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    result = await search(
        task=task,
        verifier=HoleVerifier(),
        session_factory=scripted_factory({1: [[wr(SOLUTION)]]}),  # only worker 1 solves
        store=Store(tmp_path / "t.db"),
        sandbox_factory=FakeSandbox,
        model="scripted",
        workers=3,
        episode_budget=EpisodeBudget(),
        run_budget=RunBudget(episodes_per_worker=1),
    )
    assert result.solution is not None
    assert result.winner == 1
    assert "return 42" in result.solution.files["problem.py"]


async def test_unsolved_returns_best_partial_no_hang(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    result = await search(
        task=task,
        verifier=HoleVerifier(),
        session_factory=scripted_factory({}),  # nobody edits anything
        store=Store(tmp_path / "t.db"),
        sandbox_factory=FakeSandbox,
        model="scripted",
        workers=2,
        episode_budget=EpisodeBudget(),
        run_budget=RunBudget(episodes_per_worker=2),
    )
    assert result.solution is None
    assert result.best_partial is not None
    assert result.winner is None


async def test_win_that_fails_fresh_reverify_is_not_accepted(tmp_path: Path) -> None:
    class FlakyVerifier:
        """Lies about determinism: passes inside the worker, fails in isolation."""

        deterministic = True
        verifier_id = "stub:flaky"

        def __init__(self) -> None:
            self.reverify_phase = False
            self.calls = 0

        def verify(self, artifact: Artifact, ctx: RunContext):
            if "crucible-reverify-" in str(ctx.scratch):
                return Fail(feedback="does not reproduce in isolation")
            from crucible.artifact import scan_holes

            holes = scan_holes(artifact)
            return Partial(open_holes=holes, feedback="open") if holes else Ok(produced=artifact)

    task = make_task(tmp_path)
    result = await search(
        task=task,
        verifier=FlakyVerifier(),
        session_factory=scripted_factory({0: [[wr(SOLUTION)]]}),
        store=Store(tmp_path / "t.db"),
        sandbox_factory=FakeSandbox,
        model="scripted",
        workers=1,
        episode_budget=EpisodeBudget(),
        run_budget=RunBudget(episodes_per_worker=1),
    )
    assert result.solution is None  # the gate held


async def test_workers_must_be_positive(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    with pytest.raises(ValueError, match="workers"):
        await search(
            task=task,
            verifier=HoleVerifier(),
            session_factory=scripted_factory({}),
            store=Store(tmp_path / "t.db"),
            sandbox_factory=FakeSandbox,
            model="scripted",
            workers=0,
        )


async def test_worker_exception_still_writes_run_finished(tmp_path: Path) -> None:
    class ExplodingVerifier:
        deterministic = True
        verifier_id = "stub:boom"

        def verify(self, artifact: Artifact, ctx: RunContext):
            raise RuntimeError("verifier infrastructure died")

    task = make_task(tmp_path)
    store = Store(tmp_path / "t.db")
    with pytest.raises(RuntimeError):
        await search(
            task=task,
            verifier=ExplodingVerifier(),
            session_factory=scripted_factory({0: [[wr(SOLUTION)]]}),
            store=store,
            sandbox_factory=FakeSandbox,
            model="scripted",
            workers=1,
            episode_budget=EpisodeBudget(),
            run_budget=RunBudget(episodes_per_worker=1),
        )
    # the run record is closed even on failure (store invariant)
    rows = store._conn.execute(
        "SELECT payload_json FROM events WHERE kind = 'run_finished'"
    ).fetchall()
    assert len(rows) == 1 and '"status": "error"' in rows[0][0]


async def test_run_is_fully_logged(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    store = Store(tmp_path / "t.db")
    result = await search(
        task=task,
        verifier=HoleVerifier(),
        session_factory=scripted_factory({0: [[wr(SOLUTION)]]}),
        store=store,
        sandbox_factory=FakeSandbox,
        model="scripted",
        workers=1,
        episode_budget=EpisodeBudget(),
        run_budget=RunBudget(episodes_per_worker=1),
    )
    assert result.solution is not None
    # replay: the winning artifact reconstructs from the log alone (PRD §13.5)
    replayed = store.load_artifact(result.solution.content_hash)
    assert replayed.files == result.solution.files
