"""PRD §13 acceptance criteria, one test (or pair) per criterion."""

import sys
from pathlib import Path

import pytest

from crucible.artifact import scan_holes
from crucible.budgets import EpisodeBudget, RunBudget
from crucible.llm import ToolCall
from crucible.orchestrator import SearchResult, search
from crucible.sandbox import DockerSandbox, SubprocessSandbox
from crucible.store import Store
from crucible.task import Task
from crucible.verifiers import Command, Lean, Pyright, Pytest
from crucible.verify import Verifier
from tests.acceptance.conftest import copy_fixture, scripted, wr

SOLUTION = "def double(x: int) -> int:\n    return x * 2"

# SubprocessSandbox passes only a minimal PATH, so pin the venv's interpreter/binaries
# explicitly — the sandbox would not find bare `python3 -m pytest` or `pyright`.
VENV_PYTEST = Pytest(suite="tests/", python=sys.executable)
# VENV_PYRIGHT requires `node` reachable through the sandbox's pruned PATH (pyright is a
# Node.js binary); if the environment is broken the acceptance suite fails loudly — it does
# NOT skip — so a missing `node` in PATH is surfaced immediately rather than silently.
VENV_PYRIGHT = Pyright(strict=False, binary=str(Path(sys.executable).parent / "pyright"))
VENV_CMD = Command(f"{sys.executable} -m pytest tests/ -q")


async def kata_search(
    tmp_path: Path,
    verifier: Verifier,
    scripts: dict[int, list[list[ToolCall]]],
    store: Store | None = None,
    workers: int = 1,
    episodes_per_worker: int = 1,
) -> tuple[SearchResult, Store]:
    root = copy_fixture("kata", tmp_path)
    s = store if store is not None else Store(tmp_path / "c.db")
    result = await search(
        task=Task.from_path(root, editable=["solution"]),
        verifier=verifier,
        session_factory=scripted(scripts),
        store=s,
        sandbox_factory=SubprocessSandbox,
        model="scripted",
        workers=workers,
        episode_budget=EpisodeBudget(edits=5, turns=5),
        run_budget=RunBudget(episodes_per_worker=episodes_per_worker),
    )
    return result, s


# --- §13.2 Anti-gaming: the load-bearing test ---------------------------------


@pytest.mark.integration  # runs the real hidden pytest suite via SubprocessSandbox
async def test_13_2_honest_solution_passes(tmp_path: Path) -> None:
    result, _ = await kata_search(tmp_path, VENV_PYTEST, {0: [[wr(SOLUTION)]]})
    assert result.solution is not None
    assert scan_holes(result.solution) == ()


@pytest.mark.integration
async def test_13_2_editing_the_hidden_suite_is_rejected(tmp_path: Path) -> None:
    tamper = ToolCall(
        id="1",
        name="search_replace",
        args={"file": "tests/test_kata.py", "old": "assert double(2) == 4", "new": "assert True"},
    )
    result, _ = await kata_search(tmp_path, VENV_PYTEST, {0: [[tamper]]})
    assert result.solution is None  # the edit tool refused; nothing passed

    # the edit was refused — the hidden suite is byte-identical in the best partial
    original = (
        Path(__file__).parent.parent / "fixtures" / "kata" / "tests" / "test_kata.py"
    ).read_text()
    assert result.best_partial is not None
    assert result.best_partial.files["tests/test_kata.py"] == original


@pytest.mark.integration
async def test_13_2_escape_token_is_rejected_by_integrity_gate(tmp_path: Path) -> None:
    gamed = "import pytest\npytest.skip('gamed', allow_module_level=True)"
    store = Store(tmp_path / "c.db")
    result, _ = await kata_search(tmp_path, VENV_PYTEST, {0: [[wr(gamed)]]}, store=store)
    assert result.solution is None
    # the episode ended in an integrity violation, recorded in provenance
    rows = store._conn.execute("SELECT end_reason FROM episodes").fetchall()
    assert ("integrity_violation",) in rows


# --- §13.3 Graceful failure ----------------------------------------------------


@pytest.mark.integration
async def test_13_3_unsatisfiable_spec_returns_best_partial(tmp_path: Path) -> None:
    root = copy_fixture("kata_unsat", tmp_path)
    store = Store(tmp_path / "c.db")
    result = await search(
        task=Task.from_path(root, editable=["solution"]),
        verifier=VENV_PYTEST,
        session_factory=scripted({0: [[wr(SOLUTION)]]}),
        store=store,
        sandbox_factory=SubprocessSandbox,
        model="scripted",
        workers=2,
        episode_budget=EpisodeBudget(edits=5, turns=5),
        run_budget=RunBudget(episodes_per_worker=2),
    )
    assert result.solution is None
    assert result.best_partial is not None  # no hang, no crash, budget exhausted


# --- §13.4 Abstraction proof: swap the verifier, zero engine changes -----------


@pytest.mark.integration
@pytest.mark.parametrize(
    "verifier",
    [VENV_PYTEST, VENV_PYRIGHT, VENV_CMD],
    ids=["pytest", "pyright", "command"],
)
async def test_13_4_verifier_swap_needs_zero_engine_changes(
    tmp_path: Path, verifier: Verifier
) -> None:
    # The identical search() call solves the identical task under three verifiers.
    result, _ = await kata_search(tmp_path, verifier, {0: [[wr(SOLUTION)]]})
    assert result.solution is not None


# --- §13.5 Replayability --------------------------------------------------------


@pytest.mark.unit
async def test_13_5_run_reconstructs_from_log_alone(tmp_path: Path) -> None:
    root = copy_fixture("kata", tmp_path)
    db = tmp_path / "c.db"
    result = await search(
        task=Task.from_path(root, editable=["solution"]),
        verifier=Command("true"),
        session_factory=scripted({0: [[wr(SOLUTION)]]}),
        store=Store(db),
        sandbox_factory=SubprocessSandbox,
        model="scripted",
        workers=1,
        episode_budget=EpisodeBudget(edits=5, turns=5),
        run_budget=RunBudget(episodes_per_worker=1),
    )
    assert result.solution is not None
    # a brand-new Store handle on the same file reconstructs the winner
    fresh = Store(db)
    replayed = fresh.load_artifact(result.solution.content_hash)
    assert replayed.files == result.solution.files
    assert fresh.export_v_shared(result.run_id)  # verdict trail present


# --- §13.1 Lean domain parity ----------------------------------------------------


@pytest.mark.integration  # needs crucible-lean:0 (Task 4)
async def test_13_1_lean_toy_solved_sorry_free(tmp_path: Path) -> None:
    root = tmp_path / "lean"
    root.mkdir()
    src = Path(__file__).parent.parent / "fixtures" / "lean" / "toy_add_comm.lean"
    (root / "toy_add_comm.lean").write_text(src.read_text())
    prove = ToolCall(
        id="1",
        name="write_region",
        args={"name": "proof", "content": "  exact Nat.add_comm a b"},
    )
    store = Store(tmp_path / "c.db")
    result = await search(
        task=Task.from_path(root, editable=["proof"]),
        verifier=Lean(),
        session_factory=scripted({0: [[prove]]}),
        store=store,
        sandbox_factory=lambda: DockerSandbox(image="crucible-lean:0"),
        model="scripted",
        workers=1,
        episode_budget=EpisodeBudget(edits=5, turns=5),
        run_budget=RunBudget(episodes_per_worker=1),
    )
    assert result.solution is not None
    assert "sorry" not in result.solution.files["toy_add_comm.lean"]
    # the initial verify saw the sorry as a PARTIAL hole — it is in the verdict trail
    kinds = [r["rule_constraint"]["verdict"] for r in store.export_v_shared(result.run_id)]
    assert "OK" in kinds
