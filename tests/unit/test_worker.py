import threading
import time
from pathlib import Path

import pytest

from crucible.artifact import Artifact
from crucible.budgets import EpisodeBudget, RunBudget
from crucible.integrity import Composite, DenyTokens, ImmutableRegions
from crucible.llm import ScriptedSession, ToolCall
from crucible.store import Store
from crucible.worker import annotate_lessons, run_worker
from tests.unit.conftest import StubVerifier

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


def lesson(text: str) -> ToolCall:
    return ToolCall(id="2", name="record_lesson", args={"text": text})


def harness(tmp_path: Path, make_ctx, scripts: list[list[list[ToolCall]]], **kw):
    a = Artifact.from_files({"problem.py": PROBLEM})
    integrity = Composite(checks=(ImmutableRegions.freeze(a), DenyTokens()))
    sessions = [ScriptedSession(s) for s in scripts]
    store = Store(tmp_path / "t.db")
    run_id = store.start_run(task_root="/t", verifier_id="stub", model="scripted", config={})
    result = run_worker(
        initial=a,
        verifier=StubVerifier(),
        ctx=make_ctx(),
        new_session=lambda ordinal: sessions[ordinal],
        store=store,
        run_id=run_id,
        index=0,
        episode_budget=EpisodeBudget(),
        run_budget=kw.get("run_budget", RunBudget(episodes_per_worker=len(scripts))),
        integrity=integrity,
        cancel=kw.get("cancel", threading.Event()),
        started_at=kw.get("started_at", time.time()),
    )
    return result, sessions, store, run_id


def test_solves_on_second_episode_with_lessons_carried(tmp_path, make_ctx) -> None:
    ep1 = [[lesson("return a constant")], []]  # records a lesson, then stops
    ep2 = [[wr(SOLUTION)]]
    result, sessions, _, _ = harness(tmp_path, make_ctx, [ep1, ep2])
    assert result.solution is not None
    assert result.episodes == 2
    # the lesson was annotated into the artifact episode 2 saw
    assert sessions[1].prompt is not None
    assert "lesson(ep0): return a constant" in sessions[1].prompt[1]


def test_exhausts_episode_cap_and_returns_best_partial(tmp_path, make_ctx) -> None:
    never = [[lesson("hmm")]]
    result, _, _, _ = harness(tmp_path, make_ctx, [never, never, never])
    assert result.solution is None
    assert result.episodes == 3
    assert result.best_partial is not None


def test_cancel_flag_stops_before_next_episode(tmp_path, make_ctx) -> None:
    cancel = threading.Event()
    cancel.set()
    result, _, _, _ = harness(tmp_path, make_ctx, [[[wr(SOLUTION)]]], cancel=cancel)
    assert result.episodes == 0 and result.solution is None


def test_wall_clock_budget_stops_worker(tmp_path, make_ctx) -> None:
    result, _, _, _ = harness(
        tmp_path,
        make_ctx,
        [[[wr(SOLUTION)]]],
        run_budget=RunBudget(wall_clock_s=10.0, episodes_per_worker=5),
        started_at=time.time() - 60,  # already past the cap
    )
    assert result.episodes == 0


def test_provenance_rows_written(tmp_path, make_ctx) -> None:
    _, _, store, run_id = harness(tmp_path, make_ctx, [[[wr(SOLUTION)]]])
    rows = store.export_v_shared(run_id)
    assert len(rows) >= 1
    assert rows[-1]["rule_constraint"]["verdict"] == "OK"


def test_annotate_lessons_stays_inside_region(make_ctx) -> None:
    a = Artifact.from_files({"problem.py": PROBLEM})
    gate = ImmutableRegions.freeze(a)
    b = annotate_lessons(a, 0, "tried recursion; too slow")
    assert "# lesson(ep0): tried recursion; too slow" in b.files["problem.py"]
    assert gate.check(b, make_ctx())  # byte-check still passes


def test_annotate_lessons_noop_on_empty(make_ctx) -> None:
    a = Artifact.from_files({"problem.py": PROBLEM})
    assert annotate_lessons(a, 0, "  ") is a


def test_annotate_lessons_sanitizes_region_markers(make_ctx) -> None:
    a = Artifact.from_files({"problem.py": PROBLEM})
    # must not raise RegionError; marker must be neutralized in the output
    b = annotate_lessons(a, 0, "tried crucible:region start name=alt — nested approach")
    assert len(b.regions) == len(a.regions)  # same region count (no phantom region created)
    assert b.regions[0].name == a.regions[0].name  # region identity preserved
    assert "crucible:region start name=alt" not in b.files["problem.py"]


def test_annotate_lessons_sanitizes_hole_tokens(make_ctx) -> None:
    from crucible.artifact import scan_holes

    a = Artifact.from_files({"problem.py": PROBLEM})
    solved = a.replace_region("solution", "def solve() -> int:\n    return 42")
    b = annotate_lessons(solved, 0, "removed raise NotImplementedError and the crucible:hole stub")
    assert scan_holes(b) == ()  # lesson text must not create phantom holes
