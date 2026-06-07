import inspect
from pathlib import Path

import pytest

import crucible.store
from crucible.artifact import Artifact
from crucible.store import Store
from crucible.verify import Fail, Ok, Partial

pytestmark = pytest.mark.unit


def make_artifact() -> Artifact:
    return Artifact.from_files(
        {"f.py": "# crucible:region start name=s\nx = 1\n# crucible:region end\n"}
    )


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "test.db")


def test_full_insert_chain(store: Store) -> None:
    a = make_artifact()
    run_id = store.start_run(task_root="/t", verifier_id="pytest:tests/", model="m", config={})
    worker_id = store.add_worker(run_id, idx=0)
    ep_id = store.add_episode(
        worker_id, ordinal=0, turns=3, edits=2, end_reason="solved", lessons=""
    )
    av_id = store.add_artifact_version(
        ep_id,
        a,
        holes_remaining=0,
        cost_usd=0.1,
        wall_time_s=1.5,
        integrity_ok=True,
    )
    store.add_verdict(av_id, Ok(produced=a))
    store.add_event(run_id, "run_finished", {"winner_worker": 0})
    assert store.load_artifact(a.content_hash).content_hash == a.content_hash


def test_load_artifact_round_trip_preserves_regions(store: Store) -> None:
    a = make_artifact()
    run_id = store.start_run(task_root="/t", verifier_id="v", model="m", config={})
    worker_id = store.add_worker(run_id, idx=0)
    ep_id = store.add_episode(worker_id, ordinal=0, turns=1, edits=0, end_reason="x", lessons="")
    store.add_artifact_version(
        ep_id, a, holes_remaining=1, cost_usd=0, wall_time_s=0, integrity_ok=True
    )
    loaded = store.load_artifact(a.content_hash)
    assert loaded.files == a.files
    assert loaded.regions == a.regions


def test_verdict_kinds_recorded(store: Store) -> None:
    a = make_artifact()
    run_id = store.start_run(task_root="/t", verifier_id="v", model="m", config={})
    worker_id = store.add_worker(run_id, idx=0)
    ep_id = store.add_episode(worker_id, ordinal=0, turns=1, edits=1, end_reason="x", lessons="")
    av_id = store.add_artifact_version(
        ep_id, a, holes_remaining=0, cost_usd=0, wall_time_s=0, integrity_ok=True
    )
    store.add_verdict(av_id, Partial(open_holes=(), feedback="open goals"))
    store.add_verdict(av_id, Fail(feedback="boom"))
    rows = store.export_v_shared(run_id)
    assert [r["rule_constraint"]["verdict"] for r in rows] == ["PARTIAL", "FAIL"]


def test_v_shared_shape(store: Store) -> None:
    a = make_artifact()
    run_id = store.start_run(task_root="/t", verifier_id="pytest:tests/", model="m", config={})
    worker_id = store.add_worker(run_id, idx=0)
    ep_id = store.add_episode(worker_id, ordinal=0, turns=1, edits=1, end_reason="x", lessons="")
    av_id = store.add_artifact_version(
        ep_id, a, holes_remaining=0, cost_usd=0, wall_time_s=0, integrity_ok=True
    )
    store.add_verdict(av_id, Ok(produced=a))
    (row,) = store.export_v_shared(run_id)
    assert set(row) == {"state", "action", "observation", "rule_constraint"}
    assert row["state"] == a.content_hash
    assert row["rule_constraint"] == {"type": "verifier", "ref": "pytest:tests/", "verdict": "OK"}


def test_store_source_is_append_only() -> None:
    src = inspect.getsource(crucible.store).upper()
    for forbidden in ("UPDATE ", "DELETE ", " OR REPLACE"):
        assert forbidden not in src
