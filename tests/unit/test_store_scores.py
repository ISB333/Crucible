import pytest

from crucible.artifact import Artifact
from crucible.store import Store
from crucible.verify import Scored

pytestmark = pytest.mark.unit


def _seed(store: Store) -> int:
    run = store.start_run(task_root="/x", verifier_id="chem:0.5", model="scripted", config={})
    w = store.add_worker(run, 0)
    return store.add_episode(w, ordinal=0, turns=1, edits=1, end_reason="model_stopped", lessons="")


def test_scored_verdict_persists_kind_and_score(tmp_path) -> None:
    store = Store(tmp_path / "c.db")
    ep = _seed(store)
    a = Artifact.from_files({"m.smi": "CO\n"})
    av = store.add_artifact_version(
        ep, a, holes_remaining=0, cost_usd=0.0, wall_time_s=0.0, integrity_ok=True, score=1.25
    )
    store.add_verdict(av, Scored(produced=a, value=1.25, feedback="below target"))
    row = store._conn.execute(
        "SELECT kind, score FROM verdicts WHERE artifact_version_id = ?", (av,)
    ).fetchone()
    assert row[0] == "SCORED"
    assert row[1] == pytest.approx(1.25)
    avrow = store._conn.execute(
        "SELECT score FROM artifact_versions WHERE id = ?", (av,)
    ).fetchone()
    assert avrow[0] == pytest.approx(1.25)


def test_score_is_optional_for_non_scored(tmp_path) -> None:
    store = Store(tmp_path / "c.db")
    ep = _seed(store)
    a = Artifact.from_files({"m.smi": "CO\n"})
    av = store.add_artifact_version(
        ep, a, holes_remaining=0, cost_usd=0.0, wall_time_s=0.0, integrity_ok=True
    )
    row = store._conn.execute("SELECT score FROM artifact_versions WHERE id = ?", (av,)).fetchone()
    assert row[0] is None
