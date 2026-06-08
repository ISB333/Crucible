"""Append-only SQLite provenance store (PRD §8). Insert and select only — never mutate."""

import json
import sqlite3
import threading
import time
from pathlib import Path

from crucible.artifact import Artifact
from crucible.verify import Fail, Ok, Partial, Scored, Verdict

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    started_at REAL NOT NULL,
    task_root TEXT NOT NULL,
    verifier_id TEXT NOT NULL,
    model TEXT NOT NULL,
    config_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    at REAL NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS workers (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    idx INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY,
    worker_id INTEGER NOT NULL REFERENCES workers(id),
    ordinal INTEGER NOT NULL,
    turns INTEGER NOT NULL,
    edits INTEGER NOT NULL,
    end_reason TEXT NOT NULL,
    lessons TEXT NOT NULL,
    reasoning_json TEXT  -- LLM reasoning and conversation history
);
CREATE TABLE IF NOT EXISTS artifact_versions (
    id INTEGER PRIMARY KEY,
    episode_id INTEGER NOT NULL REFERENCES episodes(id),
    content_hash TEXT NOT NULL,
    parent_hash TEXT,
    holes_remaining INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    wall_time_s REAL NOT NULL,
    integrity_ok INTEGER NOT NULL,
    files_json TEXT NOT NULL,
    score REAL
);
CREATE TABLE IF NOT EXISTS verdicts (
    id INTEGER PRIMARY KEY,
    artifact_version_id INTEGER NOT NULL REFERENCES artifact_versions(id),
    kind TEXT NOT NULL,
    feedback TEXT NOT NULL,
    locus TEXT,
    score REAL
);
"""


def _kind(v: Verdict) -> str:
    match v:
        case Ok():
            return "OK"
        case Partial():
            return "PARTIAL"
        case Fail():
            return "FAIL"
        case Scored():
            return "SCORED"


class Store:
    """Thread-safe (workers run in threads): one connection guarded by a lock."""

    def __init__(self, path: str | Path = "crucible.db") -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._lock = threading.Lock()

    def _migrate(self) -> None:
        for table in ("artifact_versions", "verdicts"):
            cols = {r[1] for r in self._conn.execute(f"PRAGMA table_info({table})")}
            if "score" not in cols:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN score REAL")
        self._conn.commit()

    def _insert(self, sql: str, params: tuple[object, ...]) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(sql, params)
            assert cur.lastrowid is not None
            return cur.lastrowid

    def start_run(self, *, task_root: str, verifier_id: str, model: str, config: dict) -> int:
        return self._insert(
            "INSERT INTO runs (started_at, task_root, verifier_id, model, config_json)"
            " VALUES (?, ?, ?, ?, ?)",
            (time.time(), task_root, verifier_id, model, json.dumps(config, sort_keys=True)),
        )

    def add_event(self, run_id: int, kind: str, payload: dict) -> int:
        return self._insert(
            "INSERT INTO events (run_id, at, kind, payload_json) VALUES (?, ?, ?, ?)",
            (run_id, time.time(), kind, json.dumps(payload, sort_keys=True)),
        )

    def add_worker(self, run_id: int, idx: int) -> int:
        return self._insert("INSERT INTO workers (run_id, idx) VALUES (?, ?)", (run_id, idx))

    def add_episode(
        self,
        worker_id: int,
        *,
        ordinal: int,
        turns: int,
        edits: int,
        end_reason: str,
        lessons: str,
        reasoning: str = "",
    ) -> int:
        return self._insert(
            "INSERT INTO episodes (worker_id, ordinal, turns, edits, end_reason, lessons, reasoning_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (worker_id, ordinal, turns, edits, end_reason, lessons, reasoning),
        )

    def add_artifact_version(
        self,
        episode_id: int,
        artifact: Artifact,
        *,
        holes_remaining: int,
        cost_usd: float,
        wall_time_s: float,
        integrity_ok: bool,
        score: float | None = None,
    ) -> int:
        return self._insert(
            "INSERT INTO artifact_versions (episode_id, content_hash, parent_hash,"
            " holes_remaining, cost_usd, wall_time_s, integrity_ok, files_json, score)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                episode_id,
                artifact.content_hash,
                artifact.parent_hash,
                holes_remaining,
                cost_usd,
                wall_time_s,
                int(integrity_ok),
                json.dumps(dict(artifact.files), sort_keys=True),
                score,
            ),
        )

    def add_verdict(self, artifact_version_id: int, verdict: Verdict) -> int:
        feedback = ""
        locus = None
        score: float | None = None
        match verdict:
            case Partial(feedback=fb):
                feedback = fb
            case Fail(feedback=fb, locus=span):
                feedback = fb
                if span is not None:
                    locus = f"{span.file}:{span.line_start}-{span.line_end}"
            case Ok():
                pass
            case Scored(value=value, feedback=fb):
                feedback = fb
                score = value
        return self._insert(
            "INSERT INTO verdicts (artifact_version_id, kind, feedback, locus, score)"
            " VALUES (?, ?, ?, ?, ?)",
            (artifact_version_id, _kind(verdict), feedback, locus, score),
        )

    def load_artifact(self, content_hash: str) -> Artifact:
        with self._lock:
            row = self._conn.execute(
                "SELECT files_json, parent_hash FROM artifact_versions"
                " WHERE content_hash = ? ORDER BY id LIMIT 1",
                (content_hash,),
            ).fetchone()
        if row is None:
            raise KeyError(f"no artifact version with hash {content_hash!r}")
        return Artifact.from_files(json.loads(row[0]), parent_hash=row[1])

    def export_v_shared(self, run_id: int) -> list[dict]:
        """Verdicts in the frozen v_shared shape (PRD §8) for later Argus ingestion."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT av.content_hash, r.verifier_id, v.kind, v.feedback"
                " FROM verdicts v"
                " JOIN artifact_versions av ON av.id = v.artifact_version_id"
                " JOIN episodes e ON e.id = av.episode_id"
                " JOIN workers w ON w.id = e.worker_id"
                " JOIN runs r ON r.id = w.run_id"
                " WHERE r.id = ? ORDER BY v.id",
                (run_id,),
            ).fetchall()
        return [
            {
                "state": content_hash,
                "action": {"type": "verify", "verifier": verifier_id},
                "observation": feedback,
                "rule_constraint": {
                    "type": "verifier",
                    "ref": verifier_id,
                    "verdict": kind,
                },
            }
            for content_hash, verifier_id, kind, feedback in rows
        ]
