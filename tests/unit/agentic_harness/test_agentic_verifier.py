"""Tests for AgenticCodingVerifier — the core verifier for the agentic harness experiment.

Tests use a runner seam (fake) to avoid needing Tess. The real-Tess integration
is deferred to Task 8's dry-run.

Five verdict paths tested:
  (a) hole unfilled → Partial
  (b) harness import failure → Fail
  (c) runner crashes → Fail
  (d) rate ≥ baseline + 0.10 → Ok
  (e) rate below threshold → Scored(value=rate)
Plus: sys.path fix ensures a harness that imports from agent_contract loads successfully.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from agentic_verifier import AgenticCodingVerifier  # type: ignore[import-not-found]
from crucible.artifact import Artifact
from crucible.verify import Fail, Ok, Partial, Scored


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _artifact(solve_body: str) -> Artifact:
    """Build a harness artifact with the given solve-region body (indented, no def line)."""
    harness = (
        "from pathlib import Path\n"
        "from agent_contract import Task, LLM, Tools\n\n"
        "def solve(task: Task, workdir: Path, llm: LLM, tools: Tools) -> None:\n"
        "# crucible:region start name=solve\n"
        f"{solve_body}\n"
        "# crucible:region end\n"
    )
    return Artifact.from_files({
        "harness.py": harness,
        "agent_contract.py": "Task = LLM = Tools = None\n",
    })


class _Ctx:
    """Minimal fake RunContext that materializes artifacts to a temp dir."""

    def __init__(self, tmp_path: Path):
        self.scratch = tmp_path

    def materialize(self, artifact: Artifact) -> Path:
        import shutil
        from uuid import uuid4

        dst = self.scratch / artifact.content_hash[:16]
        if not dst.exists():
            tmp = self.scratch / f"{dst.name}.{uuid4().hex}.tmp"
            tmp.mkdir(parents=True, exist_ok=True)
            for rel, text in artifact.files.items():
                p = tmp / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text)
            try:
                tmp.rename(dst)
            except OSError:
                shutil.rmtree(tmp, ignore_errors=True)
        return dst


def _make_verifier(
    tmp_path: Path,
    *,
    baseline_rate: float = 0.0,
    baseline_n: int = 10,
    runner=None,
) -> AgenticCodingVerifier:
    """Create a verifier with a temp baseline.json and optional runner seam."""
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"pass_rate": baseline_rate, "n": baseline_n}))
    return AgenticCodingVerifier(
        baseline_path=baseline_path,
        subset_path=tmp_path / "subset.json",
        runner=runner,
    )


def _fake_tasks(n: int = 10):
    """Fake search tasks for monkeypatching _search_subset."""
    from agent_contract import Task  # type: ignore[import-not-found]
    return [
        Task(
            id=f"search/BigCodeBench/{i}",
            spec=f"spec{i}",
            skeleton_path=Path(f"BigCodeBench/{i}/skeleton.py"),
            eval_task_id=f"BigCodeBench/{i}",
        )
        for i in range(n)
    ]


# Save original _search_subset before any monkeypatching
_ORIG_SEARCH_SUBSET = AgenticCodingVerifier._search_subset


# ---------------------------------------------------------------------------
# (a) hole unfilled → Partial
# ---------------------------------------------------------------------------

def test_hole_unfilled_returns_partial(tmp_path):
    """A solve region still containing raise NotImplementedError → Partial."""
    v = _make_verifier(tmp_path)
    art = _artifact("    raise NotImplementedError")
    result = v.verify(art, _Ctx(tmp_path))
    assert isinstance(result, Partial), f"expected Partial, got {type(result).__name__}"
    assert "NotImplementedError" in result.feedback


# ---------------------------------------------------------------------------
# (b) harness import failure → Fail
# ---------------------------------------------------------------------------

def test_harness_import_failure_returns_fail(tmp_path):
    """A harness that fails to import → Fail."""
    v = _make_verifier(tmp_path)
    harness = (
        "import nonexistent_module_xyz\n\n"
        "def solve(task, workdir, llm, tools):\n"
        "# crucible:region start name=solve\n"
        "    pass\n"
        "# crucible:region end\n"
    )
    art = Artifact.from_files({"harness.py": harness, "agent_contract.py": "x = 1\n"})
    result = v.verify(art, _Ctx(tmp_path))
    assert isinstance(result, Fail), f"expected Fail, got {type(result).__name__}"
    assert "did not import" in result.feedback.lower()


# ---------------------------------------------------------------------------
# (c) runner crashes → Fail
# ---------------------------------------------------------------------------

def test_runner_crashes_returns_fail(tmp_path, monkeypatch):
    """If the runner seam crashes → Fail."""
    def crash_runner(harness_mod, tasks, ws):
        raise RuntimeError("boom")

    v = _make_verifier(tmp_path, runner=crash_runner)
    art = _artifact("    pass")
    monkeypatch.setattr(AgenticCodingVerifier, "_search_subset", lambda self: _fake_tasks())
    result = v.verify(art, _Ctx(tmp_path))
    assert isinstance(result, Fail), f"expected Fail, got {type(result).__name__}"
    assert "crashed" in result.feedback.lower()


# ---------------------------------------------------------------------------
# (d) rate ≥ baseline + target_lift → Ok
# ---------------------------------------------------------------------------

def test_rate_meets_threshold_returns_ok(tmp_path, monkeypatch):
    """Pass rate >= baseline + target_lift → Ok."""
    def good_runner(harness_mod, tasks, ws):
        return {"pass": 2, "n": 10}  # 0.2 >= 0.0 + 0.10

    v = _make_verifier(tmp_path, baseline_rate=0.0, runner=good_runner)
    art = _artifact("    pass")
    monkeypatch.setattr(AgenticCodingVerifier, "_search_subset", lambda self: _fake_tasks())
    result = v.verify(art, _Ctx(tmp_path))
    assert isinstance(result, Ok), f"expected Ok, got {type(result).__name__}"


# ---------------------------------------------------------------------------
# (e) rate below threshold → Scored
# ---------------------------------------------------------------------------

def test_rate_below_threshold_returns_scored(tmp_path, monkeypatch):
    """Pass rate < baseline + target_lift → Scored(value=rate)."""
    def bad_runner(harness_mod, tasks, ws):
        return {"pass": 0, "n": 10}  # 0.0 < 0.0 + 0.10

    v = _make_verifier(tmp_path, baseline_rate=0.0, runner=bad_runner)
    art = _artifact("    pass")
    monkeypatch.setattr(AgenticCodingVerifier, "_search_subset", lambda self: _fake_tasks())
    result = v.verify(art, _Ctx(tmp_path))
    assert isinstance(result, Scored), f"expected Scored, got {type(result).__name__}"
    assert abs(result.value - 0.0) < 1e-9
    assert "pass_rate" in result.feedback


# ---------------------------------------------------------------------------
# sys.path fix: harness that imports from agent_contract loads successfully
# ---------------------------------------------------------------------------

def test_sys_path_fix_allows_agent_contract_import(tmp_path):
    """A harness doing `from agent_contract import ...` loads successfully
    when agent_contract.py is in the materialized workspace (sys.path fix)."""
    v = _make_verifier(tmp_path)
    art = _artifact("    pass")
    ws = _Ctx(tmp_path).materialize(art)
    harness_mod = v._load_harness(ws)
    # The harness loaded without ImportError — the sys.path fix worked.
    # Verify solve is callable (proves the import from agent_contract succeeded).
    assert callable(getattr(harness_mod, "solve", None)), "harness module must expose solve()"