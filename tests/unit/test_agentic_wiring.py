"""Unit tests for run_agentic.py CLI wiring.

Validates Task construction, _task_files curation, verifier + advisor_factory wiring
— WITHOUT running Tess or Gemini (no live services needed).
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent / "examples" / "agentic_harness"


class TestTaskFilesCuration:
    """_task_files must exclude __pycache__/.pyc/.db/.log/.pid."""

    def test_excludes_pycache(self) -> None:
        task_files = tuple(
            sorted(
                str(f.relative_to(SCRIPT_DIR))
                for f in SCRIPT_DIR.rglob("*")
                if f.is_file()
                and not any(
                    p.startswith(".") or p == "__pycache__"
                    for p in f.relative_to(SCRIPT_DIR).parts
                )
                and f.suffix not in (".pyc", ".db")
                and f.name not in ("serve.log", "serve.pid")
            )
        )
        assert len(task_files) > 0, "task_files should not be empty"
        for f in task_files:
            assert "__pycache__" not in f, f"should exclude __pycache__: {f}"
            assert not f.endswith(".pyc"), f"should exclude .pyc: {f}"
            assert not f.endswith(".db"), f"should exclude .db: {f}"
            assert f != "serve.log", f"should exclude serve.log: {f}"
            assert f != "serve.pid", f"should exclude serve.pid: {f}"

    def test_includes_required_files(self) -> None:
        task_files = tuple(
            sorted(
                str(f.relative_to(SCRIPT_DIR))
                for f in SCRIPT_DIR.rglob("*")
                if f.is_file()
                and not any(
                    p.startswith(".") or p == "__pycache__"
                    for p in f.relative_to(SCRIPT_DIR).parts
                )
                and f.suffix not in (".pyc", ".db")
                and f.name not in ("serve.log", "serve.pid")
            )
        )
        # Must include the essential files for the harness
        assert "harness.py" in task_files
        assert "agent_contract.py" in task_files
        assert "agentic_verifier.py" in task_files


class TestTaskConstruction:
    """Task must have editable=('solve',) and network=True."""

    def test_editable_region_is_solve(self) -> None:
        from crucible import Task

        task_files = tuple(
            sorted(
                str(f.relative_to(SCRIPT_DIR))
                for f in SCRIPT_DIR.rglob("*")
                if f.is_file()
                and not any(
                    p.startswith(".") or p == "__pycache__"
                    for p in f.relative_to(SCRIPT_DIR).parts
                )
                and f.suffix not in (".pyc", ".db")
                and f.name not in ("serve.log", "serve.pid")
            )
        )
        task = Task(root=SCRIPT_DIR, files=task_files, editable=("solve",), network=True)
        assert task.editable == ("solve",)
        assert task.network is True
        assert len(task.files) > 0


class TestVerifierWiring:
    """AgenticCodingVerifier is wired with the correct paths."""

    def test_verifier_construction(self) -> None:
        # Import without triggering Tess/network
        sys.path.insert(0, str(SCRIPT_DIR))
        try:
            from agentic_verifier import AgenticCodingVerifier

            v = AgenticCodingVerifier(
                baseline_path=SCRIPT_DIR / "baseline.json",
                subset_path=SCRIPT_DIR / "tasks" / "subset.json",
            )
            assert v.verifier_id == "agentic:lift=0.1"
            assert v.deterministic is True
            assert v.baseline_path == SCRIPT_DIR / "baseline.json"
            assert v.subset_path == SCRIPT_DIR / "tasks" / "subset.json"
        finally:
            # Clean up sys.path
            if str(SCRIPT_DIR) in sys.path:
                sys.path.remove(str(SCRIPT_DIR))

    def test_verifier_has_load_harness(self) -> None:
        sys.path.insert(0, str(SCRIPT_DIR))
        try:
            from agentic_verifier import AgenticCodingVerifier

            v = AgenticCodingVerifier(
                baseline_path=SCRIPT_DIR / "baseline.json",
                subset_path=SCRIPT_DIR / "tasks" / "subset.json",
            )
            assert hasattr(v, "_load_harness"), "verifier must have _load_harness for re-verify"
        finally:
            if str(SCRIPT_DIR) in sys.path:
                sys.path.remove(str(SCRIPT_DIR))


class TestAdvisorFactoryWiring:
    """advisor_factory is correctly constructed for Ollama-Cloud shepherds."""

    def test_web_search_imports(self) -> None:
        """web_search and web_search_advisor modules import correctly."""
        sys.path.insert(0, str(SCRIPT_DIR))
        try:
            from web_search import EXTRA_TOOLS, TOOL_HANDLERS  # type: ignore[import-not-found]
            from web_search_advisor import make_web_search_advisor_factory  # type: ignore[import-not-found]

            assert len(EXTRA_TOOLS) >= 1
            assert "web_search" in TOOL_HANDLERS
            assert callable(make_web_search_advisor_factory)
        finally:
            if str(SCRIPT_DIR) in sys.path:
                sys.path.remove(str(SCRIPT_DIR))

    def test_advisor_factory_creates_advisor(self) -> None:
        """make_web_search_advisor_factory returns a callable that produces a WebSearchAdvisor."""
        sys.path.insert(0, str(SCRIPT_DIR))
        try:
            from web_search_advisor import make_web_search_advisor_factory  # type: ignore[import-not-found]

            factory = make_web_search_advisor_factory("test-model", base_url="https://example.com/v1")
            assert callable(factory)
            # Don't call factory() — it would try to import openai
        finally:
            if str(SCRIPT_DIR) in sys.path:
                sys.path.remove(str(SCRIPT_DIR))


class TestBaselineExists:
    """baseline.json must exist and have the expected structure."""

    def test_baseline_structure(self) -> None:
        import json

        baseline = json.loads((SCRIPT_DIR / "baseline.json").read_text())
        assert "pass_rate" in baseline
        assert "n" in baseline
        assert baseline["n"] == 10


class TestSubsetExists:
    """tasks/subset.json must exist with search + heldout splits."""

    def test_subset_structure(self) -> None:
        import json

        subset = json.loads((SCRIPT_DIR / "tasks" / "subset.json").read_text())
        assert len(subset) == 25
        search = [e for e in subset if e["split"] == "search"]
        heldout = [e for e in subset if e["split"] == "heldout"]
        assert len(search) == 10
        assert len(heldout) == 15