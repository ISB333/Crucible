"""Verify the 25-task subset is correctly materialized.

Locks the subset: 25 entries (10 search + 15 heldout), each task dir has
spec.md (non-empty, contains function signature) and skeleton.py ("pass").
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

SUBSET_DIR = Path(__file__).resolve().parent.parent.parent.parent / "examples" / "agentic_harness" / "tasks"
SUBSET_JSON = SUBSET_DIR / "subset.json"


def _load_subset() -> list[dict]:
    assert SUBSET_JSON.exists(), f"subset.json not found at {SUBSET_JSON}"
    return json.loads(SUBSET_JSON.read_text())


class TestSubsetJson:
    def test_has_25_entries(self) -> None:
        subset = _load_subset()
        assert len(subset) == 25

    def test_10_search_15_heldout(self) -> None:
        subset = _load_subset()
        search = [e for e in subset if e["split"] == "search"]
        heldout = [e for e in subset if e["split"] == "heldout"]
        assert len(search) == 10
        assert len(heldout) == 15

    def test_unique_task_ids(self) -> None:
        subset = _load_subset()
        ids = [e["task_id"] for e in subset]
        assert len(ids) == len(set(ids)), "duplicate task_ids in subset"


class TestTaskDirs:
    @pytest.fixture(params=_load_subset())
    def task_entry(self, request) -> dict:
        return request.param

    def test_spec_md_exists_and_nonempty(self, task_entry: dict) -> None:
        spec_path = SUBSET_DIR / task_entry["task_id"] / "spec.md"
        assert spec_path.exists(), f"missing {spec_path}"
        content = spec_path.read_text()
        assert len(content) > 0, f"empty spec.md for {task_entry['task_id']}"
        # spec.md = code_prompt which contains the function signature
        assert "def " in content, f"spec.md for {task_entry['task_id']} missing function signature"

    def test_skeleton_py_is_pass(self, task_entry: dict) -> None:
        skel_path = SUBSET_DIR / task_entry["task_id"] / "skeleton.py"
        assert skel_path.exists(), f"missing {skel_path}"
        content = skel_path.read_text()
        # skeleton.py should be exactly "pass\n" — the placeholder body
        assert content.strip() == "pass", (
            f"skeleton.py for {task_entry['task_id']} should be 'pass', "
            f"got: {content!r}"
        )

    def test_skeleton_not_code_prompt(self, task_entry: dict) -> None:
        """Contract: skeleton.py must NOT contain code_prompt content (no 'def ')."""
        skel_path = SUBSET_DIR / task_entry["task_id"] / "skeleton.py"
        content = skel_path.read_text()
        assert "def " not in content, (
            f"skeleton.py for {task_entry['task_id']} contains 'def ' — "
            f"this would duplicate the signature when check_solution prepends code_prompt"
        )