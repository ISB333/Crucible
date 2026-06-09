import pytest

from crucible.artifact import Artifact
from crucible.edits import search_replace, write_region

pytestmark = pytest.mark.unit

PROBLEM = """\
def spec() -> int:
    return 42

# crucible:region start name=solution
def solve() -> int:
    raise NotImplementedError
# crucible:region end

assert spec() == 42
"""


def make() -> Artifact:
    return Artifact.from_files({"problem.py": PROBLEM})


def test_edit_inside_region_applies_and_links_parent() -> None:
    a = make()
    r = search_replace(a, "problem.py", "raise NotImplementedError", "return 42")
    assert r.applied
    assert "return 42" in r.artifact.files["problem.py"]
    assert r.artifact.parent_hash == a.content_hash
    # original untouched (immutability)
    assert "NotImplementedError" in a.files["problem.py"]


def test_edit_outside_region_rejected() -> None:
    r = search_replace(make(), "problem.py", "return 42\n\n#", "return 41\n\n#")
    assert not r.applied
    assert r.error is not None and "outside editable regions" in r.error


def test_edit_spanning_region_boundary_rejected() -> None:
    r = search_replace(
        make(), "problem.py", "raise NotImplementedError\n# crucible", "pass\n# crucible"
    )
    assert not r.applied  # touches the marker token


def test_ambiguous_old_text_rejected() -> None:
    a = Artifact.from_files(
        {"f.py": "# crucible:region start name=s\nx = 1\nx = 1\n# crucible:region end\n"}
    )
    r = search_replace(a, "f.py", "x = 1", "x = 2")
    assert not r.applied
    assert r.error is not None and "2 times" in r.error


def test_old_text_not_found_rejected() -> None:
    r = search_replace(make(), "problem.py", "no such text", "x")
    assert not r.applied
    assert r.error is not None and "not found" in r.error


def test_unknown_file_rejected() -> None:
    r = search_replace(make(), "nope.py", "a", "b")
    assert not r.applied


def test_marker_injection_rejected() -> None:
    r = search_replace(
        make(),
        "problem.py",
        "raise NotImplementedError",
        "pass\n# crucible:region end\nimport os\n# crucible:region start name=solution",
    )
    assert not r.applied
    assert r.error is not None and "marker" in r.error


def test_write_region_replaces_whole_body() -> None:
    r = write_region(make(), "solution", "def solve() -> int:\n    return 42")
    assert r.applied
    assert "NotImplementedError" not in r.artifact.files["problem.py"]


def test_write_region_unknown_name_rejected() -> None:
    r = write_region(make(), "nope", "x")
    assert not r.applied


def test_write_region_strips_wrapping_code_fence() -> None:
    fenced = "```python\ndef solve() -> int:\n    return 42\n```"
    r = write_region(make(), "solution", fenced)
    assert r.applied
    body = r.artifact.files["problem.py"]
    assert "```" not in body
    assert "def solve() -> int:\n    return 42" in body


def test_write_region_leaves_unfenced_code_untouched() -> None:
    code = "def solve() -> int:\n    return 42  # no fence"
    r = write_region(make(), "solution", code)
    assert r.applied
    assert "return 42  # no fence" in r.artifact.files["problem.py"]
