import pytest

from crucible.artifact import Artifact, masked_view, scan_holes

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


def test_masked_view_equal_when_only_region_changed() -> None:
    a = make()
    b = a.replace_region("solution", "def solve() -> int:\n    return 42")
    assert masked_view(a) == masked_view(b)


def test_masked_view_differs_when_immutable_part_changed() -> None:
    tampered = PROBLEM.replace("assert spec() == 42", "assert True")
    b = Artifact.from_files({"problem.py": tampered})
    assert masked_view(make()) != masked_view(b)


def test_masked_view_detects_added_file() -> None:
    b = Artifact.from_files({"problem.py": PROBLEM, "evil.py": "x = 1\n"})
    assert masked_view(make()) != masked_view(b)


def test_scan_holes_finds_not_implemented() -> None:
    holes = scan_holes(make())
    assert len(holes) == 1
    assert holes[0].kind == "not_implemented"
    assert holes[0].file == "problem.py"
    assert holes[0].line == 5


def test_scan_holes_finds_sentinel() -> None:
    a = Artifact.from_files({"f.py": "# crucible:hole compute the bound\n"})
    holes = scan_holes(a)
    assert len(holes) == 1
    assert holes[0].kind == "sentinel"


def test_scan_holes_empty_when_solved() -> None:
    a = make().replace_region("solution", "def solve() -> int:\n    return 42")
    assert scan_holes(a) == ()
