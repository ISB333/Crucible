import pytest

from crucible.artifact import Artifact, Region, RegionError

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


def test_parses_one_region() -> None:
    a = make()
    assert a.regions == (Region(file="problem.py", name="solution", start=4, end=6),)


def test_region_text_returns_editable_lines_only() -> None:
    a = make()
    text = a.region_text(a.regions[0])
    assert text == "def solve() -> int:\n    raise NotImplementedError"


def test_replace_region_changes_body_and_links_parent() -> None:
    a = make()
    b = a.replace_region("solution", "def solve() -> int:\n    return 42")
    assert "return 42" in b.files["problem.py"]
    assert "NotImplementedError" not in b.files["problem.py"]
    assert b.parent_hash == a.content_hash
    assert b.content_hash != a.content_hash
    # markers survived the splice
    assert b.regions[0].name == "solution"


def test_content_hash_is_deterministic() -> None:
    assert make().content_hash == make().content_hash


def test_unclosed_region_raises() -> None:
    with pytest.raises(RegionError):
        Artifact.from_files({"f.py": "# crucible:region start name=x\n"})


def test_nested_region_raises() -> None:
    bad = "# crucible:region start name=a\n# crucible:region start name=b\n"
    with pytest.raises(RegionError):
        Artifact.from_files({"f.py": bad})


def test_unknown_region_name_raises() -> None:
    with pytest.raises(KeyError):
        make().replace_region("nope", "x")


def test_crlf_input_is_normalized_to_lf() -> None:
    crlf = "x = 1\r\n# crucible:region start name=s\r\ny = 2\r\n# crucible:region end\r\n"
    a = Artifact.from_files({"f.py": crlf})
    assert "\r" not in a.files["f.py"]
    b = a.replace_region("s", "y = 3")
    # bytes outside the region are untouched by the edit
    assert b.files["f.py"].splitlines(keepends=True)[0] == "x = 1\n"
