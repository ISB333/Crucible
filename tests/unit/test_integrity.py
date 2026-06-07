import pytest

from crucible.artifact import Artifact
from crucible.integrity import Composite, DenyTokens, ImmutableRegions, IntegrityCheck

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


def test_region_only_change_passes_byte_check(make_ctx) -> None:
    original = make()
    gate = ImmutableRegions.freeze(original)
    edited = original.replace_region("solution", "def solve() -> int:\n    return 42")
    assert gate.check(edited, make_ctx())


def test_tampered_spec_fails_byte_check(make_ctx) -> None:
    gate = ImmutableRegions.freeze(make())
    tampered = Artifact.from_files({"problem.py": PROBLEM.replace("== 42", "== solve()")})
    assert not gate.check(tampered, make_ctx())


def test_added_file_fails_byte_check(make_ctx) -> None:
    gate = ImmutableRegions.freeze(make())
    grown = Artifact.from_files({"problem.py": PROBLEM, "conftest.py": "collect_ignore = ['*']\n"})
    assert not gate.check(grown, make_ctx())


def test_pytest_skip_in_region_fails_deny_list(make_ctx) -> None:
    a = make().replace_region("solution", "import pytest\npytest.skip('nope')")
    deny = DenyTokens()
    assert not deny.check(a, make_ctx())
    assert "pytest.skip" in deny.violations(a)[0]


def test_sorry_in_lean_region_fails_deny_list(make_ctx) -> None:
    lean = "-- crucible:region start name=proof\nsorry\n-- crucible:region end\n"
    a = Artifact.from_files({"toy.lean": lean})
    assert not DenyTokens().check(a, make_ctx())


def test_clean_solution_passes_deny_list(make_ctx) -> None:
    a = make().replace_region("solution", "def solve() -> int:\n    return 42")
    assert DenyTokens().check(a, make_ctx())


def test_extra_tokens_are_enforced(make_ctx) -> None:
    a = make().replace_region("solution", "def solve() -> int:\n    return eval('42')")
    assert DenyTokens().check(a, make_ctx())
    assert not DenyTokens(extra=(r"\beval\b",)).check(a, make_ctx())


def test_composite_requires_all(make_ctx) -> None:
    original = make()
    gate = Composite(checks=(ImmutableRegions.freeze(original), DenyTokens()))
    good = original.replace_region("solution", "def solve() -> int:\n    return 42")
    bad = original.replace_region("solution", "import pytest\npytest.skip('x')")
    assert gate.check(good, make_ctx())
    assert not gate.check(bad, make_ctx())
    assert isinstance(gate, IntegrityCheck)
