import pytest

from crucible.artifact import Artifact
from crucible.budgets import EpisodeBudget
from crucible.integrity import Composite, DenyTokens, ImmutableRegions
from crucible.llm import ScriptedSession, ToolCall
from crucible.worker import run_episode
from tests.unit.conftest import StubVerifier

pytestmark = pytest.mark.unit

PROBLEM = """\
def spec() -> int:
    return 42

# crucible:region start name=solution
def solve() -> int:
    raise NotImplementedError
# crucible:region end
"""

SOLUTION = "def solve() -> int:\n    return 42"


def setup(make_ctx):
    a = Artifact.from_files({"problem.py": PROBLEM})
    integrity = Composite(checks=(ImmutableRegions.freeze(a), DenyTokens()))
    return a, StubVerifier(), make_ctx(), integrity


def wr(content: str, call_id: str = "1") -> ToolCall:
    return ToolCall(id=call_id, name="write_region", args={"name": "solution", "content": content})


def test_solving_episode(make_ctx) -> None:
    a, verifier, ctx, integrity = setup(make_ctx)
    session = ScriptedSession([[wr(SOLUTION)]])
    out = run_episode(a, verifier, ctx, session, EpisodeBudget(), integrity)
    assert out.solved and out.end_reason == "solved"
    assert "return 42" in out.artifact.files["problem.py"]
    assert out.edits == 1


def test_rejected_edit_feeds_error_back(make_ctx) -> None:
    a, verifier, ctx, integrity = setup(make_ctx)
    bad = ToolCall(
        id="1", name="search_replace", args={"file": "problem.py", "old": "return 42", "new": "x"}
    )
    session = ScriptedSession([[bad]])
    out = run_episode(a, verifier, ctx, session, EpisodeBudget(), integrity)
    assert not out.solved
    assert session.received[0][0].content.startswith("EDIT REJECTED")
    assert out.artifact.content_hash == a.content_hash  # nothing applied


def test_deny_token_triggers_revert(make_ctx) -> None:
    a, verifier, ctx, integrity = setup(make_ctx)
    session = ScriptedSession([[wr("import pytest\npytest.skip('gamed')")]])
    out = run_episode(a, verifier, ctx, session, EpisodeBudget(), integrity)
    assert not out.solved
    assert out.end_reason == "integrity_violation"
    assert out.artifact.content_hash == a.content_hash  # reverted to episode start


def test_verifies_after_every_edit(make_ctx) -> None:
    a, verifier, ctx, integrity = setup(make_ctx)
    calls = [
        [wr("def solve() -> int:\n    return 41", "1")],
        [wr(SOLUTION, "2")],
    ]
    session = ScriptedSession(calls)
    run_episode(a, verifier, ctx, session, EpisodeBudget(), integrity)
    # initial verify + one per applied edit (2) + final verify = 4
    assert verifier.calls == 4


def test_record_lesson_collected(make_ctx) -> None:
    a, verifier, ctx, integrity = setup(make_ctx)
    lesson = ToolCall(id="1", name="record_lesson", args={"text": "42 is the answer"})
    session = ScriptedSession([[lesson]])
    out = run_episode(a, verifier, ctx, session, EpisodeBudget(), integrity)
    assert out.lessons == "42 is the answer"


def test_turn_budget_ends_episode(make_ctx) -> None:
    a, verifier, ctx, integrity = setup(make_ctx)
    noop = ToolCall(id="1", name="record_lesson", args={"text": "spin"})
    session = ScriptedSession([[noop]] * 50)
    out = run_episode(a, verifier, ctx, session, EpisodeBudget(turns=3), integrity)
    assert out.end_reason == "turn_budget" and out.turns == 3


def test_advisory_verifier_never_solves(make_ctx) -> None:
    a, _, ctx, integrity = setup(make_ctx)
    advisory = StubVerifier(deterministic=False)
    session = ScriptedSession([[wr(SOLUTION)]])
    out = run_episode(a, advisory, ctx, session, EpisodeBudget(), integrity)
    assert not out.solved  # PRD §3: advisory cannot be the sole accept signal


def test_unknown_tool_is_reported_not_fatal(make_ctx) -> None:
    a, verifier, ctx, integrity = setup(make_ctx)
    weird = ToolCall(id="1", name="rm_rf", args={})
    session = ScriptedSession([[weird]])
    out = run_episode(a, verifier, ctx, session, EpisodeBudget(), integrity)
    assert "unknown tool" in session.received[0][0].content
    assert not out.solved


def test_edit_budget_ends_episode(make_ctx) -> None:
    a, _, ctx, integrity = setup(make_ctx)
    # Use advisory (non-deterministic) verifier so episode doesn't solve before budget exhausted
    verifier = StubVerifier(deterministic=False)
    edits = [[wr(f"def solve() -> int:\n    return {i}", str(i))] for i in range(10)]
    session = ScriptedSession(edits)
    out = run_episode(a, verifier, ctx, session, EpisodeBudget(edits=2), integrity)
    assert out.end_reason == "edit_budget" and out.edits == 2
