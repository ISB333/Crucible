import pytest

from crucible.llm import TOOL_SCHEMAS, LLMSession, ScriptedSession, ToolCall, ToolResult

pytestmark = pytest.mark.unit


def test_tool_schemas_cover_the_three_tools() -> None:
    assert [t["name"] for t in TOOL_SCHEMAS] == ["search_replace", "write_region", "record_lesson"]
    for t in TOOL_SCHEMAS:
        assert t["description"]
        assert t["input_schema"]["type"] == "object"


def test_scripted_session_pops_batches_in_order() -> None:
    c1 = ToolCall(id="1", name="write_region", args={"name": "s", "content": "x = 2"})
    c2 = ToolCall(id="2", name="record_lesson", args={"text": "tried x=2"})
    s = ScriptedSession([[c1], [c2]])
    assert s.start("sys", "user") == [c1]
    assert s.reply([ToolResult(call_id="1", content="VERDICT: FAIL")]) == [c2]
    assert s.reply([ToolResult(call_id="2", content="lesson recorded")]) == []  # exhausted
    assert s.received[0][0].content == "VERDICT: FAIL"
    assert s.cost_usd == 0.0


def test_scripted_session_satisfies_protocol() -> None:
    assert isinstance(ScriptedSession([]), LLMSession)


def test_toolcall_args_are_defensively_copied() -> None:
    src = {"name": "s", "content": "x"}
    call = ToolCall(id="1", name="write_region", args=src)
    src["injected"] = "evil"
    assert "injected" not in call.args
