from types import SimpleNamespace

import pytest

from crucible.providers import (
    extract_anthropic_calls,
    extract_openai_calls,
    extract_text_tool_calls,
    to_openai_tools,
)

pytestmark = pytest.mark.unit


def test_extract_anthropic_calls_filters_tool_use_blocks() -> None:
    blocks = [
        SimpleNamespace(type="text", text="thinking…"),
        SimpleNamespace(
            type="tool_use", id="tu_1", name="write_region", input={"name": "s", "content": "x"}
        ),
    ]
    calls = extract_anthropic_calls(blocks)
    assert len(calls) == 1
    assert calls[0].id == "tu_1" and calls[0].name == "write_region"
    assert calls[0].args == {"name": "s", "content": "x"}


def test_extract_openai_calls_parses_json_arguments() -> None:
    tc = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(
            name="search_replace", arguments='{"file": "f.py", "old": "a", "new": "b"}'
        ),
    )
    msg = SimpleNamespace(tool_calls=[tc])
    calls = extract_openai_calls(msg)
    assert calls[0].args == {"file": "f.py", "old": "a", "new": "b"}


def test_extract_openai_calls_handles_no_tool_calls() -> None:
    assert extract_openai_calls(SimpleNamespace(tool_calls=None)) == []


def test_extract_openai_calls_rejects_malformed_json() -> None:
    tc = SimpleNamespace(
        id="c", function=SimpleNamespace(name="write_region", arguments="{not json")
    )
    assert extract_openai_calls(SimpleNamespace(tool_calls=[tc])) == []


def test_to_openai_tools_wraps_schemas() -> None:
    tools = to_openai_tools()
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "search_replace"
    assert "parameters" in tools[0]["function"]


def test_to_openai_tools_includes_extra() -> None:
    from crucible.llm import CONSULT_ADVISOR_SCHEMA, TOOL_SCHEMAS

    base = to_openai_tools()
    assert len(base) == len(TOOL_SCHEMAS)
    extended = to_openai_tools(TOOL_SCHEMAS + [CONSULT_ADVISOR_SCHEMA])
    names = [t["function"]["name"] for t in extended]
    assert "consult_advisor" in names
    assert len(extended) == len(TOOL_SCHEMAS) + 1


def test_extract_text_tool_calls_parses_fenced_json() -> None:
    content = (
        "Sure, here is the call:\n```json\n"
        '{"name": "write_region", "arguments": {"name": "solution", "content": "[1, 2, 5]"}}\n'
        "```"
    )
    calls = extract_text_tool_calls(content)
    assert len(calls) == 1
    assert calls[0].name == "write_region"
    assert calls[0].args == {"name": "solution", "content": "[1, 2, 5]"}


def test_extract_text_tool_calls_parses_bare_json() -> None:
    calls = extract_text_tool_calls('{"name": "record_lesson", "arguments": {"text": "hi"}}')
    assert len(calls) == 1 and calls[0].name == "record_lesson" and calls[0].args == {"text": "hi"}


def test_extract_text_tool_calls_handles_string_arguments_and_parameters_alias() -> None:
    calls = extract_text_tool_calls('{"name": "write_region", "parameters": "{\\"name\\": \\"s\\"}"}')
    assert len(calls) == 1 and calls[0].args == {"name": "s"}


def test_extract_text_tool_calls_ignores_plain_prose() -> None:
    assert extract_text_tool_calls("I think we should use a greedy algorithm here.") == []
    assert extract_text_tool_calls('{"foo": 1}') == []  # no name field


def test_extract_text_tool_calls_deterministic_id() -> None:
    content = '{"name": "write_region", "arguments": {"name": "solution", "content": "x"}}'
    assert extract_text_tool_calls(content)[0].id == extract_text_tool_calls(content)[0].id


def test_consult_advisor_schema_shape() -> None:
    from crucible.llm import CONSULT_ADVISOR_SCHEMA

    assert CONSULT_ADVISOR_SCHEMA["name"] == "consult_advisor"
    assert "question" in CONSULT_ADVISOR_SCHEMA["input_schema"]["properties"]
    assert CONSULT_ADVISOR_SCHEMA["input_schema"]["required"] == ["question"]
