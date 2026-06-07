from types import SimpleNamespace

import pytest

from crucible.providers import extract_anthropic_calls, extract_openai_calls, to_openai_tools

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
