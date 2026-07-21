"""Tests for the FROZEN agent contract primitives."""
from __future__ import annotations

import json
from pathlib import Path

from agent_contract import Task, LLM, Tools, solve, load_subset  # type: ignore[import-not-found]


def _make_fake_client(capture: dict | list, key: str = "messages"):
    """Build a FakeClient that mirrors openai.OpenAI's client.chat.completions.create path."""
    class FakeCompletions:
        @staticmethod
        def create(*, model, messages, max_tokens, temperature, extra_body):
            if isinstance(capture, list):
                capture.append(messages)
            else:
                capture.update(model=model, max_tokens=max_tokens,
                               temperature=temperature, extra_body=extra_body)
            class Msg:
                content = "ok"
            class Choice:
                message = Msg()
            return type("R", (), {"choices": [Choice()]})()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    return FakeClient()


def test_llm_chat_is_openai_compat(tmp_path):
    """LLM.chat delegates to an OpenAI-compatible client and returns content."""
    calls: list[list[dict]] = []
    llm = LLM(base_url="http://x", model="tess")
    object.__setattr__(llm, "_client", _make_fake_client(calls))
    result = llm.chat([{"role": "user", "content": "hi"}])
    assert result == "ok"
    assert calls == [[{"role": "user", "content": "hi"}]]


def test_llm_chat_passes_params():
    """LLM.chat forwards max_tokens, temperature, and enable_thinking=False."""
    captured: dict = {}
    llm = LLM(base_url="http://x", model="tess")
    object.__setattr__(llm, "_client", _make_fake_client(captured))
    llm.chat([{"role": "user", "content": "hi"}],
             max_tokens=128, temperature=0.0)
    assert captured["model"] == "tess"
    assert captured["max_tokens"] == 128
    assert captured["temperature"] == 0.0
    assert captured["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_solve_is_not_implemented_stub():
    """solve() must raise NotImplementedError — the editable stub."""
    try:
        solve(
            Task(id="t", spec="s", skeleton_path=Path("s"), eval_task_id="t"),
            Path("."),
            LLM("http://x", "tess"),
            Tools(Path(".")),
        )
        assert False, "solve should raise NotImplementedError"
    except NotImplementedError:
        pass


def test_load_subset(tmp_path):
    """load_subset reads subset.json and builds Task objects."""
    (tmp_path / "BigCodeBench" / "0").mkdir(parents=True)
    (tmp_path / "BigCodeBench" / "1").mkdir(parents=True)
    (tmp_path / "BigCodeBench" / "0" / "spec.md").write_text("spec0")
    (tmp_path / "BigCodeBench" / "1" / "spec.md").write_text("spec1")

    (tmp_path / "subset.json").write_text(json.dumps([
        {"task_id": "BigCodeBench/0", "split": "search"},
        {"task_id": "BigCodeBench/1", "split": "heldout"},
    ]))

    subs = load_subset(tmp_path / "subset.json")
    assert len(subs) == 2
    assert [t.eval_task_id for t in subs] == ["BigCodeBench/0", "BigCodeBench/1"]
    assert subs[0].id == "search/BigCodeBench/0"
    assert subs[1].id == "heldout/BigCodeBench/1"
    assert subs[0].spec == "spec0"
    assert subs[0].skeleton_path == Path("BigCodeBench/0") / "skeleton.py"


def test_tools_read_write_list(tmp_path):
    """Tools read/write/list round-trip in the workdir."""
    tools = Tools(tmp_path)
    tools.write_file("hello.py", "print('hi')")
    assert tools.read_file("hello.py") == "print('hi')"
    assert "hello.py" in tools.list_dir()


def test_tools_run_visible_tests(tmp_path):
    """Tools.run_visible_tests executes a test file and returns stdout."""
    (tmp_path / "test_x.py").write_text("print('all tests passed')")
    tools = Tools(tmp_path)
    out = tools.run_visible_tests("test_x.py")
    assert "all tests passed" in out


def test_task_is_frozen():
    """Task dataclass must be frozen — mutation raises FrozenInstanceError."""
    t = Task(id="t", spec="s", skeleton_path=Path("s"), eval_task_id="t")
    try:
        t.id = "changed"  # type: ignore[misc]
        assert False, "Task should be frozen"
    except AttributeError:
        pass


def test_llm_is_frozen():
    """LLM dataclass must be frozen — mutation raises FrozenInstanceError."""
    llm = LLM(base_url="http://x", model="tess")
    try:
        llm.model = "changed"  # type: ignore[misc]
        assert False, "LLM should be frozen"
    except AttributeError:
        pass