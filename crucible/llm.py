"""Provider-agnostic LLM shim (PRD §10): the engine sees only LLMSession."""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", dict(self.args))  # defensive copy


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    content: str


@runtime_checkable
class LLMSession(Protocol):
    def start(self, system: str, user: str) -> list[ToolCall]: ...
    def reply(self, results: Sequence[ToolResult]) -> list[ToolCall]: ...
    @property
    def cost_usd(self) -> float: ...  # 0.0 when unknown (e.g. local models)
    @property
    def messages(self) -> list[dict]: ...  # Returns the full conversation history with reasoning


# Anthropic-style schemas; providers translate as needed (see providers.py).
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search_replace",
        "description": (
            "Replace ONE unique occurrence of `old` with `new` inside an editable region"
            " of `file`. Rejected if `old` is not found, matches more than once, or lies"
            " outside an editable region. After each applied edit you receive the"
            " verifier's verdict on the new artifact."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file": {"type": "string"},
                "old": {"type": "string"},
                "new": {"type": "string"},
            },
            "required": ["file", "old", "new"],
        },
    },
    {
        "name": "write_region",
        "description": "Replace the entire body of the named editable region with `content`.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "content": {"type": "string"}},
            "required": ["name", "content"],
        },
    },
    {
        "name": "record_lesson",
        "description": (
            "Record a short lesson learned this episode. Lessons are written into the"
            " artifact at episode end so the next episode starts wiser."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
]


@dataclass
class ScriptedSession:
    """Deterministic test double: pops one batch of tool calls per model turn."""

    script: Sequence[Sequence[ToolCall]]
    received: list[list[ToolResult]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._queue: deque[Sequence[ToolCall]] = deque(self.script)
        self.prompt: tuple[str, str] | None = None
        self._messages: list[dict] = []

    def start(self, system: str, user: str) -> list[ToolCall]:
        self.prompt = (system, user)
        self._messages.append({"role": "system", "content": system})
        self._messages.append({"role": "user", "content": user})
        return list(self._queue.popleft()) if self._queue else []

    def reply(self, results: Sequence[ToolResult]) -> list[ToolCall]:
        for r in results:
            self._messages.append({"role": "tool", "tool_call_id": r.call_id, "content": r.content})
        self.received.append(list(results))
        calls = list(self._queue.popleft()) if self._queue else []
        if calls:
            self._messages.append({"role": "assistant", "content": f"Tool calls: {[c.name for c in calls]}"})
        return calls

    @property
    def cost_usd(self) -> float:
        return 0.0

    @property
    def messages(self) -> list[dict]:
        """Returns the full conversation history with reasoning."""
        return self._messages
