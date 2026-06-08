"""Provider sessions. SDKs are optional extras, imported lazily inside the classes.

Secrets come from the environment only (ANTHROPIC_API_KEY / OPENAI_API_KEY /
OPENAI_BASE_URL) — never from code or config files.
"""

import json
import os
from collections.abc import Iterable, Sequence
from typing import Any

from crucible.llm import TOOL_SCHEMAS, LLMSession, ToolCall, ToolResult

# USD per million tokens (input, output); unknown models cost 0.0 (= no USD-cap signal).
# Prices verified 2026-06-06 (Opus 4.6+: 5/25; Sonnet 4.6: 3/15; Haiku 4.5: 1/5).
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus": (5.0, 25.0),
    "claude-sonnet": (3.0, 15.0),
    "claude-haiku": (1.0, 5.0),
    "gemini-flash": (0.35, 1.05),
    "gemini-pro": (1.25, 5.0),
}


def price_for(model: str) -> tuple[float, float]:
    for prefix, price in PRICES_PER_MTOK.items():
        if model.startswith(prefix):
            return price
    return (0.0, 0.0)


def extract_anthropic_calls(content_blocks: Iterable[Any]) -> list[ToolCall]:
    return [
        ToolCall(id=b.id, name=b.name, args=dict(b.input))
        for b in content_blocks
        if getattr(b, "type", None) == "tool_use"
    ]


def extract_openai_calls(message: Any) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for tc in message.tool_calls or []:
        try:
            args = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, TypeError):
            continue  # validate model output before acting on it
        if isinstance(args, dict):
            calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args))
    return calls


def to_openai_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in TOOL_SCHEMAS
    ]


class AnthropicSession:
    def __init__(self, model: str, max_tokens: int = 4096) -> None:
        import anthropic  # optional extra: crucible[anthropic]

        self._client = anthropic.Anthropic()  # key from ANTHROPIC_API_KEY
        self._model_name = model
        self._model = model
        self._max_tokens = max_tokens
        self._system = ""
        self._messages: list[dict[str, Any]] = []
        self._in_tokens = 0
        self._out_tokens = 0

    def start(self, system: str, user: str) -> list[ToolCall]:
        self._system = system
        self._messages = [{"role": "user", "content": user}]
        return self._step()

    def reply(self, results: Sequence[ToolResult]) -> list[ToolCall]:
        self._messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": r.call_id, "content": r.content}
                    for r in results
                ],
            }
        )
        return self._step()

    def _step(self) -> list[ToolCall]:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system,
            messages=self._messages,  # type: ignore[arg-type]
            tools=TOOL_SCHEMAS,  # type: ignore[arg-type]
        )
        self._messages.append({"role": "assistant", "content": resp.content})
        self._in_tokens += resp.usage.input_tokens
        self._out_tokens += resp.usage.output_tokens
        return extract_anthropic_calls(resp.content)

    @property
    def cost_usd(self) -> float:
        pin, pout = price_for(self._model_name)
        return (self._in_tokens * pin + self._out_tokens * pout) / 1_000_000

    @property
    def messages(self) -> list[dict]:
        """Returns the full conversation history with reasoning."""
        return self._messages


class OpenAICompatSession:
    """OpenAI or any OpenAI-compatible endpoint (set OPENAI_BASE_URL for local models)."""

    def __init__(self, model: str, base_url: str | None = None) -> None:
        from openai import OpenAI  # optional extra: crucible[openai]

        self._client = OpenAI(base_url=base_url or os.environ.get("OPENAI_BASE_URL"))
        self._model_name = model
        self._model = model
        self._messages: list[dict[str, Any]] = []
        self._in_tokens = 0
        self._out_tokens = 0

    def start(self, system: str, user: str) -> list[ToolCall]:
        self._messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self._step()

    def reply(self, results: Sequence[ToolResult]) -> list[ToolCall]:
        for r in results:
            self._messages.append({"role": "tool", "tool_call_id": r.call_id, "content": r.content})
        return self._step()

    def _step(self) -> list[ToolCall]:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=self._messages,  # type: ignore[arg-type]
            tools=to_openai_tools(),  # type: ignore[arg-type]
        )
        msg = resp.choices[0].message
        self._messages.append(msg.model_dump(exclude_none=True))
        if resp.usage is not None:
            self._in_tokens += resp.usage.prompt_tokens
            self._out_tokens += resp.usage.completion_tokens
        return extract_openai_calls(msg)

    @property
    def cost_usd(self) -> float:
        pin, pout = price_for(self._model_name)
        return (self._in_tokens * pin + self._out_tokens * pout) / 1_000_000

    @property
    def messages(self) -> list[dict]:
        """Returns the full conversation history with reasoning."""
        return self._messages


class GeminiSession:
    """Google Gemini API session (requires GOOGLE_API_KEY or GEMINI_API_KEY)."""

    def __init__(self, model: str, max_tokens: int = 4096) -> None:
        import google.genai as genai  # optional extra: crucible[gemini]
        import google.genai.types as types  # type: ignore[reportMissingImports]
        # type: ignore[reportMissingImports]

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY must be set in environment")
        self._client = genai.Client(api_key=api_key)  # type: ignore[reportPrivateImportUsage]
        self._model_name = model
        self._model = model
        self._max_tokens = max_tokens
        self._system = ""
        self._history: list[dict[str, Any]] = []
        self._in_tokens = 0
        self._out_tokens = 0
        self._types = types
        self._call_id_to_name: dict[str, str] = {}

    def start(self, system: str, user: str) -> list[ToolCall]:
        self._system = system
        self._history = [{"role": "user", "parts": [user]}]
        return self._step()

    def reply(self, results: Sequence[ToolResult]) -> list[ToolCall]:
        # Create function response parts for each tool result
        parts = []
        for r in results:
            # Get the function name from the call ID
            function_name = self._call_id_to_name.get(r.call_id, "unknown")
            # Create a FunctionResponse object with the tool result
            function_response = self._types.FunctionResponse(
                id=r.call_id,
                name=function_name,
                response={"output": r.content},
            )
            # Wrap the FunctionResponse in a Part object
            part = self._types.Part(function_response=function_response)
            parts.append(part)
        self._history.append({"role": "model", "parts": parts})
        return self._step()

    def _step(self) -> list[ToolCall]:
        # Convert conversation history for Gemini
        gemini_history = []
        for msg in self._history:
            parts = []
            for part in msg["parts"]:
                if isinstance(part, str):
                    # Create a Part object with text content
                    parts.append(self._types.Part(text=part))
                else:
                    # If part is already a Part object, use it as is
                    parts.append(part)
            content = self._types.Content(
                role=msg["role"],
                parts=parts,
            )
            gemini_history.append(content)

        # Convert TOOL_SCHEMAS to Gemini format (input_schema -> parameters_json_schema)
        function_declarations = []
        for tool in TOOL_SCHEMAS:
            fd = self._types.FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters_json_schema=tool["input_schema"],
            )
            function_declarations.append(fd)

        # Call Gemini API
        response = self._client.models.generate_content(
            model=self._model,
            contents=gemini_history,
            config={
                "max_output_tokens": self._max_tokens,
                "tool_config": {
                    "function_calling_config": {
                        "mode": "auto",
                    },
                },
                "tools": [
                    self._types.Tool(function_declarations=function_declarations)
                ],
            },
        )

        # Extract tool calls
        calls = []
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content:
                for part in candidate.content.parts:
                    if part.function_call:
                        call = ToolCall(
                            id=part.function_call.id,
                            name=part.function_call.name,
                            args=part.function_call.args,
                        )
                        calls.append(call)
                        # Track call ID to function name for later responses
                        self._call_id_to_name[call.id] = call.name

        # Append model's response to history
        model_parts = []
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content:
                for part in candidate.content.parts:
                    model_parts.append(part)
        self._history.append({"role": "model", "parts": model_parts})
        return calls

    @property
    def cost_usd(self) -> float:
        pin, pout = price_for(self._model_name)
        return (self._in_tokens * pin + self._out_tokens * pout) / 1_000_000

    @property
    def messages(self) -> list[dict]:
        """Returns the full conversation history with reasoning."""
        # Serialize Google API objects to plain dicts for JSON serialization
        serialized = []
        for msg in self._history:
            serialized_msg = {"role": msg["role"]}
            parts = []
            for part in msg["parts"]:
                if hasattr(part, "text"):
                    if part.text:
                        parts.append({"text": part.text})
                elif hasattr(part, "function_call"):
                    fc = part.function_call
                    parts.append({"function_call": {"name": fc.name, "args": fc.args, "id": fc.id}})
                elif hasattr(part, "function_response"):
                    fr = part.function_response
                    parts.append({"function_response": {"name": fr.name, "response": fr.response, "id": fr.id}})
                elif isinstance(part, dict):
                    # Filter out None values
                    filtered_part = {k: v for k, v in part.items() if v is not None}
                    if filtered_part:
                        parts.append(filtered_part)
                elif part is not None:
                    parts.append(str(part))
            serialized_msg["parts"] = parts
            serialized.append(serialized_msg)
        return serialized


def make_session(model: str, base_url: str | None = None) -> LLMSession:
    if model.startswith("claude"):
        return AnthropicSession(model)
    if model.startswith("gemini"):
        return GeminiSession(model)
    return OpenAICompatSession(model, base_url=base_url)
