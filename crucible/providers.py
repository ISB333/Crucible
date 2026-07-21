"""Provider sessions. SDKs are optional extras, imported lazily inside the classes.

Secrets come from the environment only (ANTHROPIC_API_KEY / OPENAI_API_KEY /
OPENAI_BASE_URL) — never from code or config files.
"""

import hashlib
import json
import os
import re
import time
from collections.abc import Iterable, Sequence
from typing import Any, Protocol, runtime_checkable

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


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def extract_text_tool_calls(content: str) -> list[ToolCall]:
    """Fallback for local models (common behind Ollama) that emit a tool call as JSON
    text instead of a native ``tool_calls`` entry. Only used when the provider returned
    no structured calls. Recognizes ``{"name", "arguments"|"parameters"}`` optionally
    wrapped in a ```json fence; returns [] for plain prose so well-behaved models are
    unaffected. Validate model output before acting on it (Ananke #10)."""
    if not content or '"name"' not in content:
        return []
    text = content.strip()
    if text.startswith("```"):  # unwrap a ```json ... ``` fence
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    candidates = [text]
    m = _JSON_OBJECT.search(text)
    if m and m.group(0) != text:
        candidates.append(m.group(0))
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {}
        if isinstance(name, str) and isinstance(args, dict):
            cid = "text-" + hashlib.sha1(cand.encode()).hexdigest()[:8]
            return [ToolCall(id=cid, name=name, args=args)]
    return []


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


def to_openai_tools(tools: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in (tools if tools is not None else TOOL_SCHEMAS)
    ]


class AnthropicSession:
    def __init__(self, model: str, max_tokens: int = 4096, extra_tools: Sequence[dict[str, Any]] = ()) -> None:
        import anthropic  # optional extra: crucible[anthropic]

        self._client = anthropic.Anthropic()  # key from ANTHROPIC_API_KEY
        self._model_name = model
        self._model = model
        self._max_tokens = max_tokens
        self._tools = [*TOOL_SCHEMAS, *extra_tools]
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
            tools=self._tools,  # type: ignore[arg-type]
        )
        self._messages.append({"role": "assistant", "content": self._serialize_content(resp.content)})
        self._in_tokens += resp.usage.input_tokens
        self._out_tokens += resp.usage.output_tokens
        return extract_anthropic_calls(resp.content)

    @staticmethod
    def _serialize_content(blocks: Iterable[Any]) -> list[dict[str, Any]]:
        result = []
        for b in blocks:
            t = getattr(b, "type", None)
            if t == "text":
                result.append({"type": "text", "text": b.text})
            elif t == "tool_use":
                result.append({"type": "tool_use", "id": b.id, "name": b.name, "input": dict(b.input)})
            else:
                result.append({"type": t or "unknown", "raw": str(b)})
        return result

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

    def __init__(self, model: str, base_url: str | None = None, extra_tools: Sequence[dict[str, Any]] = ()) -> None:
        from openai import OpenAI  # optional extra: crucible[openai]

        # For local models (Ollama), use empty string as API key
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")

        # Use explicit empty string for Ollama/local models, or env var if set
        if base_url and "localhost" in base_url:
            api_key = "ollama"  # Any non-empty string works for Ollama
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "ollama")

        self._client = OpenAI(base_url=base_url, api_key=api_key, max_retries=2, timeout=300.0)
        self._model_name = model
        self._model = model
        self._tools = [*TOOL_SCHEMAS, *extra_tools]
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
            tools=to_openai_tools(self._tools),  # type: ignore[arg-type]
        )
        msg = resp.choices[0].message
        if resp.usage is not None:
            self._in_tokens += resp.usage.prompt_tokens
            self._out_tokens += resp.usage.completion_tokens
        calls = extract_openai_calls(msg)
        if not calls and msg.content:
            # Fallback: small local models emit the call as JSON text, not a native
            # tool_call. Synthesize a proper tool_calls message so the follow-up tool
            # results reference valid ids.
            text_calls = extract_text_tool_calls(msg.content)
            if text_calls:
                self._messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "id": c.id,
                                "type": "function",
                                "function": {"name": c.name, "arguments": json.dumps(c.args)},
                            }
                            for c in text_calls
                        ],
                    }
                )
                return text_calls
        self._messages.append(msg.model_dump(exclude_none=True))
        return calls

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

    def __init__(self, model: str, max_tokens: int = 4096, extra_tools: Sequence[dict[str, Any]] = ()) -> None:
        import google.genai as genai  # optional extra: crucible[gemini]
        import google.genai.types as types  # type: ignore[reportMissingImports]
        # type: ignore[reportMissingImports]

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY must be set in environment")
        # 180s per-request timeout (ms). The SDK default is None (infinite) — a
        # transient Gemini stall would hang the whole run until wall-clock. With
        # the 3-attempt retry below, a stall now fails fast and recovers instead.
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=180_000),
        )  # type: ignore[reportPrivateImportUsage]
        self._model_name = model
        self._model = model
        self._max_tokens = max_tokens
        self._tools = [*TOOL_SCHEMAS, *extra_tools]
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
        parts = []
        for r in results:
            function_name = self._call_id_to_name.get(r.call_id, "unknown")
            function_response = self._types.FunctionResponse(
                id=r.call_id,
                name=function_name,
                response={"output": r.content},
            )
            parts.append(self._types.Part(function_response=function_response))
        # Function responses must be "user" role — "model" causes a server disconnect
        self._history.append({"role": "user", "parts": parts})
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
        for tool in self._tools:
            fd = self._types.FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters_json_schema=tool["input_schema"],
            )
            function_declarations.append(fd)

        # Call Gemini API — retry on transient network errors (disconnect, timeout)
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=gemini_history,
                    config=self._types.GenerateContentConfig(
                        max_output_tokens=self._max_tokens,
                        tool_config=self._types.ToolConfig(
                            function_calling_config=self._types.FunctionCallingConfig(
                                mode=self._types.FunctionCallingConfigMode.AUTO,
                            ),
                        ),
                        tools=[self._types.Tool(function_declarations=function_declarations)],
                    ),
                )
                break
            except Exception as exc:
                msg = str(exc)
                if any(t in msg for t in ("disconnected", "RemoteProtocol", "ConnectionError", "Timeout")):
                    last_exc = exc
                    time.sleep(2**attempt)
                else:
                    raise
        else:
            raise last_exc  # type: ignore[misc]

        # Extract tool calls
        calls = []
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content:
                for part in candidate.content.parts or []:
                    if part.function_call:
                        call = ToolCall(
                            id=part.function_call.id or "",
                            name=part.function_call.name or "",
                            args=part.function_call.args or {},
                        )
                        calls.append(call)
                        # Track call ID to function name for later responses
                        self._call_id_to_name[call.id] = call.name

        # Append model's response to history
        model_parts = []
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content:
                for part in candidate.content.parts or []:
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
                fc = getattr(part, "function_call", None)
                fr = getattr(part, "function_response", None)
                text = getattr(part, "text", None)
                if fc is not None:
                    parts.append({"function_call": {"name": fc.name, "args": dict(fc.args or {}), "id": fc.id}})
                elif fr is not None:
                    parts.append({"function_response": {"name": fr.name, "response": dict(fr.response or {}), "id": fr.id}})
                elif text:
                    parts.append({"text": text})
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


def make_session(model: str, base_url: str | None = None, extra_tools: Sequence[dict[str, Any]] = ()) -> LLMSession:
    if model.startswith("claude"):
        return AnthropicSession(model, extra_tools=extra_tools)
    if model.startswith("gemini"):
        return GeminiSession(model, extra_tools=extra_tools)
    return OpenAICompatSession(model, base_url=base_url, extra_tools=extra_tools)


@runtime_checkable
class AdvisorSession(Protocol):
    def complete(self, system: str, user: str) -> str: ...
    @property
    def cost_usd(self) -> float: ...


class _AnthropicAdvisor:
    def __init__(self, model: str, max_tokens: int = 1024) -> None:
        import anthropic

        self._client = anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens
        self._in = 0
        self._out = 0

    def complete(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self._model, max_tokens=self._max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        self._in += resp.usage.input_tokens
        self._out += resp.usage.output_tokens
        return "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text")

    @property
    def cost_usd(self) -> float:
        pin, pout = price_for(self._model)
        return (self._in * pin + self._out * pout) / 1_000_000


class _OpenAIAdvisor:
    def __init__(self, model: str, base_url: str | None = None) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url or os.environ.get("OPENAI_BASE_URL"))
        self._model = model
        self._in = 0
        self._out = 0

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        if resp.usage is not None:
            self._in += resp.usage.prompt_tokens
            self._out += resp.usage.completion_tokens
        return resp.choices[0].message.content or ""

    @property
    def cost_usd(self) -> float:
        pin, pout = price_for(self._model)
        return (self._in * pin + self._out * pout) / 1_000_000


class _GeminiAdvisor:
    def __init__(self, model: str, max_tokens: int = 1024) -> None:
        import google.genai as genai
        import google.genai.types as types  # type: ignore[reportMissingImports]

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY must be set in environment")
        # 180s per-request timeout (ms). The SDK default is None (infinite) — a
        # transient Gemini stall would hang the whole run until wall-clock. With
        # the 3-attempt retry in GeminiSession, a stall now fails fast and recovers instead.
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=180_000),
        )  # type: ignore[reportPrivateImportUsage]
        self._model = model
        self._max_tokens = max_tokens
        self._types = types
        self._in = 0
        self._out = 0

    def complete(self, system: str, user: str) -> str:
        resp = self._client.models.generate_content(
            model=self._model,
            contents=[self._types.Content(role="user", parts=[self._types.Part(text=user)])],
            config={"max_output_tokens": self._max_tokens, "system_instruction": system},
        )
        return getattr(resp, "text", "") or ""

    @property
    def cost_usd(self) -> float:
        pin, pout = price_for(self._model)
        return (self._in * pin + self._out * pout) / 1_000_000


def make_advisor_session(model: str, base_url: str | None = None) -> AdvisorSession:
    if model.startswith("claude"):
        return _AnthropicAdvisor(model)
    if model.startswith("gemini"):
        return _GeminiAdvisor(model)
    return _OpenAIAdvisor(model, base_url=base_url)
