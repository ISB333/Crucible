"""Agentic shepherd: GLM-5.2 via Ollama Cloud with a web_search tool.

Unlike the stateless LLMAdvisor (one completion), this advisor runs a small tool
loop: it may call web_search as many times as it wants to research the blocker,
then returns a concrete suggestion. Falls back to "(advisor unavailable)" on any
failure (the run continues without a shepherd).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from web_search import WEB_SEARCH_TOOL, web_search  # type: ignore[import-not-found]

from crucible.advisor import AdvisorRequest, AdvisorResponse, render_request, sanitize_advice

_MAX_TOOL_TURNS = 6
_OPENAI_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": WEB_SEARCH_TOOL["name"],
        "description": WEB_SEARCH_TOOL["description"],
        "parameters": WEB_SEARCH_TOOL["input_schema"],
    },
}


class WebSearchAdvisor:
    """An Advisor that researches via web_search before answering."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None) -> None:
        self._model = model
        self._base_url = base_url or "https://ollama.com/v1"
        self._api_key = (
            api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("OLLAMA_API_KEY", "")
        )
        self._cost = 0.0

    def _client(self):
        from openai import OpenAI

        return OpenAI(base_url=self._base_url, api_key=self._api_key or "ollama")

    def consult(self, req: AdvisorRequest) -> AdvisorResponse:
        system, user = render_request(req)
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    system + "\n\nYou have a web_search tool; use it to research concrete "
                    "coding techniques, BigCodeBench patterns, and Python solutions before answering."
                ),
            },
            {"role": "user", "content": user},
        ]
        try:
            client = self._client()
            for _ in range(_MAX_TOOL_TURNS):
                resp = client.chat.completions.create(
                    model=self._model,
                    messages=messages,  # type: ignore[arg-type]
                    tools=[_OPENAI_TOOL],  # type: ignore[arg-type]
                    temperature=0.2,
                )
                msg = resp.choices[0].message
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    messages.append(
                        {"role": "assistant", "content": getattr(msg, "content", None) or ""}
                    )
                    for tc in tool_calls:
                        fn = getattr(tc, "function", None)
                        args_json = getattr(fn, "arguments", "{}") if fn else "{}"
                        name = (
                            getattr(fn, "name", WEB_SEARCH_TOOL["name"])
                            if fn
                            else WEB_SEARCH_TOOL["name"]
                        )
                        if name == WEB_SEARCH_TOOL["name"]:
                            import json as _json

                            try:
                                result = web_search(_json.loads(args_json) if args_json else {})
                            except Exception:
                                result = web_search({})
                        else:
                            result = f"unknown tool {name!r}"
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": getattr(tc, "id", ""),
                                "content": result,
                            }
                        )
                    continue
                advice = sanitize_advice(getattr(msg, "content", "") or "")
                if not advice:
                    return AdvisorResponse(
                        "(advisor returned nothing — proceed on your own)", 0.0, False
                    )
                return AdvisorResponse(advice, 0.0, True)
            return AdvisorResponse("(advisor hit tool-loop cap — proceed on your own)", 0.0, False)
        except Exception:
            return AdvisorResponse("(advisor unavailable — proceed on your own)", 0.0, False)

    @property
    def cost_usd(self) -> float:
        return self._cost


def make_web_search_advisor_factory(
    model: str, base_url: str | None = None
) -> Callable[[], WebSearchAdvisor]:
    def _f() -> WebSearchAdvisor:
        return WebSearchAdvisor(model=model, base_url=base_url)

    return _f
