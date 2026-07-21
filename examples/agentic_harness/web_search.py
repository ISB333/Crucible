"""Tavily web-search tool exposed to the worker and the shepherd.

Lets the models research coding techniques and BigCodeBench task patterns on demand
instead of guessing. Requires TAVILY_API_KEY in the environment.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable

WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Search the web for Python coding techniques, library APIs, BigCodeBench task "
        "patterns, and algorithm implementations. Returns titled snippets. Use this "
        "to ground your harness edits in real knowledge."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "the search query"}},
        "required": ["query"],
    },
}

EXTRA_TOOLS = [WEB_SEARCH_TOOL]


def web_search(args: dict) -> str:
    """Tavily search -> concatenated titled snippets (truncated)."""
    key = os.environ.get("TAVILY_API_KEY", "")
    if not key:
        return "web_search unavailable: TAVILY_API_KEY not set"
    query = str(args.get("query", "")).strip()
    if not query:
        return "web_search: empty query"
    body = json.dumps(
        {"api_key": key, "query": query, "max_results": 5, "search_depth": "basic"}
    ).encode()
    req = urllib.request.Request(
        "https://api.tavily.com/search",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            data = json.loads(r.read())
    except Exception as exc:
        return f"web_search error: {exc!r}"
    results = data.get("results", [])
    if not results:
        return f"web_search: no results for {query!r}"
    return "\n\n".join(
        f"## {res.get('title', '(untitled)')}\nURL: {res.get('url', '')}\n"
        f"{str(res.get('content', ''))[:600]}"
        for res in results
    )


TOOL_HANDLERS: dict[str, Callable[[dict], str]] = {"web_search": web_search}
