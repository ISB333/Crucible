"""FROZEN agent contract: the primitives the harness orchestrates.

The worker edits ONLY the body of `solve` in harness.py. Everything here is frozen.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Task:
    id: str
    spec: str            # docstring + signature + visible example (what the agent sees)
    skeleton_path: Path  # workdir-relative file the harness completes
    eval_task_id: str    # BigCodeBench task id for the official hidden-test check


@dataclass(frozen=True)
class LLM:
    """OpenAI-compatible chat client. Tess is a reasoning model -> disable thinking."""
    base_url: str
    model: str
    _client: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        from openai import OpenAI
        object.__setattr__(self, "_client", OpenAI(base_url=self.base_url, api_key="not-needed"))

    def chat(self, messages: list[dict], max_tokens: int = 256, temperature: float = 0.0) -> str:
        r = self._client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens,
            temperature=temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return r.choices[0].message.content or ""


@dataclass
class Tools:
    """Bound to a per-task sandbox workdir."""
    workdir: Path

    def read_file(self, rel: str) -> str:
        return (self.workdir / rel).read_text()

    def write_file(self, rel: str, content: str) -> None:
        (self.workdir / rel).write_text(content)

    def list_dir(self, rel: str = ".") -> list[str]:
        return [p.name for p in (self.workdir / rel).iterdir()]

    def run_visible_tests(self, test_rel: str) -> str:
        """Run the VISIBLE example tests only (never the hidden BigCodeBench tests)."""
        import subprocess
        return subprocess.run(
            ["python", test_rel], cwd=self.workdir, capture_output=True, text=True, timeout=30
        ).stdout


def solve(task: Task, workdir: Path, llm: LLM, tools: Tools) -> None:
    """The worker replaces this body. Default: NotImplementedError (an unfilled hole)."""
    raise NotImplementedError


def load_subset(path: Path) -> list[Task]:
    """Load the curated subset.json -> list[Task]. spec/skeleton are resolved at solve time."""
    raw = json.loads(Path(path).read_text())
    tasks_root = Path(path).parent
    out: list[Task] = []
    for e in raw:
        tid = e["task_id"]
        spec = (tasks_root / tid / "spec.md").read_text()
        out.append(Task(id=e["split"] + "/" + tid, spec=spec,
                        skeleton_path=Path(tid) / "skeleton.py", eval_task_id=tid))
    return out