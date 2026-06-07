"""Verdict types, RunContext, and the Verifier contract — the crux (PRD §3)."""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable
from uuid import uuid4

from crucible.artifact import Artifact, Hole
from crucible.sandbox import Sandbox
from crucible.task import Task


@dataclass(frozen=True)
class Span:
    file: str
    line_start: int  # 0-based
    line_end: int


@dataclass(frozen=True)
class Ok:
    produced: Artifact


@dataclass(frozen=True)
class Partial:
    open_holes: tuple[Hole, ...]
    feedback: str


@dataclass(frozen=True)
class Fail:
    feedback: str
    locus: Span | None = None


@dataclass(frozen=True)
class Scored:
    """A valid, gradable artifact (PRD §3 optimization extension, v0.5).

    `value` is higher-is-better, range defined by the verifier. A `Scored` verdict is never
    a sole accept — the verifier returns `Ok` when the target is met. The engine ranks
    `Scored` artifacts to return the best when no `Ok` is reached.
    """

    produced: Artifact
    value: float
    feedback: str = ""


Verdict = Ok | Partial | Fail | Scored


@dataclass
class RunContext:
    task: Task
    sandbox: Sandbox
    scratch: Path  # root for materialized workspaces

    def materialize(self, artifact: Artifact) -> Path:
        """Write the artifact under a content-addressed dir; reuse if present."""
        dst = self.scratch / artifact.content_hash[:16]
        if not dst.exists():
            tmp = dst.parent / f"{dst.name}.{uuid4().hex}.tmp"
            for rel, text in artifact.files.items():
                p = tmp / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text)
            try:
                tmp.rename(dst)
            except OSError:  # lost the race — another caller landed dst first
                shutil.rmtree(tmp, ignore_errors=True)
        return dst


@runtime_checkable
class Verifier(Protocol):
    @property
    def deterministic(self) -> bool:
        # False => verdict is advisory, never a sole accept (PRD §3)
        ...

    def verify(self, artifact: Artifact, ctx: RunContext) -> Verdict: ...


def render(v: Verdict) -> str:
    """Verdict -> the feedback text the next LLM turn reasons over.

    PRD §1.3: the feedback is the gradient — a bare boolean is useless.
    """
    match v:
        case Ok():
            return "VERDICT: OK — all obligations met."
        case Partial(open_holes=holes, feedback=fb):
            listing = "\n".join(f"- {h.file}:{h.line + 1} [{h.kind}] {h.text}" for h in holes)
            return f"VERDICT: PARTIAL — {len(holes)} open hole(s):\n{listing}\n{fb}".rstrip()
        case Scored(value=value, feedback=fb):
            return f"VERDICT: SCORED {value:.4f} (higher is better)\n{fb}".rstrip()
        case Fail(feedback=fb, locus=locus):
            at = f" at {locus.file}:{locus.line_start + 1}" if locus else ""
            return f"VERDICT: FAIL{at}\n{fb}".rstrip()
