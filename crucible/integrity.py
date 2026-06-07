"""Integrity gate (PRD §7) — checks 1+2. Check 3 (fresh-sandbox re-verify) lands in Plan 2.

A pass that fails this gate is rejected: passing the verifier only counts if the
immutable spec was untouched and no escape token was used.
"""

import re
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from crucible.artifact import Artifact, masked_view
from crucible.sandbox import Sandbox
from crucible.task import Task
from crucible.verify import Ok, RunContext, Verifier


@runtime_checkable
class IntegrityCheck(Protocol):
    def check(self, artifact: Artifact, ctx: RunContext) -> bool: ...


@dataclass(frozen=True)
class ImmutableRegions:
    """Check 1: bytes outside editable regions must equal the frozen original."""

    baseline: Mapping[str, str]  # masked view of the frozen original

    @classmethod
    def freeze(cls, original: Artifact) -> "ImmutableRegions":
        return cls(MappingProxyType(masked_view(original)))

    def check(self, artifact: Artifact, ctx: RunContext) -> bool:
        return masked_view(artifact) == dict(self.baseline)


# Regexes, per file suffix, scanned over editable-region text only (PRD §7 deny-lists).
DEFAULT_DENY: dict[str, tuple[str, ...]] = {
    ".py": (
        r"pytest\.skip",
        r"\bxfail\b",
        r"unittest\.mock",
        r"\bmonkeypatch\b",
        r"#\s*type:\s*ignore",
        r"\bassert\s+True\b",
    ),
    ".lean": (r"\bsorry\b", r"\bsorryAx\b", r"\baxiom\b", r"\bnative_decide\b"),
}


@dataclass(frozen=True)
class DenyTokens:
    """Check 2: a solution must not contain disallowed escape tokens."""

    extra: tuple[str, ...] = ()

    def violations(self, artifact: Artifact) -> list[str]:
        out: list[str] = []
        for r in artifact.regions:
            patterns = DEFAULT_DENY.get(Path(r.file).suffix, ()) + self.extra
            body = artifact.region_text(r)
            for pat in patterns:
                if m := re.search(pat, body):
                    out.append(f"{r.file} region {r.name!r}: forbidden token {m.group(0)!r}")
        return out

    def check(self, artifact: Artifact, ctx: RunContext) -> bool:
        return not self.violations(artifact)


@dataclass(frozen=True)
class Composite:
    checks: tuple[IntegrityCheck, ...]

    def check(self, artifact: Artifact, ctx: RunContext) -> bool:
        return all(c.check(artifact, ctx) for c in self.checks)


def fresh_reverify(
    artifact: Artifact,
    verifier: Verifier,
    task: Task,
    sandbox_factory: Callable[[], Sandbox],
) -> bool:
    """Check 3: the pass must reproduce in a fresh scratch + fresh locked sandbox.
    Infra exceptions from verify propagate to the caller (run-fatal in v0)."""
    with tempfile.TemporaryDirectory(prefix="crucible-reverify-") as td:
        ctx = RunContext(task=task, sandbox=sandbox_factory(), scratch=Path(td))
        # Partial counts as not-reproduced — only a clean Ok is accepted.
        return isinstance(verifier.verify(artifact, ctx), Ok)
