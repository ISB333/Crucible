"""Pyright verifier (PRD §3): OK iff zero type errors, no holes."""

import json
from dataclasses import dataclass
from pathlib import Path

from crucible.artifact import Artifact, scan_holes
from crucible.verifiers.command import tail
from crucible.verify import Fail, Ok, Partial, RunContext, Span, Verdict

_CONFIG = '{{"typeCheckingMode": "{mode}", "pythonVersion": "3.12"}}\n'


@dataclass(frozen=True)
class Pyright:
    strict: bool = True
    timeout_s: float = 300.0
    binary: str = "pyright"  # overridable: sandboxes have a minimal PATH
    deterministic: bool = True

    @property
    def verifier_id(self) -> str:
        return f"pyright:{'strict' if self.strict else 'standard'}"

    def verify(self, artifact: Artifact, ctx: RunContext) -> Verdict:
        ws = ctx.materialize(artifact)
        config = ws / "pyrightconfig.json"
        if not config.exists():  # deterministic given the artifact hash — safe to cache
            config.write_text(_CONFIG.format(mode="strict" if self.strict else "standard"))
        res = ctx.sandbox.run([self.binary, "--outputjson", "."], cwd=ws, timeout_s=self.timeout_s)
        if res.timed_out:
            return Fail(feedback=f"timeout after {self.timeout_s}s")
        holes = scan_holes(artifact)
        try:
            report = json.loads(res.stdout)
            errors = [
                d for d in report.get("generalDiagnostics", []) if d.get("severity") == "error"
            ]
        except (json.JSONDecodeError, TypeError, ValueError):
            return Fail(
                feedback=f"pyright produced no parsable report:\n{tail(res.stdout + res.stderr)}"
            )
        if holes:
            return Partial(open_holes=holes, feedback=tail(res.stdout))
        if not errors:
            return Ok(produced=artifact)
        first = errors[0]
        locus = None
        feedback_lines = []
        for d in errors[:20]:
            rel = Path(d.get("file", "?")).name
            line = int(d.get("range", {}).get("start", {}).get("line", 0))
            feedback_lines.append(f"{rel}:{line + 1}: {d.get('message', '')}")
        line = int(first.get("range", {}).get("start", {}).get("line", 0))
        locus = Span(file=Path(first.get("file", "?")).name, line_start=line, line_end=line)
        return Fail(
            feedback=f"{len(errors)} type error(s):\n" + "\n".join(feedback_lines), locus=locus
        )
