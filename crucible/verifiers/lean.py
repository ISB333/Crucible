"""Lean verifier (PRD §3): OK iff every .lean file compiles sorry-free."""

import re
from dataclasses import dataclass

from crucible.artifact import Artifact, Hole
from crucible.verifiers.command import tail
from crucible.verify import Fail, Ok, Partial, RunContext, Verdict

_SORRY_RE = re.compile(r"\bsorry\b")

# Matches Lean's compiler diagnostic for sorry use:
#   Lean 4.30+:  declaration uses `sorry`  (backtick-quoted)
#   Lean pre-4.x: declaration uses 'sorry'  (single-quoted)
# This check is defense-in-depth: integrity gates 1+2 independently block
# sorry tokens in editable regions and edits outside them, so a spoofed
# #eval-printed line matching this pattern cannot bypass those earlier gates.
_SORRY_WARNING_RE = re.compile(r"declaration uses [`']sorry[`']")


def _sorry_holes(artifact: Artifact, file: str) -> list[Hole]:
    return [
        Hole(file=file, line=i, kind="sorry", text=line.strip())
        for i, line in enumerate(artifact.files[file].splitlines())
        if _SORRY_RE.search(line)
    ]


@dataclass(frozen=True)
class Lean:
    timeout_s: float = 600.0
    deterministic: bool = True

    @property
    def verifier_id(self) -> str:
        return "lean"

    def verify(self, artifact: Artifact, ctx: RunContext) -> Verdict:
        lean_files = [p for p in sorted(artifact.files) if p.endswith(".lean")]
        if not lean_files:
            return Fail(feedback="no .lean files in artifact")
        ws = ctx.materialize(artifact)
        holes: list[Hole] = []
        outputs: list[str] = []
        for f in lean_files:
            res = ctx.sandbox.run(
                ["lean", f], cwd=ws, timeout_s=self.timeout_s, network=ctx.task.network
            )
            if res.timed_out:
                return Fail(feedback=f"timeout compiling {f} after {self.timeout_s}s")
            out = (res.stdout + "\n" + res.stderr).strip()
            if res.exit_code != 0:
                return Fail(feedback=tail(out) or f"lean exit code {res.exit_code}")
            if _SORRY_WARNING_RE.search(out):
                holes.extend(_sorry_holes(artifact, f))
            if out:
                outputs.append(out)
        if holes:
            return Partial(open_holes=tuple(holes), feedback=tail("\n".join(outputs)))
        return Ok(produced=artifact)
