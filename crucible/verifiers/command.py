"""Command verifier — the escape hatch: any CLI-checkable artifact (PRD §3)."""

import shlex
from dataclasses import dataclass

from crucible.artifact import Artifact, scan_holes
from crucible.verify import Fail, Ok, Partial, RunContext, Verdict

FEEDBACK_LIMIT = 4000  # prune verbose tool output before it reaches the LLM context

_SHELL_TOKENS = frozenset({"&&", "||", "|", ";", ">", ">>", "<"})


def tail(text: str, limit: int = FEEDBACK_LIMIT) -> str:
    return text if len(text) <= limit else "…" + text[-limit:]


@dataclass(frozen=True)
class Command:
    cmd: str
    timeout_s: float = 120.0
    deterministic: bool = True

    def __post_init__(self) -> None:
        bad = _SHELL_TOKENS.intersection(shlex.split(self.cmd))
        if bad:
            raise ValueError(
                f"cmd contains shell operator(s) {sorted(bad)} but runs without a shell;"
                " wrap it: sh -c '...'"
            )

    @property
    def verifier_id(self) -> str:
        return f"cmd:{self.cmd}"

    def verify(self, artifact: Artifact, ctx: RunContext) -> Verdict:
        ws = ctx.materialize(artifact)
        res = ctx.sandbox.run(
            shlex.split(self.cmd), cwd=ws, timeout_s=self.timeout_s, network=ctx.task.network
        )
        output = tail((res.stdout + "\n" + res.stderr).strip())
        if res.timed_out:
            msg = f"timeout after {self.timeout_s}s"
            return Fail(feedback=f"{msg}\n{output}".rstrip() if output else msg)
        holes = scan_holes(artifact)
        if res.exit_code == 0 and not holes:
            return Ok(produced=artifact)
        if holes:
            return Partial(open_holes=holes, feedback=output)
        return Fail(feedback=output or f"exit code {res.exit_code}")
