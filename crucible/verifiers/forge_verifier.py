"""Forge verifier (PRD §3): OK iff `forge test` is green with >= 1 test, no holes.

A 0-tests run (delete/rename the test so nothing runs) is a gaming vector -> Fail.
"""

import re
from dataclasses import dataclass

from crucible.artifact import Artifact, scan_holes
from crucible.verifiers.command import tail
from crucible.verify import Fail, Ok, Partial, RunContext, Verdict

# Foundry summary line, e.g. "... : 1 tests passed, 0 failed, 0 skipped (1 total tests)".
# Verify the exact wording against `forge --version` in crucible-solidity:0 and adjust if needed.
_PASSED_RE = re.compile(r"(\d+) tests? passed")
_FAILED_RE = re.compile(r"(\d+) failed")
_TOTAL_RE = re.compile(r"\((\d+) total tests?\)")


def forge_counts(output: str) -> dict[str, int]:
    """Counts from the last summary-shaped line (model-controlled stdout: anchor on shape)."""
    for line in reversed(output.splitlines()):
        if _PASSED_RE.search(line) and _TOTAL_RE.search(line):
            passed = int(_PASSED_RE.search(line).group(1))  # type: ignore[union-attr]
            failed_m = _FAILED_RE.search(line)
            total = int(_TOTAL_RE.search(line).group(1))  # type: ignore[union-attr]
            return {
                "passed": passed,
                "failed": int(failed_m.group(1)) if failed_m else 0,
                "total": total,
            }
    return {}


@dataclass(frozen=True)
class Forge:
    timeout_s: float = 300.0
    verbosity: str = "-vvv"  # surface call traces in feedback (the Ralph-loop gradient)
    deterministic: bool = True

    @property
    def verifier_id(self) -> str:
        return "forge"

    def verify(self, artifact: Artifact, ctx: RunContext) -> Verdict:
        ws = ctx.materialize(artifact)
        res = ctx.sandbox.run(
            ["forge", "test", self.verbosity],
            cwd=ws,
            timeout_s=self.timeout_s,
            network=ctx.task.network,
        )
        if res.timed_out:
            return Fail(feedback=f"timeout after {self.timeout_s}s")
        output = tail((res.stdout + "\n" + res.stderr).strip())
        holes = scan_holes(artifact)
        if holes:
            return Partial(open_holes=holes, feedback=output)
        counts = forge_counts(res.stdout)
        if counts.get("total", 0) == 0:
            print(f"res.stdout: {repr(res.stdout)}")
            print(f"res.stderr: {repr(res.stderr)}")
            return Fail(feedback=f"no tests ran — refusing to accept\n{output}")
        if res.exit_code == 0 and counts.get("failed", 1) == 0 and counts.get("passed", 0) >= 1:
            return Ok(produced=artifact)
        return Fail(feedback=output or f"forge exit code {res.exit_code}")
