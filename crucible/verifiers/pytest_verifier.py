"""Pytest verifier (PRD §3): OK iff suite green, no skips/xfails, no holes."""

import re
from dataclasses import dataclass

from crucible.artifact import Artifact, scan_holes
from crucible.verifiers.command import tail
from crucible.verify import Fail, Ok, Partial, RunContext, Verdict

_COUNT_RE = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed)")
_SUMMARY_LINE_RE = re.compile(r"\bin \d+(?:\.\d+)?s\b\s*=*\s*$")


def summary_counts(output: str) -> dict[str, int]:
    """Counts from the last pytest-summary-shaped line (ends with 'in N.NNs').

    Test stdout is model-controlled, so any parse is spoofable in principle —
    an atexit hook could print an exactly-shaped line after the real summary.
    This anchor blocks casual injection; the deterministic backstop is the
    integrity deny-list (pytest.skip / xfail tokens are rejected in editable
    regions), so a suite cannot actually skip/xfail without losing at check 2.
    """
    for line in reversed(output.splitlines()):
        if _SUMMARY_LINE_RE.search(line):
            matches = _COUNT_RE.findall(line)
            if matches:
                return {kind: int(n) for n, kind in matches}
    return {}


@dataclass(frozen=True)
class Pytest:
    suite: str = "tests/"
    timeout_s: float = 300.0
    python: str = "python3"
    deterministic: bool = True

    @property
    def verifier_id(self) -> str:
        return f"pytest:{self.suite}"

    def verify(self, artifact: Artifact, ctx: RunContext) -> Verdict:
        ws = ctx.materialize(artifact)
        argv = [
            self.python,
            "-m",
            "pytest",
            self.suite,
            "-q",
            "-p",
            "no:cacheprovider",
            "--tb=short",
            "-rA",
        ]
        res = ctx.sandbox.run(argv, cwd=ws, timeout_s=self.timeout_s, network=ctx.task.network)
        if res.timed_out:
            return Fail(feedback=f"timeout after {self.timeout_s}s")
        output = tail((res.stdout + "\n" + res.stderr).strip())
        holes = scan_holes(artifact)
        if holes:
            return Partial(open_holes=holes, feedback=output)
        counts = summary_counts(res.stdout)
        gamed = any(counts.get(k, 0) for k in ("skipped", "xfailed", "xpassed"))
        if res.exit_code == 0 and not gamed:
            return Ok(produced=artifact)
        if res.exit_code == 0 and gamed:
            msg = "suite green but contains skipped/xfailed tests — not accepted"
            return Fail(feedback=f"{msg}\n{output}")
        return Fail(feedback=output or f"pytest exit code {res.exit_code}")
