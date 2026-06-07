"""Molecular verifier (PRD §3 optimization, v0.5): score a SMILES with a deterministic surrogate.

Ok iff valid + scaffold preserved + score >= target; Scored(value) if valid-but-below; Fail if
invalid or scaffold missing; Partial if the molecule is still a hole. The surrogate is deterministic
(baked into crucible-chem:0), so Scored/Ok verdicts reproduce under fresh_reverify.
"""

import json
from dataclasses import dataclass

from crucible.artifact import Artifact, scan_holes
from crucible.verifiers.command import tail
from crucible.verify import Fail, Ok, Partial, RunContext, Scored, Verdict


@dataclass(frozen=True)
class Chem:
    target: float  # score >= target => Ok (full accept via the unchanged first-wins path)
    scaffold: str = ""  # SMARTS the molecule must contain (empty => no constraint)
    molecule_file: str = "molecule.smi"
    scorer: str = "/opt/score_smiles.py"
    timeout_s: float = 120.0
    deterministic: bool = True

    @property
    def verifier_id(self) -> str:
        return f"chem:{self.target}"

    def verify(self, artifact: Artifact, ctx: RunContext) -> Verdict:
        holes = scan_holes(artifact)
        if holes:
            return Partial(open_holes=holes, feedback="molecule not written yet")
        ws = ctx.materialize(artifact)
        res = ctx.sandbox.run(
            ["python", self.scorer, self.molecule_file, self.scaffold],
            cwd=ws,
            timeout_s=self.timeout_s,
            network=ctx.task.network,
        )
        if res.timed_out:
            return Fail(feedback=f"timeout after {self.timeout_s}s")
        try:
            report = json.loads(res.stdout)
        except (json.JSONDecodeError, TypeError):
            return Fail(feedback=f"scorer produced no JSON:\n{tail(res.stdout + res.stderr)}")
        if not report.get("valid"):
            return Fail(feedback=str(report.get("error") or "invalid molecule"))
        if self.scaffold and not report.get("has_scaffold"):
            return Fail(feedback=f"required scaffold {self.scaffold!r} not present")
        score = float(report.get("score", 0.0))
        if score >= self.target:
            return Ok(produced=artifact)
        return Scored(
            produced=artifact, value=score, feedback=f"score {score:.4f} < target {self.target}"
        )
