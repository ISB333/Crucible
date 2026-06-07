"""Rubric verifier (PRD §3): LLM judge >= threshold. Advisory only in v0."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from crucible.artifact import Artifact
from crucible.verify import Fail, Ok, RunContext, Verdict

_JUDGE_PROMPT = """\
You are a strict reviewer. Score the artifact below against the rubric.
Respond with ONLY a JSON object: {{"score": <float 0..1>, "feedback": "<one paragraph>"}}

Rubric:
{spec}

Artifact:
{files}
"""


def _default_judge(model: str) -> Callable[[str], str]:
    def judge(prompt: str) -> str:
        import anthropic  # optional extra: crucible[anthropic]

        client = anthropic.Anthropic()
        resp: Any = client.messages.create(
            model=model, max_tokens=1024, messages=[{"role": "user", "content": prompt}]
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    return judge


@dataclass(frozen=True)
class Rubric:
    """LLM judge >= threshold.

    Advisory only in v0: artifact text reaches the judge prompt, so treat the
    verdict as a hint — deterministic=False means it can never be the sole
    accept signal.
    """

    spec: str
    threshold: float = 0.8
    model: str = "claude-sonnet-4-6"
    judge: Callable[[str], str] | None = None  # injectable for tests
    deterministic: bool = False  # advisory: never the sole accept signal (PRD §3)

    @property
    def verifier_id(self) -> str:
        return f"rubric:{self.threshold}"

    def verify(self, artifact: Artifact, ctx: RunContext) -> Verdict:
        files = "\n\n".join(f"=== {p} ===\n{t}" for p, t in sorted(artifact.files.items()))
        prompt = _JUDGE_PROMPT.format(spec=self.spec, files=files)
        judge = self.judge or _default_judge(self.model)
        raw = judge(prompt)
        # Validate model output before acting on it (Constitution art. 10).
        try:
            start, end = raw.index("{"), raw.rindex("}") + 1
            parsed = json.loads(raw[start:end])
            raw_score = parsed["score"]
            if isinstance(raw_score, bool) or not isinstance(raw_score, int | float):
                return Fail(feedback="judge returned non-numeric score — treated as reject")
            score = float(raw_score)
            feedback = str(parsed.get("feedback", ""))
        except (ValueError, KeyError, TypeError):
            return Fail(feedback="judge output unparsable — treated as reject")
        if not 0.0 <= score <= 1.0:
            return Fail(feedback=f"judge score {score!r} out of range [0,1] — treated as reject")
        if score >= self.threshold:
            return Ok(produced=artifact)
        return Fail(feedback=f"rubric score {score:.2f} < {self.threshold}: {feedback}")
