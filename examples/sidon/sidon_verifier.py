"""Sidon set verifier: checks the Sidon property and scores by set size."""
import json
from dataclasses import dataclass

from crucible.artifact import Artifact, scan_holes
from crucible.verify import Fail, Ok, Partial, RunContext, Scored, Verdict


@dataclass(frozen=True)
class SidonVerifier:
    """Score = size of the returned Sidon set. Ok when size >= target_size."""

    target_size: int = 100
    max_val: int = 10000
    timeout_s: float = 30.0
    deterministic: bool = True

    @property
    def verifier_id(self) -> str:
        return f"sidon:target={self.target_size},max={self.max_val}"

    def verify(self, artifact: Artifact, ctx: RunContext) -> Verdict:
        holes = scan_holes(artifact)
        if holes:
            return Partial(open_holes=holes, feedback="solution not implemented yet")

        ws = ctx.materialize(artifact)
        inline = (
            "import json, sys; sys.path.insert(0, '.'); "
            "from problem import generate_sidon_set; "
            "print(json.dumps(generate_sidon_set()))"
        )
        res = ctx.sandbox.run(
            ["python3", "-c", inline],
            cwd=ws,
            timeout_s=self.timeout_s,
            network=ctx.task.network,
        )

        if res.timed_out:
            return Fail(
                feedback=(
                    f"Timed out after {self.timeout_s}s — the algorithm is too slow.\n"
                    "Hint: a greedy O(n²) checker is fine for sets up to ~200 elements;"
                    " avoid O(n³) or sleeping."
                )
            )

        if res.exit_code != 0:
            output = (res.stderr + res.stdout).strip()[-2000:]
            return Fail(feedback=f"Runtime error:\n{output}")

        try:
            parsed = json.loads(res.stdout)
        except (json.JSONDecodeError, ValueError):
            return Fail(feedback=f"runner.py must print a JSON array; got:\n{res.stdout[:300]}")

        if not isinstance(parsed, list):
            return Fail(feedback="generate_sidon_set() must return a list")

        result_set: list[int] = parsed

        if not all(isinstance(x, int) for x in result_set):
            return Fail(feedback="All elements must be integers")

        if len(set(result_set)) != len(result_set):
            return Fail(feedback="Elements must be unique")

        if any(x < 1 or x > self.max_val for x in result_set):
            return Fail(feedback=f"All elements must be in [1, {self.max_val}]")

        # Sidon property: all sums a+b (a <= b) must be distinct
        seen: set[int] = set()
        for i in range(len(result_set)):
            for j in range(i, len(result_set)):
                s = result_set[i] + result_set[j]
                if s in seen:
                    a, b = result_set[i], result_set[j]
                    return Fail(
                        feedback=(
                            f"Sidon property violated: sum {s} appears twice "
                            f"(one instance: {a} + {b}).\n"
                            "All pairwise sums — including a+a — must be distinct."
                        )
                    )
                seen.add(s)

        n = len(result_set)
        if n >= self.target_size:
            return Ok(produced=artifact)

        return Scored(
            produced=artifact,
            value=float(n),
            feedback=(
                f"Valid Sidon set, size {n} / {self.target_size}.\n"
                + _hint(n)
            ),
        )


def _hint(n: int) -> str:
    if n <= 4:
        return "Tip: a greedy algorithm (try each candidate, check sums) easily reaches 30+."
    if n <= 20:
        return "Tip: greedy from a random shuffle usually reaches 40–50 within seconds."
    if n <= 50:
        return (
            "Tip: algebraic constructions (Erdős–Turán, Bose, Singer difference sets)"
            " can reach 80–120 for max_val=10000."
        )
    return "Tip: finite-field constructions (q prime power, B₂[q] set) push beyond 100."
