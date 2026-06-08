"""Sidon set optimization — stress test for Crucible.

Usage (key detected automatically from .env):
  uv run python examples/sidon/run_sidon.py
  uv run python examples/sidon/run_sidon.py --model gemini-2.0-flash
  uv run python examples/sidon/run_sidon.py --model claude-sonnet-4-6
  uv run python examples/sidon/run_sidon.py --model gpt-4o
"""

import argparse
import os
import sys
from pathlib import Path

# Load .env before anything else so key detection works
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))


def _default_model() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude-sonnet-4-6"
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return "gemini-3.1-flash-lite"
    if os.environ.get("OPENAI_API_KEY"):
        return "gpt-4o"
    return "claude-sonnet-4-6"  # will produce a clear auth error from the SDK


def _check_key(model: str) -> None:
    if model.startswith("claude") and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Error: ANTHROPIC_API_KEY not set. Add it to .env or export it.")
    if model.startswith("gemini") and not (
        os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    ):
        sys.exit("Error: GOOGLE_API_KEY (or GEMINI_API_KEY) not set. Add it to .env or export it.")
    if not model.startswith(("claude", "gemini")) and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Error: OPENAI_API_KEY not set. Add it to .env or export it.")


from crucible import Task, budgets, run  # noqa: E402
from crucible.budgets import RunBudget  # noqa: E402
from sidon_verifier import SidonVerifier  # type: ignore[import-not-found]  # noqa: E402

parser = argparse.ArgumentParser(description="Sidon set optimization via Crucible")
parser.add_argument("--model", default=_default_model(), help="LLM model to use")
parser.add_argument("--workers", type=int, default=5)
parser.add_argument("--target", type=int, default=100, help="Set size target for Ok verdict")
parser.add_argument(
    "--sandbox",
    choices=["subprocess", "docker"],
    default="subprocess",
    help="subprocess: no Docker needed (safe for stdlib-only code)",
)
args = parser.parse_args()

_check_key(args.model)
print(f"Model: {args.model}  |  Workers: {args.workers}  |  Target: {args.target}")

result = run(
    task=Task.from_path(SCRIPT_DIR / "problem.py", editable=["solution"]),
    verifier=SidonVerifier(target_size=args.target),
    model=args.model,
    workers=args.workers,
    episode=budgets(edits=30, turns=15),
    run_budget=RunBudget(episodes_per_worker=10, plateau_patience=3),
    sandbox=args.sandbox,
    db=str(SCRIPT_DIR / "sidon.db"),
)

artifact = result.solution or result.best_partial
region_text = artifact.region_text(artifact.region("solution")).strip()

if result.solution:
    print(f"\nSOLVED — hit target size {args.target} (run {result.run_id})")
else:
    print(f"\nBest result — target {args.target} not reached (run {result.run_id}):")

print(region_text[:2000])

# Independent mathematical verification — never trust the LLM's self-report
print("\n--- Verification ---")
try:
    ns: dict = {}
    exec(artifact.files["problem.py"], ns)
    sidon: list[int] = ns["generate_sidon_set"]()
    errors: list[str] = []
    if not isinstance(sidon, list):
        errors.append("not a list")
    elif not all(isinstance(x, int) for x in sidon):
        errors.append("non-integer elements")
    elif len(set(sidon)) != len(sidon):
        errors.append("duplicate elements")
    elif any(x < 1 or x > 10000 for x in sidon):
        errors.append("elements out of [1, 10000]")
    else:
        seen: set[int] = set()
        violated = None
        for i in range(len(sidon)):
            for j in range(i, len(sidon)):
                s = sidon[i] + sidon[j]
                if s in seen:
                    violated = (sidon[i], sidon[j], s)
                    break
                seen.add(s)
            if violated:
                break
        if violated:
            a, b, s = violated
            errors.append(f"Sidon violated: {a}+{b}={s} is a duplicate sum")
    if errors:
        print(f"INVALID: {'; '.join(errors)}")
    else:
        n = len(sidon)
        print(f"VALID Sidon set")
        print(f"  Size : {n}  (target {args.target})")
        print(f"  Range: [{min(sidon)}, {max(sidon)}]")
        print(f"  Sums : {n*(n+1)//2} distinct pairwise sums (including a+a)")
except Exception as exc:
    print(f"Verification error: {exc}")

print(f"\nInspect reasoning: uv run crucible reasoning --db examples/sidon/sidon.db")
