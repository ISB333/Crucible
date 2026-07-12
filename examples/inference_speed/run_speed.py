"""Wave 1: verifier-grounded search over the 9B inference config.

Usage (GOOGLE_API_KEY from .env):
  uv run python examples/inference_speed/run_speed.py
  uv run python examples/inference_speed/run_speed.py --model gemini-2.5-flash \
      --advisor gemini-2.5-pro --workers 4 --target-agg 30 --target-single 8

The worker (Gemini Flash) edits the `config` region; the SpeedQualityVerifier
measures single-stream + aggregate tok/s with a lossless quality gate; the
advisor (Gemini Pro) shepherds on plateau. The 9B is passive cargo.
"""

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))


def _default_worker() -> str:
    return "gemini-3.1-flash-lite"


def _default_advisor() -> str | None:
    # OLLAMA_MODEL names an Ollama-Cloud shepherd (e.g. "glm-5.2:cloud").
    return os.environ.get("OLLAMA_MODEL")


from speed_verifier import SpeedQualityVerifier  # type: ignore[import-not-found]  # noqa: E402
from web_search import EXTRA_TOOLS, TOOL_HANDLERS  # type: ignore[import-not-found]  # noqa: E402
from web_search_advisor import (  # type: ignore[import-not-found]  # noqa: E402
    make_web_search_advisor_factory,
)

from crucible import AdvisorPolicy, Task, budgets, run  # noqa: E402
from crucible.budgets import RunBudget  # noqa: E402

ap = argparse.ArgumentParser(description="CPU inference speed search via Crucible")
ap.add_argument("--model", default=_default_worker(), help="worker (Gemini) model")
ap.add_argument("--advisor", default=_default_advisor(), help="shepherd model (e.g. glm-5.2:cloud)")
ap.add_argument("--advisor-max-calls", type=int, default=8)
ap.add_argument("--advisor-fail-streak", type=int, default=3)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--target-agg", type=float, default=30.0)
ap.add_argument("--target-single", type=float, default=8.0)
ap.add_argument("--sandbox", choices=["subprocess", "docker"], default="subprocess")
ap.add_argument("--db", default=str(SCRIPT_DIR / "speed.db"))
ap.add_argument("--episodes", type=int, default=6, help="episodes per worker")
ap.add_argument("--edits", type=int, default=20, help="max edits per episode")
ap.add_argument("--turns", type=int, default=10, help="max turns per episode")
args = ap.parse_args()

if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
    sys.exit("Error: GOOGLE_API_KEY (or GEMINI_API_KEY) not set. Add it to .env or export it.")

# Route a non-Gemini/Claude shepherd through Ollama Cloud when OLLAMA_API_KEY is set.
# The advisor uses the OpenAI-compatible provider; the Gemini worker ignores OPENAI_*.
if (
    args.advisor
    and not args.advisor.startswith(("claude", "gemini"))
    and os.environ.get("OLLAMA_API_KEY")
):
    os.environ.setdefault("OPENAI_BASE_URL", "https://ollama.com/v1")
    os.environ["OPENAI_API_KEY"] = os.environ["OLLAMA_API_KEY"]

advisor = None
advisor_factory = None
if args.advisor:
    advisor = AdvisorPolicy(
        model=args.advisor,
        max_calls_per_run=args.advisor_max_calls,
        fail_streak=args.advisor_fail_streak,
    )
    # Ollama-Cloud shepherd: an agentic WebSearchAdvisor (GLM-5.2 + web_search tool).
    if not args.advisor.startswith(("claude", "gemini")) and os.environ.get("OLLAMA_API_KEY"):
        advisor_factory = make_web_search_advisor_factory(
            args.advisor, base_url="https://ollama.com/v1"
        )

# Build the task from a curated file list: Task.from_path(dir) would include
# __pycache__/*.pyc (binary, breaks read_text) and speed.db. Exclude those.
_task_files = tuple(
    sorted(
        str(f.relative_to(SCRIPT_DIR))
        for f in SCRIPT_DIR.rglob("*")
        if f.is_file()
        and not any(
            p.startswith(".") or p == "__pycache__" for p in f.relative_to(SCRIPT_DIR).parts
        )
        and f.suffix not in (".pyc", ".db")
    )
)

result = run(
    task=Task(root=SCRIPT_DIR, files=_task_files, editable=("config",), network=True),
    verifier=SpeedQualityVerifier(
        target_agg=args.target_agg,
        target_single=args.target_single,
        baseline_path=SCRIPT_DIR / "baseline.json",
    ),
    model=args.model,
    workers=args.workers,
    episode=budgets(edits=args.edits, turns=args.turns),
    run_budget=RunBudget(episodes_per_worker=args.episodes, plateau_patience=3),
    sandbox=args.sandbox,
    db=args.db,
    advisor=advisor,
    advisor_factory=advisor_factory,
    extra_tools=EXTRA_TOOLS,
    tool_handlers=TOOL_HANDLERS,
)

artifact = result.solution or result.best_partial
cfg_text = artifact.region_text(artifact.region("config")).strip()

if result.solution:
    print(
        f"\nSOLVED — hit target agg>={args.target_agg} & single>={args.target_single} "
        f"(run {result.run_id})"
    )
else:
    print(f"\nBest partial — target not reached (run {result.run_id}):")
print(cfg_text[:2000])

# Independent re-verification: re-run the best config, never trust the search's claim.
print("\n--- Independent re-verification ---")
from harness import run_harness  # noqa: E402

try:
    ns: dict = {}
    exec(artifact.files["config.py"], ns)
    best_cfg = ns["CONFIG"]
    r = run_harness(best_cfg, SCRIPT_DIR, max_tokens=64)
    print(
        json.dumps(
            {
                "single_stream": r["single_stream"]["tok_s"],
                "aggregate": r["aggregate"]["tok_s"],
                "quality_path": r["quality"]["path"],
                "loaded_model": r["loaded_model"],
            },
            indent=2,
        )
    )
except Exception as exc:
    print(f"re-verification failed (best config is invalid): {exc!r}")

print("\nInspect reasoning: uv run crucible reasoning --db examples/inference_speed/speed.db")
