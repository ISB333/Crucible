"""Crucible search over the agent harness. Gemini rewrites harness.py:solve;
AgenticCodingVerifier measures Tess on a BigCodeBench subset; GLM-5.2 shepherds.

Usage (GOOGLE_API_KEY + OLLAMA_API_KEY from .env):
  uv run python examples/agentic_harness/run_agentic.py
  uv run python examples/agentic_harness/run_agentic.py --workers 1 --episodes 1 \
      --edits 1 --turns 4 --wall-clock 45m --skip-reverify

Dry-run (fastest possible validation):
  bash examples/agentic_harness/serve_tess.sh &
  uv run python examples/agentic_harness/run_agentic.py \
      --workers 1 --episodes 1 --edits 1 --turns 4 --wall-clock 45m --skip-reverify
  bash examples/agentic_harness/serve_tess.sh --stop
"""
import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from agentic_verifier import AgenticCodingVerifier, _locked_run_subset  # noqa: E402
from web_search import EXTRA_TOOLS, TOOL_HANDLERS  # type: ignore[import-not-found]  # noqa: E402
from web_search_advisor import (  # type: ignore[import-not-found]  # noqa: E402
    make_web_search_advisor_factory,
)

from crucible import AdvisorPolicy, Task, budgets, run  # noqa: E402
from crucible.budgets import RunBudget, parse_duration  # noqa: E402

ap = argparse.ArgumentParser(description="Crucible search over the agentic coding harness")
ap.add_argument("--model", default="gemini-3.1-flash-lite", help="worker (Gemini) model")
ap.add_argument("--advisor", default=os.environ.get("OLLAMA_MODEL"),
                help="shepherd model (e.g. glm-5.2:cloud)")
ap.add_argument("--advisor-max-calls", type=int, default=8)
ap.add_argument("--advisor-fail-streak", type=int, default=3)
ap.add_argument("--workers", type=int, default=3)
ap.add_argument("--episodes", type=int, default=6, help="episodes per worker")
ap.add_argument("--edits", type=int, default=4, help="max edits per episode (each ~30 min eval)")
ap.add_argument("--turns", type=int, default=8, help="max turns per episode")
ap.add_argument("--wall-clock", default="10h",
                help="hard wall-clock cap (e.g. '10h', '45m'). Default 10h.")
ap.add_argument("--plateau-patience", type=int, default=3,
                help="stop a worker after N episodes with no rank improvement")
ap.add_argument("--sandbox", choices=["subprocess", "docker"], default="subprocess")
ap.add_argument("--db", default=str(SCRIPT_DIR / "agentic.db"))
ap.add_argument("--skip-reverify", action="store_true",
                help="skip the independent re-verification on the full 25-task subset")
args = ap.parse_args()

if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
    sys.exit("Error: GOOGLE_API_KEY (or GEMINI_API_KEY) not set. Add it to .env or export it.")

# Route a non-Gemini/Claude shepherd through Ollama Cloud when OLLAMA_API_KEY is set.
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

# Build the task from a curated file list: exclude __pycache__/.pyc/.db/.log/.pid
# so the artifact doesn't include runtime artifacts.
_task_files = tuple(
    sorted(
        str(f.relative_to(SCRIPT_DIR))
        for f in SCRIPT_DIR.rglob("*")
        if f.is_file()
        and not any(
            p.startswith(".") or p == "__pycache__"
            for p in f.relative_to(SCRIPT_DIR).parts
        )
        and f.suffix not in (".pyc", ".db")
        and f.name not in ("serve.log", "serve.pid")
    )
)

verifier = AgenticCodingVerifier(
    baseline_path=SCRIPT_DIR / "baseline.json",
    subset_path=SCRIPT_DIR / "tasks" / "subset.json",
)

result = run(
    task=Task(root=SCRIPT_DIR, files=_task_files, editable=("solve",), network=True),
    verifier=verifier,
    model=args.model,
    workers=args.workers,
    episode=budgets(edits=args.edits, turns=args.turns),
    run_budget=RunBudget(
        episodes_per_worker=args.episodes,
        plateau_patience=args.plateau_patience,
        wall_clock_s=parse_duration(args.wall_clock),
    ),
    sandbox=args.sandbox,
    db=args.db,
    advisor=advisor,
    advisor_factory=advisor_factory,
    extra_tools=EXTRA_TOOLS,
    tool_handlers=TOOL_HANDLERS,
)

artifact = result.solution or result.best_partial
solve_text = artifact.region_text(artifact.region("solve"))

if result.solution:
    print(f"\nSOLVED (run {result.run_id})")
else:
    best_partial = result.best_partial
    # best_partial is always an Artifact; Scored verdict carries the pass_rate in the DB
    print(f"\nBest partial (run {result.run_id}):")
print(solve_text[:2000])

# Independent re-verification (R6): re-run the best harness on the FULL 25-task subset,
# never trust the search's claim. Uses the verifier's _load_harness for sys.path hygiene.
#
# _load_harness loads ws / "harness.py", so we create a temp workspace with:
#   - agent_contract.py (copied from SCRIPT_DIR, so the import resolves)
#   - harness.py (the best solve body wrapped in the right module structure)
# This avoids mutating the real harness.py and ensures clean sys.path hygiene.
if not args.skip_reverify:
    print("\n--- Independent re-verification (full 25-task subset) ---")
    import shutil
    import tempfile

    from agent_contract import load_subset  # noqa: E402

    tmp_ws = Path(tempfile.mkdtemp(prefix="reverify_"))
    try:
        # Copy agent_contract.py so the import resolves from the temp workspace
        shutil.copy2(SCRIPT_DIR / "agent_contract.py", tmp_ws / "agent_contract.py")
        # Write the best harness: frozen def-solve signature + editable body from the artifact.
        # The region text (solve_text) is ONLY the indented body — the def line is frozen
        # outside the region so the worker can't drop it.
        best_text = solve_text
        (tmp_ws / "harness.py").write_text(
            "from pathlib import Path\n"
            "from agent_contract import Task, LLM, Tools\n\n"
            "def solve(task: Task, workdir: Path, llm: LLM, tools: Tools) -> None:\n"
            + best_text
        )
        v = AgenticCodingVerifier(
            baseline_path=SCRIPT_DIR / "baseline.json",
            subset_path=SCRIPT_DIR / "tasks" / "subset.json",
        )
        mod = v._load_harness(tmp_ws)
        full_tasks = load_subset(SCRIPT_DIR / "tasks" / "subset.json")
        res = _locked_run_subset(mod, full_tasks, SCRIPT_DIR)
        print(json.dumps({
            "pass": res["pass"],
            "n": res["n"],
            "pass_rate": round(res["pass"] / res["n"], 3) if res["n"] > 0 else 0,
        }))
    except Exception as exc:
        print(f"re-verification failed: {exc!r}")
    finally:
        shutil.rmtree(tmp_ws, ignore_errors=True)
else:
    print("\n(Skipped re-verification: --skip-reverify)")

print(f"\nInspect reasoning: uv run crucible reasoning --db {args.db}")