"""CLI (PRD §9):

crucible run problem.py --editable solution --verifier pytest:tests/ --workers 10 \
    --episode-edits 90 --episode-turns 40 --run-budget 2h,200usd
"""

import argparse
from collections.abc import Callable
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from crucible import run as sdk_run
from crucible.budgets import EpisodeBudget, RunBudget, parse_duration
from crucible.orchestrator import SearchResult
from crucible.task import Task
from crucible.verifiers import Command, Lean, Pyright, Pytest
from crucible.verify import Verifier


def parse_verifier(spec: str) -> Verifier:
    kind, _, arg = spec.partition(":")
    match kind:
        case "pytest":
            return Pytest(suite=arg or "tests/")
        case "pyright":
            return Pyright(strict=(arg or "strict") == "strict")
        case "cmd":
            return Command(arg)
        case "lean":
            return Lean()
        case _:
            raise ValueError(f"unknown verifier spec {spec!r} (pytest:|pyright:|cmd:|lean)")


def parse_run_budget(text: str) -> RunBudget:
    wall = RunBudget().wall_clock_s
    usd: float | None = None
    for part in text.split(","):
        part = part.strip()
        if part.endswith("usd"):
            amount = part[:-3]
            if not amount:
                raise ValueError(f"run budget {part!r}: missing amount before 'usd' (e.g. 200usd)")
            usd = float(amount)
        else:
            wall = parse_duration(part)
    return RunBudget(wall_clock_s=wall, usd=usd)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="crucible")
    sub = p.add_subparsers(dest="command", required=True)

    # Run command (existing)
    r = sub.add_parser("run", help="search for a verified solution")
    r.add_argument("path", help="problem file or directory")
    r.add_argument("--editable", action="append", required=True, help="editable region name")
    r.add_argument("--verifier", required=True, help="pytest:SUITE | pyright:MODE | cmd:CMD | lean")
    r.add_argument("--model", default="claude-sonnet-4-6")
    r.add_argument("--workers", type=int, default=10)
    r.add_argument("--episode-edits", type=int, default=90)
    r.add_argument("--episode-turns", type=int, default=40)
    r.add_argument("--run-budget", default="2h", help="e.g. 2h,200usd")
    r.add_argument("--sandbox", choices=["docker", "subprocess"], default="docker")
    r.add_argument("--image", default=None, help="docker image override")
    r.add_argument("--out", default="crucible-out", help="where to write the result artifact")
    r.add_argument("--db", default="crucible.db")

    # Show reasoning command
    s = sub.add_parser("reasoning", help="show LLM reasoning for a run")
    s.add_argument("run_id", type=int, help="run ID to display reasoning for")
    s.add_argument("--db", default="crucible.db", help="database file to read from")
    s.add_argument("--worker", type=int, help="specific worker index to show")
    s.add_argument("--episode", type=int, help="specific episode number to show")

    return p


def show_reasoning(args) -> int:
    """Show LLM reasoning for a specific run."""
    from crucible.store import Store
    import json

    store = Store(args.db)
    conn = store._conn
    cursor = conn.cursor()

    # Build query to fetch reasoning
    query = """
        SELECT w.id as worker_id, w.idx, e.ordinal as episode, e.reasoning_json
        FROM workers w
        JOIN episodes e ON w.id = e.worker_id
        JOIN runs r ON w.run_id = r.id
        WHERE r.id = ?
    """
    params = [args.run_id]

    if args.worker is not None:
        query += " AND w.idx = ?"
        params.append(args.worker)
    if args.episode is not None:
        query += " AND e.ordinal = ?"
        params.append(args.episode)

    query += " ORDER BY w.idx, e.ordinal"

    cursor.execute(query, params)
    rows = cursor.fetchall()

    if not rows:
        print(f"No reasoning found for run {args.run_id}")
        if args.worker is not None:
            print(f"Worker {args.worker}")
        if args.episode is not None:
            print(f"Episode {args.episode}")
        return 1

    for worker_id, worker_idx, episode, reasoning_json in rows:
        print(f"\n=== Worker {worker_idx} — Episode {episode} ===")
        if reasoning_json:
            try:
                messages = json.loads(reasoning_json)
                for i, msg in enumerate(messages):
                    role = msg.get("role", "unknown")
                    print(f"\n[{i}] {role}")
                    if "content" in msg:
                        content = msg["content"]
                        if isinstance(content, str):
                            print(content)
                        elif isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict):
                                    if item.get("type") == "text":
                                        print(item.get("text", ""))
                                    elif item.get("type") == "tool_use":
                                        print(f"Tool: {item.get('name')}({item.get('input', {})})")
                                    elif item.get("type") == "tool_result":
                                        print(f"Result: {item.get('content', '')}")
                                elif hasattr(item, "__dict__"):
                                    # Handle TextBlock and ToolUseBlock objects (from Anthropic SDK)
                                    if hasattr(item, "text"):
                                        print(item.text)
                                    elif hasattr(item, "name"):
                                        print(f"Tool: {item.name}({item.input})")
                                else:
                                    # Fallback to string representation
                                    print(str(item))
                        elif isinstance(content, dict):
                            print(json.dumps(content, indent=2))
                        else:
                            print(str(content))
            except Exception as e:
                print(f"Error parsing reasoning: {e}")
                print(reasoning_json)
        else:
            print("No reasoning recorded")

    return 0


def main(argv: list[str] | None = None, run_fn: Callable[..., SearchResult] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "run":
        fn = run_fn or sdk_run
        result = fn(
            task=Task.from_path(args.path, editable=args.editable),
            verifier=parse_verifier(args.verifier),
            model=args.model,
            workers=args.workers,
            episode=EpisodeBudget(edits=args.episode_edits, turns=args.episode_turns),
            run_budget=parse_run_budget(args.run_budget),
            sandbox=args.sandbox,
            image=args.image,
            db=args.db,
        )
        out = Path(args.out)
        artifact = result.solution or result.best_partial
        for rel, text in artifact.files.items():
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(text)
        if result.solution is not None:
            print(f"SOLVED — verified artifact written to {out} (run {result.run_id})")
            return 0
        print(f"UNSOLVED — best partial written to {out} (run {result.run_id})")
        return 2
    elif args.command == "reasoning":
        return show_reasoning(args)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
