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

from crucible import AdvisorPolicy, run as sdk_run
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
    # LLM shepherding / advisor flags
    r.add_argument("--advisor", default=None, help="advisor model name (e.g. claude-opus-4-8)")
    r.add_argument("--advisor-max-calls", type=int, default=None, help="max advisor calls per run")
    r.add_argument("--advisor-fail-streak", type=int, default=3, help="trigger advisor after N non-improving episodes")

    # List runs command
    ls = sub.add_parser("runs", help="list runs stored in the database")
    ls.add_argument("--db", default="crucible.db", help="database file to read from")

    # Show reasoning command
    s = sub.add_parser("reasoning", help="show LLM reasoning for a run")
    s.add_argument("run_id", type=int, nargs="?", help="run ID (default: latest run)")
    s.add_argument("--db", default="crucible.db", help="database file to read from")
    s.add_argument("--worker", type=int, help="filter to a specific worker index")
    s.add_argument("--episode", type=int, help="filter to a specific episode number")

    return p


def _render_message(msg: dict) -> None:
    """Print one message from a stored (plain-dict) conversation."""
    import json as _json

    role = msg.get("role", "?")
    # Anthropic / OpenAI format: "content" key
    if "content" in msg:
        content = msg["content"]
        if isinstance(content, str):
            print(f"[{role}] {content}")
        elif isinstance(content, list):
            print(f"[{role}]")
            for item in content:
                if not isinstance(item, dict):
                    print(f"  {item}")
                    continue
                t = item.get("type")
                if t == "text":
                    print(f"  {item.get('text', '')}")
                elif t == "tool_use":
                    print(f"  → {item.get('name')}({_json.dumps(item.get('input', {}), ensure_ascii=False)})")
                elif t == "tool_result":
                    print(f"  ← {item.get('content', '')}")
                else:
                    # OpenAI tool call wrapper or unknown
                    print(f"  {_json.dumps(item, ensure_ascii=False)}")
        else:
            print(f"[{role}] {content}")
    # Gemini format: "parts" key
    elif "parts" in msg:
        print(f"[{role}]")
        for part in msg["parts"]:
            if not isinstance(part, dict):
                print(f"  {part}")
                continue
            if "text" in part:
                print(f"  {part['text']}")
            elif "function_call" in part:
                fc = part["function_call"]
                print(f"  → {fc.get('name')}({_json.dumps(fc.get('args', {}), ensure_ascii=False)})")
            elif "function_response" in part:
                fr = part["function_response"]
                output = fr.get("response", {}).get("output", str(fr))
                print(f"  ← {output}")
    else:
        import json as _json
        print(f"[{role}] {_json.dumps(msg, ensure_ascii=False)}")


def _require_db(path: str) -> bool:
    from pathlib import Path
    if not Path(path).exists():
        print(f"Error: database file not found: {path}")
        return False
    return True


def list_runs(args) -> int:
    import datetime
    from crucible.store import Store

    if not _require_db(args.db):
        return 1
    store = Store(args.db)
    rows = store._conn.execute(
        "SELECT id, started_at, task_root, model FROM runs ORDER BY id DESC LIMIT 50"
    ).fetchall()
    if not rows:
        print("No runs found.")
        return 0
    print(f"{'ID':>4}  {'Started':>20}  {'Model':<30}  Task")
    print("-" * 80)
    for run_id, started_at, task_root, model in rows:
        ts = datetime.datetime.fromtimestamp(started_at).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{run_id:>4}  {ts:>20}  {model:<30}  {task_root}")
    return 0


def show_reasoning(args) -> int:
    import json
    from crucible.store import Store

    if not _require_db(args.db):
        return 1
    store = Store(args.db)
    conn = store._conn

    # Resolve run_id: use provided value or fall back to latest run
    run_id = args.run_id
    if run_id is None:
        row = conn.execute("SELECT id FROM runs ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            print("No runs found in database.")
            return 1
        run_id = row[0]
        print(f"(Using latest run: {run_id})")

    query = """
        SELECT w.idx, e.ordinal, e.reasoning_json
        FROM workers w
        JOIN episodes e ON w.id = e.worker_id
        WHERE w.run_id = ?
    """
    params: list[object] = [run_id]
    if args.worker is not None:
        query += " AND w.idx = ?"
        params.append(args.worker)
    if args.episode is not None:
        query += " AND e.ordinal = ?"
        params.append(args.episode)
    query += " ORDER BY w.idx, e.ordinal"

    rows = conn.execute(query, params).fetchall()
    if not rows:
        print(f"No reasoning found for run {run_id}.")
        return 1

    for worker_idx, episode, reasoning_json in rows:
        print(f"\n{'='*60}")
        print(f"Worker {worker_idx} — Episode {episode}")
        print("=" * 60)
        if not reasoning_json:
            print("(no reasoning recorded)")
            continue
        try:
            messages = json.loads(reasoning_json)
            for msg in messages:
                _render_message(msg)
        except Exception as exc:
            print(f"(parse error: {exc})")
            print(reasoning_json[:500])

    return 0


def main(argv: list[str] | None = None, run_fn: Callable[..., SearchResult] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "run":
        fn = run_fn or sdk_run
        # Build advisor policy from CLI flags
        advisor = None
        if args.advisor is not None:
            advisor = AdvisorPolicy(
                model=args.advisor,
                max_calls_per_run=args.advisor_max_calls,
                fail_streak=args.advisor_fail_streak,
            )
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
            advisor=advisor,
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
    elif args.command == "runs":
        return list_runs(args)
    elif args.command == "reasoning":
        return show_reasoning(args)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
