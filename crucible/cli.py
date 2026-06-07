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
    return p


def main(argv: list[str] | None = None, run_fn: Callable[..., SearchResult] | None = None) -> int:
    args = build_parser().parse_args(argv)
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


if __name__ == "__main__":
    raise SystemExit(main())
