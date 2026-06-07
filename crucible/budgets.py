"""Budgets (PRD §5): exceeding any budget ends gracefully with best_partial."""

import re
from dataclasses import dataclass
from typing import overload


@dataclass(frozen=True)
class EpisodeBudget:
    edits: int = 90  # paper used ~90 edits/episode
    turns: int = 40


@dataclass(frozen=True)
class RunBudget:
    wall_clock_s: float = 7200.0
    usd: float | None = None  # None = no USD cap (cost may be unknown for local models)
    episodes_per_worker: int = 8
    plateau_patience: int | None = None  # stop a worker after N episodes with no rank improvement


_DURATION_RE = re.compile(r"^(\d+(?:\.\d+)?)([hms])$")
_UNIT_S = {"h": 3600.0, "m": 60.0, "s": 1.0}


def parse_duration(text: str) -> float:
    m = _DURATION_RE.match(text.strip())
    if m is None:
        raise ValueError(f"bad duration {text!r}; use e.g. '2h', '30m', '45s'")
    return float(m.group(1)) * _UNIT_S[m.group(2)]


@overload
def budgets() -> RunBudget: ...


@overload
def budgets(*, edits: int, turns: int | None = ...) -> EpisodeBudget: ...


@overload
def budgets(*, edits: int | None = ..., turns: int) -> EpisodeBudget: ...


@overload
def budgets(
    *, wall_clock: str | float | None = ..., usd: float | None = ..., episodes: int | None = ...
) -> RunBudget: ...


def budgets(
    *,
    edits: int | None = None,
    turns: int | None = None,
    wall_clock: str | float | None = None,
    usd: float | None = None,
    episodes: int | None = None,
) -> EpisodeBudget | RunBudget:
    """SDK helper (PRD §9): episode keys and run keys must not be mixed.

    No keys -> default RunBudget.
    """
    episode_keys = edits is not None or turns is not None
    run_keys = wall_clock is not None or usd is not None or episodes is not None
    if episode_keys and run_keys:
        raise ValueError("mixed episode keys (edits/turns) with run keys (wall_clock/usd/episodes)")
    if episode_keys:
        base = EpisodeBudget()
        return EpisodeBudget(
            edits=edits if edits is not None else base.edits,
            turns=turns if turns is not None else base.turns,
        )
    base = RunBudget()
    wall = (
        parse_duration(wall_clock)
        if isinstance(wall_clock, str)
        else (wall_clock if wall_clock is not None else base.wall_clock_s)
    )
    return RunBudget(
        wall_clock_s=wall,
        usd=float(usd) if usd is not None else None,
        episodes_per_worker=episodes if episodes is not None else base.episodes_per_worker,
    )
