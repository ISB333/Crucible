import pytest

from crucible.budgets import EpisodeBudget, RunBudget, budgets, parse_duration

pytestmark = pytest.mark.unit


def test_defaults_match_prd() -> None:
    assert EpisodeBudget() == EpisodeBudget(edits=90, turns=40)
    rb = RunBudget()
    assert rb.wall_clock_s == 7200.0 and rb.usd is None and rb.episodes_per_worker == 8


def test_parse_duration() -> None:
    assert parse_duration("2h") == 7200.0
    assert parse_duration("30m") == 1800.0
    assert parse_duration("45s") == 45.0
    with pytest.raises(ValueError):
        parse_duration("2 fortnights")


def test_budgets_helper_dispatches_on_keys() -> None:
    assert budgets(edits=90, turns=40) == EpisodeBudget(edits=90, turns=40)
    rb = budgets(wall_clock="2h", usd=200)
    assert rb == RunBudget(wall_clock_s=7200.0, usd=200.0)


def test_budgets_no_args_returns_run_budget() -> None:
    assert budgets() == RunBudget()


def test_budgets_helper_rejects_mixed_keys() -> None:
    with pytest.raises(ValueError):
        budgets(edits=10, usd=5)  # type: ignore[call-overload]


def test_budgets_helper_respects_explicit_zero() -> None:
    assert budgets(edits=0, turns=0) == EpisodeBudget(edits=0, turns=0)
    rb = budgets(wall_clock=0.0, episodes=0)
    assert rb == RunBudget(wall_clock_s=0.0, episodes_per_worker=0)
