def test_load_tasks_returns_dict():
    from bcb_wrapper import load_tasks  # type: ignore[import-not-found]

    tasks = load_tasks(subset="hard")
    assert isinstance(tasks, dict) and len(tasks) > 50


def test_check_known_bad_fails():
    from bcb_wrapper import check_solution, load_tasks  # type: ignore[import-not-found]

    tasks = load_tasks(subset="hard")
    tid = next(iter(tasks))
    # A trivially-bad solution must fail the hidden tests.
    assert check_solution(tid, "def foo():\n    pass\n") is False


def test_check_known_good_passes():
    from bcb_wrapper import check_solution, load_tasks  # type: ignore[import-not-found]

    tasks = load_tasks(subset="hard")
    tid = next(iter(tasks))
    # The canonical (reference) solution must pass the hidden tests.
    assert check_solution(tid, tasks[tid]["canonical_solution"]) is True