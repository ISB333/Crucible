def test_load_tasks_returns_dict():
    from bcb_wrapper import load_tasks

    tasks = load_tasks(subset="hard")
    assert isinstance(tasks, dict) and len(tasks) > 50


def test_check_known_bad_fails():
    from bcb_wrapper import load_tasks, check_solution

    tasks = load_tasks(subset="hard")
    tid = next(iter(tasks))
    # A trivially-bad solution must fail the hidden tests.
    assert check_solution(tid, "def foo():\n    pass\n") is False