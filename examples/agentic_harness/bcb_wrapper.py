"""Thin verified wrapper over BigCodeBench: dataset load + hidden-test execution check.

BigCodeBench ships its own guarded execution (reliability_guard + time_limit + hidden
tests) — that IS the anti-reward-hacking core. This wrapper just exposes it cleanly.
"""
from __future__ import annotations

from bigcodebench.data import get_bigcodebench
from bigcodebench.eval import untrusted_check


def load_tasks(subset: str = "full") -> dict[str, dict]:
    """task_id -> task dict (subset: 'full' | 'hard'). Default 'full' (the 1140-task
    superset) so check_solution* can resolve any task_id regardless of which
    benchmark subset it was curated from. Caches after first load."""
    return get_bigcodebench(subset=subset)


def check_solution(
    task_id: str,
    solution_code: str,
    timeout: float = 10.0,
    max_as_limit: float = 30 * 1024,
    max_data_limit: float = 30 * 1024,
    max_stack_limit: float = 10,
) -> bool:
    """Run solution_code against the task's hidden tests in BigCodeBench's guarded env.

    Returns True only if the hidden tests pass. The guarded env (reliability_guard)
    prevents the solution from reading tests / monkeypatching the framework.

    Resource limits default to BigCodeBench's official values (30*1024 MB AS/data,
    10 MB stack) — passing 0 would prevent any allocation, making every solution fail.

    Real untrusted_check signature (discovered via inspect):
        untrusted_check(
            code: str,           # full source = code_prompt + solution body
            test_code: str,     # task['test']
            entry_point: str,    # task['entry_point']
            max_as_limit: float, max_data_limit: float, max_stack_limit: float,
            min_time_limit: float = 10, gt_time_limit: float = 60
        ) -> Tuple[str, dict]   # ('pass' | 'fail' | ..., {test_case: error_msg})

    Task dict keys: task_id, complete_prompt, instruct_prompt, canonical_solution,
                    code_prompt, test, entry_point, doc_struct, libs, q_idx, question,
                    score, _id
    """
    tasks = load_tasks()
    task = tasks[task_id]
    # solution_code is just the function body; prepend the prompt (imports + signature).
    full_code = task["code_prompt"] + solution_code
    result, _details = untrusted_check(
        code=full_code,
        test_code=task["test"],
        entry_point=task["entry_point"],
        max_as_limit=max_as_limit,
        max_data_limit=max_data_limit,
        max_stack_limit=max_stack_limit,
        min_time_limit=0.1,
        gt_time_limit=timeout,
    )
    return result == "pass"


def _count_tests(test_code: str) -> int:
    """Count individual test methods in a BigCodeBench test field.

    Every curated task uses the ``class TestCases(unittest.TestCase)`` structure
    with ``def test_*`` methods (verified across the 25-task subset: 5-6 each),
    so counting ``def test_`` occurrences is a reliable total. This total is what
    turns the binary pass/fail into a graded signal (partial pass rate), giving the
    harness search a gradient even when Tess's code fails some tests.
    """
    return test_code.count("def test_")


def check_solution_graded(
    task_id: str,
    solution_code: str,
    timeout: float = 10.0,
    max_as_limit: float = 30 * 1024,
    max_data_limit: float = 30 * 1024,
    max_stack_limit: float = 10,
) -> float:
    """Graded check: fraction of hidden tests that pass, in [0.0, 1.0].

    ``untrusted_check`` returns ``(result, {failing_test: error})`` — the dict only
    contains FAILED tests. So pass fraction = (total - failed) / total, where total
    is the ``def test_`` count. This is the verifier's graded signal: a candidate
    harness that makes Tess pass 5/6 tests scores 0.833 even when it fails the task,
    so the search has a gradient to climb (vs the flat-0 binary signal on Hard).
    """
    tasks = load_tasks()
    task = tasks[task_id]
    total = _count_tests(task["test"])
    if total == 0:
        return 0.0
    full_code = task["code_prompt"] + solution_code
    result, details = untrusted_check(
        code=full_code,
        test_code=task["test"],
        entry_point=task["entry_point"],
        max_as_limit=max_as_limit,
        max_data_limit=max_data_limit,
        max_stack_limit=max_stack_limit,
        min_time_limit=0.1,
        gt_time_limit=timeout,
    )
    if result == "pass":
        return 1.0
    failed = len(details or {})
    return max(0.0, (total - failed) / total)
