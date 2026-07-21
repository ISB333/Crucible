"""Static reward-hacking gate (defense-in-depth on top of BigCodeBench's reliability_guard).

Rejects solutions that manipulate the test framework. BigCodeBench's guarded execution
already prevents reading hidden tests; this catches framework-level cheating a 9B might
stumble into and guarantees the verifier confronts the genuine solution.
"""
from __future__ import annotations

import re

_REJECT_PATTERNS = [
    r"\bunittest\.mock\b",
    r"\bmock\.patch\b",
    r"sys\.modules\s*\[",
    r"\b__import__\s*\(\s*['\"]unittest",
    r"\bdef\s+assertEqual\b",
    r"\bdef\s+assertTrue\b",
    r"\bdef\s+assert\b",
    r"\bunittest\.TestCase\b.*=\s*",
]
_REJECT = [re.compile(p) for p in _REJECT_PATTERNS]


def is_clean(solution_code: str) -> bool:
    if not solution_code or not solution_code.strip():
        return False
    return not any(rx.search(solution_code) for rx in _REJECT)
