"""Make example modules (harness, speed_verifier, ...) importable from unit tests.

Mirrors the sidon example's sys.path.insert(SCRIPT_DIR) pattern, centralized here
so each test file just does `from harness import ...` without repeating the path setup.
"""

import sys
from pathlib import Path

# conftest.py lives at Crucible/tests/unit/inference_speed/conftest.py
# parents[0]=inference_speed, [1]=unit, [2]=tests, [3]=Crucible (repo root)
_EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "inference_speed"
if str(_EXAMPLE) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE))
