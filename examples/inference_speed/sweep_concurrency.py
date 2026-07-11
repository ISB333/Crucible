"""Sweep n_concurrent to characterize the batching scaling ceiling (no Gemini spend).

Maps how aggregate tok/s scales with concurrency, to show whether the ~5 tok/s
plateau is bandwidth-ceiling-limited (aggregate flattens as n_concurrent grows)
or concurrency-limited (aggregate keeps rising). Free: only loads the 9B.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from harness import Config, run_harness, wait_for_ready  # noqa: E402


def main() -> int:
    print(f"{'n_conc':>6} {'single':>8} {'aggregate':>10} {'lossless':>9}")
    print("-" * 40)
    for n in [1, 2, 4, 8, 12]:
        cfg = Config(n_concurrent=n)
        r = run_harness(
            cfg,
            SCRIPT_DIR,
            max_tokens=32,
            waiter=lambda url, timeout_s=600: wait_for_ready(url, timeout_s),
        )
        if "error" in r:
            print(f"{n:>6} ERROR: {r['error']}")
            continue
        single = r["single_stream"]["tok_s"]
        agg = r["aggregate"]["tok_s"]
        # lossless = probe outputs non-empty (real measurement)
        ok = all(r["probe_outputs"].values())
        print(f"{n:>6} {single:>8.2f} {agg:>10.2f} {str(ok):>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
