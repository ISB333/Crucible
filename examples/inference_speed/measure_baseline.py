"""Wave 0: measure the honest baseline with the default Config and freeze it.

Usage:
  uv run python examples/inference_speed/measure_baseline.py [--out baseline.json]

Writes {single_stream, aggregate, probe_reference, config}. probe_reference is the
greedy probe output set — the lossless reference every future candidate must match.
No Gemini spend; only loads the 9B.
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from harness import Config, run_harness, wait_for_ready  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SCRIPT_DIR / "baseline.json"))
    ap.add_argument(
        "--max-tokens", type=int, default=64, help="tokens per prompt for the measurement"
    )
    ap.add_argument(
        "--ready-timeout", type=int, default=600, help="seconds to wait for the 9B to load"
    )
    args = ap.parse_args()

    cfg = Config()  # the frozen default
    result = run_harness(
        cfg,
        SCRIPT_DIR,
        max_tokens=args.max_tokens,
        waiter=lambda url, timeout_s=args.ready_timeout: wait_for_ready(url, timeout_s),
    )
    if "error" in result:
        print(f"harness error: {result['error']}", file=sys.stderr)
        return 1

    baseline = {
        "single_stream": float(result["single_stream"]["tok_s"]),
        "aggregate": float(result["aggregate"]["tok_s"]),
        "probe_reference": result["probe_outputs"],
        "config": result["config"],
    }
    Path(args.out).write_text(json.dumps(baseline, indent=2, ensure_ascii=False))
    print(f"baseline written to {args.out}")
    print(f"  single_stream = {baseline['single_stream']:.2f} tok/s")
    print(f"  aggregate     = {baseline['aggregate']:.2f} tok/s")
    print(f"  probes        = {len(baseline['probe_reference'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
