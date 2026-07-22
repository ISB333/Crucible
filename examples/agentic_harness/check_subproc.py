"""Run one BigCodeBench check_solution in a fresh, single-threaded subprocess.

Called by the verifier, which runs multi-threaded (the orchestrator dispatches workers
via asyncio.to_thread). BigCodeBench's untrusted_check uses os.fork internally; forking a
multi-threaded process that has the `filelock` library loaded trips Python 3.13's
fork-safety guard ("os.fork is unsafe while filelock is changing descriptor ownership").
This script is launched via subprocess.run (posix_spawn, not os.fork), so the verifier
process is never forked. The subprocess is single-threaded, so its internal fork is safe
(proven by measure_baseline, which ran 10 serial checks with no crash).

Protocol: read JSON {"task_id": str, "solution": str} on stdin, write JSON {"pass": bool}
on stdout.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bcb_wrapper import check_solution  # type: ignore[import-not-found]


def main() -> None:
    data = json.loads(sys.stdin.read())
    result = check_solution(data["task_id"], data["solution"])
    sys.stdout.write(json.dumps({"pass": bool(result)}))


if __name__ == "__main__":
    main()