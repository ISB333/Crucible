"""AgenticCodingVerifier — deterministic, stateless.

For a candidate harness: run it on the search subset (concurrent against the batched
Tess server), check each solution with BigCodeBench's hidden tests + the static gate,
score = pass rate. Ok when pass_rate >= baseline + threshold AND clean.
"""
from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from crucible.artifact import Artifact
from crucible.verify import Fail, Ok, Partial, RunContext, Scored, Verdict

_verify_lock = threading.Lock()


@dataclass(frozen=True)
class AgenticCodingVerifier:
    target_lift: float = 0.10            # +10 absolute over baseline -> Ok
    baseline_path: Path = Path("examples/agentic_harness/baseline.json")
    subset_path: Path = Path("examples/agentic_harness/tasks/subset.json")
    search_n: int = 10
    runner: "Callable[..., dict] | None" = None   # seam: (harness_module, subset, ws) -> {"pass":n,"n":N}
    deterministic: bool = True

    @property
    def verifier_id(self) -> str:
        return f"agentic:lift={self.target_lift}"

    def verify(self, artifact: Artifact, ctx: RunContext) -> Verdict:
        # 1. hole check on the editable solve region only (NOT generic scan_holes)
        if self._solve_has_hole(artifact):
            return Partial(open_holes=(), feedback="solve region unfilled (NotImplementedError)")
        ws = ctx.materialize(artifact)
        # 2. load the harness module the worker wrote
        try:
            harness_mod = self._load_harness(ws)
        except Exception as exc:
            return Fail(feedback=f"harness did not import: {exc!r}")
        # 3. run on the search subset (seam allows faking the 9B runs in tests)
        runner = self.runner or _locked_run_subset
        try:
            res = runner(harness_mod, self._search_subset(), ws)
        except Exception as exc:
            return Fail(feedback=f"verifier run crashed: {exc!r}")
        passed, n = res["pass"], res["n"]
        if n == 0:
            return Fail(feedback="no tasks ran")
        rate = passed / n
        baseline = json.loads(Path(self.baseline_path).read_text())["pass_rate"]
        if rate >= baseline + self.target_lift:
            return Ok(produced=artifact)
        return Scored(produced=artifact, value=rate,
                      feedback=f"pass_rate={rate:.3f} (baseline {baseline:.3f})")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _solve_has_hole(self, artifact: Artifact) -> bool:
        """Check the solve region for raise NotImplementedError (region-scoped only)."""
        try:
            region = artifact.region("solve")
            text = artifact.region_text(region)
        except Exception:
            return True
        return "raise NotImplementedError" in text

    def _load_harness(self, ws: Path):
        """Import the harness module from the materialized workspace.

        CRITICAL: harness.py does ``from agent_contract import Task, LLM, Tools``.
        For that import to resolve when exec'ing from the workspace, the workspace
        directory must be on sys.path.  We insert it before exec_module and remove
        it afterward — along with any modules imported from the workspace — to avoid
        polluting sys.modules for subsequent imports (e.g., _search_subset).
        """
        import importlib.util

        spec = importlib.util.spec_from_file_location("harness_cand", ws / "harness.py")
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(ws))
        before_keys = set(sys.modules)
        try:
            spec.loader.exec_module(mod)
        finally:
            if str(ws) in sys.path:
                sys.path.remove(str(ws))
            # Remove modules that were loaded from the workspace during exec_module
            # (e.g., agent_contract) so they don't shadow the real modules.
            for name in set(sys.modules) - before_keys:
                sys.modules.pop(name, None)
        return mod

    def _search_subset(self):
        """Load the search split of the curated subset (first search_n tasks)."""
        from agent_contract import load_subset

        tasks = load_subset(self.subset_path)
        return [t for t in tasks if t.id.startswith("search")][: self.search_n]


def _locked_run_subset(harness_mod, tasks, ws) -> dict:
    """Real runner: concurrent sandboxed solves against Tess + BigCodeBench check."""
    from concurrent.futures import ThreadPoolExecutor

    from agent_contract import LLM
    from bcb_wrapper import check_solution
    from reward_hacking_gate import is_clean
    from sandbox import run_solve_capped, sandbox_fresh_workdir

    llm = LLM(base_url="http://127.0.0.1:9090/v1", model="tess")

    def one(t):
        with _verify_lock:  # bound memory; tasks still batch on the Tess server
            wd = sandbox_fresh_workdir(t)
        run_solve_capped(harness_mod.solve, t, wd, llm,
                         max_turns=8, max_tokens=256, wall_s=180)
        try:
            sol = (wd / t.skeleton_path).read_text()
        except Exception:
            return 0
        return 1 if (is_clean(sol) and check_solution(t.eval_task_id, sol)) else 0

    with ThreadPoolExecutor(max_workers=min(12, len(tasks))) as pool:
        results = list(pool.map(one, tasks))
    return {"pass": sum(results), "n": len(results)}