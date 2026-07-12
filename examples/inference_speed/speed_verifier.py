"""SpeedQualityVerifier — deterministic, stateless (PRD §3 contract).

Grades a candidate inference config: runs the frozen harness, compares tok/s and
lossless quality against a frozen baseline, returns Scored(value=aggregate) /
Ok / Fail. Incumbent tracking and plateau detection are the orchestrator's job
(Scored ranking + plateau_patience), not this verifier's.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from check_draft_compat import tokenizers_compatible  # type: ignore[import-not-found]
from harness import (  # type: ignore[import-not-found]
    TARGET_MODEL,
    Config,
    lossless_match,
    run_harness,
)

from crucible.artifact import HOLE_SENTINEL, NOT_IMPLEMENTED_RE, Artifact, Hole
from crucible.verify import Fail, Ok, Partial, RunContext, Scored, Verdict


def _config_region_holes(artifact: Artifact) -> tuple[Hole, ...]:
    """Holes scoped to the editable `config` region only.

    The whole-artifact scan_holes() would also scan frozen docs (e.g. README.md
    documents the sentinel), producing a permanent false-positive PARTIAL that
    blocks every measurement. A hole in a frozen file isn't the worker's fault;
    only the config region is the worker's edit surface, so only it is checked.
    """
    try:
        region = artifact.region("config")
        text = artifact.region_text(region)
    except Exception:
        return ()
    holes: list[Hole] = []
    for i, line in enumerate(text.splitlines()):
        if HOLE_SENTINEL in line:
            holes.append(Hole(file="config.py", line=i, kind="sentinel", text=line.strip()))
        elif NOT_IMPLEMENTED_RE.search(line):
            holes.append(Hole(file="config.py", line=i, kind="not_implemented", text=line.strip()))
    return tuple(holes)


@dataclass(frozen=True)
class SpeedQualityVerifier:
    """Ok when agg>=target_agg and single>=target_single and quality-clean.

    Otherwise Scored(value=agg) when quality-clean but below target, or Fail.
    """

    target_agg: float = 30.0
    target_single: float = 8.0
    baseline_path: Path = Path("examples/inference_speed/baseline.json")
    runner: Callable[[Config, Path], dict] | None = None  # seam: (config, workspace) -> result dict
    config_loader: Callable[[Path], Config] | None = None  # seam: workspace -> Config
    compat_checker: Callable[[str, str], bool] | None = None  # seam: (target, draft) -> bool
    deterministic: bool = True

    @property
    def verifier_id(self) -> str:
        return f"speed:agg={self.target_agg},single={self.target_single}"

    def verify(self, artifact: Artifact, ctx: RunContext) -> Verdict:
        holes = _config_region_holes(artifact)
        if holes:
            return Partial(
                open_holes=holes, feedback="config region contains a hole sentinel — not filled"
            )

        ws = ctx.materialize(artifact)
        baseline = json.loads(Path(self.baseline_path).read_text())

        config_loader = self.config_loader or _load_config_from_workspace
        try:
            config = config_loader(ws)
        except Exception as exc:
            return Fail(feedback=f"config region did not produce a valid Config: {exc!r}")

        runner = self.runner or _locked_run_harness

        # Speculative-decoding integrity: a draft must share the target tokenizer,
        # else accepted tokens would not match what the target would emit (quality loss).
        if config.draft_model:
            compat = (
                self.compat_checker(TARGET_MODEL, config.draft_model)
                if self.compat_checker
                else tokenizers_compatible(TARGET_MODEL, config.draft_model)
            )
            if not compat:
                return Fail(
                    feedback=(
                        "draft tokenizer incompatible with target — "
                        "spec decoding would mis-accept tokens; "
                        f"draft={config.draft_model}"
                    )
                )

        try:
            result = runner(config, ws)
        except Exception as exc:  # crashed launch / measurement
            return Fail(feedback=f"harness crashed: {exc!r}")

        if "error" in result:
            return Fail(feedback=f"harness error: {result['error']}")

        # Integrity: the loaded model must be the fixed target (don't trust config).
        if result.get("loaded_model") != TARGET_MODEL:
            return Fail(
                feedback=(
                    f"loaded model {result.get('loaded_model')!r} != target {TARGET_MODEL!r}; "
                    "swapping the target model is a gaming vector "
                    "(editing the measure, not the speed)."
                )
            )

        # Lossless quality gate: candidate probe outputs vs frozen baseline reference.
        ok, mism = lossless_match(result.get("probe_outputs", {}), baseline["probe_reference"])
        if not ok:
            return Fail(
                feedback=(
                    f"lossless gate FAILED — probe outputs diverge "
                    f"from greedy reference on {mism}. "
                    "The optimization changed the output, i.e. degraded quality."
                )
            )

        agg = float(result["aggregate"]["tok_s"])
        single = float(result["single_stream"]["tok_s"])

        feedback = (
            f"single={single:.2f} tok/s  aggregate={agg:.2f} tok/s  "
            f"(targets: single>={self.target_single}, agg>={self.target_agg})  "
            f"quality=lossless-clean  baseline: single={baseline['single_stream']}, "
            f"agg={baseline['aggregate']}."
        )

        if agg >= self.target_agg and single >= self.target_single:
            return Ok(produced=artifact)
        return Scored(produced=artifact, value=agg, feedback=feedback)


def _load_config_from_workspace(ws: Path) -> Config:
    """Import the artifact's config.py (which defines CONFIG) from the materialized workspace."""
    import importlib.util
    import sys

    ws_str = str(ws)
    if ws_str not in sys.path:
        sys.path.insert(0, ws_str)
    spec = importlib.util.spec_from_file_location("candidate_config", ws / "config.py")
    assert spec is not None and spec.loader is not None, f"could not load config.py from {ws}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CONFIG


# RAM safety: workers run in threads (orchestrator uses asyncio.to_thread). Without
# serialization, N workers verifying concurrently would spawn N 9B servers (~3 GB each)
# and OOM the VPS. The lock makes only the local 9B measurement serial; Gemini LLM turns
# (the bulk of episode wall time) still parallelize across workers.
_verify_lock = threading.Lock()


def _locked_run_harness(cfg: Config, workspace: Path) -> dict:
    with _verify_lock:
        return run_harness(cfg, workspace)
