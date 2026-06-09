"""The Ralph loop (PRD §5): episodes of multi-turn editing, verify after every edit."""

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from crucible.artifact import Artifact, scan_holes
from crucible.advisor import Advisor, AdvisorPolicy, AdvisorRequest
from crucible.budgets import EpisodeBudget, RunBudget
from crucible.edits import search_replace, write_region
from crucible.integrity import IntegrityCheck
from crucible.llm import LLMSession, ToolCall, ToolResult
from crucible.store import Store
from crucible.verify import Fail, Ok, Partial, RunContext, Scored, Verdict, Verifier, render

SYSTEM_PROMPT = """\
You are a Crucible worker. You iterate on an artifact until the verifier accepts it.

Rules:
- You may ONLY change text inside the editable regions (named below). Edits anywhere
  else are rejected by the tool.
- Use search_replace for targeted edits, write_region to rewrite a whole region.
- After every applied edit you receive the verifier's verdict. FAIL/PARTIAL feedback
  is your gradient: read it and fix the cause.
- A solution must have zero holes (no NotImplementedError, no crucible:hole, no sorry).
- Do not game the verifier: no skips, no xfail, no mocking the system under test,
  no `assert True`, no axioms. Tampering is detected and the attempt is rejected.
- Use record_lesson to note insights worth keeping for the next attempt.
- Stop calling tools when the verdict is OK or you cannot make progress.
"""


def initial_prompt(artifact: Artifact, verdict: Verdict) -> str:
    parts = [f"=== {path} ===\n{artifact.files[path]}" for path in sorted(artifact.files)]
    names = ", ".join(r.name for r in artifact.regions) or "(none)"
    parts.append(f"Editable regions: {names}")
    parts.append("Current verdict:\n" + render(verdict))
    return "\n\n".join(parts)


@dataclass(frozen=True)
class EpisodeOutcome:
    artifact: Artifact
    solved: bool
    turns: int
    edits: int
    end_reason: str  # solved | model_stopped | turn_budget | edit_budget | integrity_violation
    lessons: str
    cost_usd: float
    final_verdict: Verdict
    advisor_calls: int = 0
    advisor_cost_usd: float = 0.0
    advisor_records: tuple[dict, ...] = ()


def _handle_call(
    call: ToolCall,
    artifact: Artifact,
    verifier: Verifier,
    ctx: RunContext,
    lessons: list[str],
    consult: "Callable[[str, str], str] | None",
) -> tuple[ToolResult, Artifact, bool, Verdict | None]:
    # Validate model output before acting on it: coerce args defensively.
    if call.name == "record_lesson":
        lessons.append(str(call.args.get("text", "")).strip())
        return ToolResult(call.id, "lesson recorded"), artifact, False, None
    if call.name == "consult_advisor":
        if consult is None:
            return ToolResult(call.id, "advisor not enabled"), artifact, False, None
        advice = consult(str(call.args.get("question", "")), "self")
        return ToolResult(call.id, f"[ADVISOR]: {advice}"), artifact, False, None
    if call.name == "search_replace":
        er = search_replace(
            artifact,
            str(call.args.get("file", "")),
            str(call.args.get("old", "")),
            str(call.args.get("new", "")),
        )
    elif call.name == "write_region":
        er = write_region(
            artifact, str(call.args.get("name", "")), str(call.args.get("content", ""))
        )
    else:
        return ToolResult(call.id, f"unknown tool {call.name!r}"), artifact, False, None
    if not er.applied:
        return ToolResult(call.id, f"EDIT REJECTED: {er.error}"), artifact, False, None
    verdict = verifier.verify(er.artifact, ctx)  # verify after every edit (PRD §5)
    return ToolResult(call.id, render(verdict)), er.artifact, True, verdict


def run_episode(
    artifact: Artifact,
    verifier: Verifier,
    ctx: RunContext,
    session: LLMSession,
    budget: EpisodeBudget,
    integrity: IntegrityCheck,
    advisor: "Advisor | None" = None,
    policy: "AdvisorPolicy | None" = None,
    advisor_cap: int = 0,
) -> tuple[EpisodeOutcome, LLMSession]:
    start_artifact = artifact
    lessons: list[str] = []
    edits = 0
    end_reason = "model_stopped"
    last_verdict = verifier.verify(artifact, ctx)
    best_rank = verdict_rank(last_verdict)
    streak = 0
    advisor_records: list[dict] = []
    consult_used = 0

    def consult(question: str, trigger: str) -> str:
        nonlocal consult_used
        if advisor is None or policy is None or consult_used >= advisor_cap:
            return "(advisor budget exhausted)"
        req = AdvisorRequest(
            files=dict(artifact.files),
            editable_regions=[r.name for r in artifact.regions],
            verdict_text=render(last_verdict),
            recent_lessons=list(lessons),
            question=question,
            trigger=trigger,
            scope=policy.scope,
        )
        resp = advisor.consult(req)
        consult_used += 1
        advisor_records.append(
            {"trigger": trigger, "advice": resp.advice, "cost_usd": resp.cost_usd, "ok": resp.ok}
        )
        return resp.advice

    use_consult = consult if (advisor is not None and policy is not None) else None
    calls = session.start(SYSTEM_PROMPT, initial_prompt(artifact, last_verdict))
    turns = 1
    while calls:
        if turns >= budget.turns:
            end_reason = "turn_budget"
            break
        results: list[ToolResult] = []
        last_applied_idx: int | None = None
        for call in calls:
            result, artifact, applied, verdict = _handle_call(
                call, artifact, verifier, ctx, lessons, use_consult
            )
            results.append(result)
            if applied and verdict is not None:
                last_applied_idx = len(results) - 1
                last_verdict = verdict
                rank = verdict_rank(verdict)
                if rank > best_rank:
                    best_rank, streak = rank, 0
                else:
                    streak += 1
            edits += int(applied)
            if edits >= budget.edits:
                break
        # engine-trigger: nudge once when the worker is plateauing
        if (
            policy is not None
            and policy.plateau_trigger
            and streak >= policy.fail_streak
            and consult_used < advisor_cap
            and last_applied_idx is not None
        ):
            advice = consult("", "engine")
            r = results[last_applied_idx]
            results[last_applied_idx] = ToolResult(r.call_id, f"{r.content}\n\n[ADVISOR]: {advice}")
            streak = 0
        if edits >= budget.edits:
            end_reason = "edit_budget"
            break
        calls = session.reply(results)
        turns += 1
    final = verifier.verify(artifact, ctx)
    clean = integrity.check(artifact, ctx)
    solved = clean and isinstance(final, Ok) and not scan_holes(artifact) and verifier.deterministic
    if not clean:
        artifact = start_artifact  # revert to last good (PRD §5/§7)
        end_reason = "integrity_violation"
    if solved:
        end_reason = "solved"
    advisor_cost = sum(r["cost_usd"] for r in advisor_records)
    outcome = EpisodeOutcome(
        artifact=artifact,
        solved=solved,
        turns=turns,
        edits=edits,
        end_reason=end_reason,
        lessons="\n".join(line for line in lessons if line),
        cost_usd=session.cost_usd + advisor_cost,
        final_verdict=final,
        advisor_calls=consult_used,
        advisor_cost_usd=advisor_cost,
        advisor_records=tuple(advisor_records),
    )
    return outcome, session


def verdict_rank(verdict: Verdict) -> tuple[int, float]:
    """Comparable progress key for a single verdict. Higher is better:
    Ok > Scored(value) > Partial(-holes) > Fail."""
    match verdict:
        case Ok():
            return (3, 0.0)
        case Scored(value=value):
            return (2, value)
        case Partial(open_holes=holes):
            return (1, -float(len(holes)))
        case Fail():
            return (0, 0.0)


def outcome_rank(outcome: EpisodeOutcome) -> tuple[int, float]:
    """A comparable progress key. Higher is better. Tiers keep families from mixing scales:
    Ok > Scored(value) > Partial(-holes) > Fail. A reverted (integrity-violating) episode
    ranks at the bottom — its kept artifact is the pre-edit start."""
    if outcome.end_reason == "integrity_violation":
        return (0, 0.0)
    return verdict_rank(outcome.final_verdict)


_COMMENT_PREFIX = {".py": "#", ".lean": "--", ".sh": "#", ".md": ">"}


def _sanitize_lesson(line: str) -> str:
    """Lesson text is model-controlled: neutralize tokens that the region parser
    or hole scanner would misread inside annotation comments."""
    return (
        line.replace("crucible:region", "crucible_region")
        .replace("crucible:hole", "crucible_hole")
        .replace("NotImplementedError", "NIE")
    )


def annotate_lessons(artifact: Artifact, episode_no: int, lessons: str) -> Artifact:
    """Write lessons into the artifact (PRD §5) — inside the first editable region,
    because anywhere else would trip the immutable-region byte-check."""
    if not lessons.strip() or not artifact.regions:
        return artifact
    r = artifact.regions[0]
    prefix = _COMMENT_PREFIX.get(Path(r.file).suffix, "#")
    block = "\n".join(
        f"{prefix} lesson(ep{episode_no}): {_sanitize_lesson(line)}"
        for line in lessons.splitlines()
        if line.strip()
    )
    return artifact.replace_region(r.name, artifact.region_text(r) + "\n" + block)


@dataclass(frozen=True)
class WorkerResult:
    index: int
    solution: Artifact | None
    best_partial: Artifact
    episodes: int
    cost_usd: float
    best_rank: tuple[int, float]


def run_worker(
    *,
    initial: Artifact,
    verifier: Verifier,
    ctx: RunContext,
    new_session: Callable[[int], LLMSession],  # episode ordinal -> fresh session
    store: Store,
    run_id: int,
    index: int,
    episode_budget: EpisodeBudget,
    run_budget: RunBudget,
    integrity: IntegrityCheck,
    cancel: threading.Event,
    started_at: float,
) -> WorkerResult:
    """One independent Ralph loop. Standalone callable by design: this is the seam
    P1.3 plugs into the reactor's _transform in v2."""
    worker_id = store.add_worker(run_id, index)
    artifact = initial
    best_rank: tuple[int, float] = (-1, 0.0)  # sentinel: below every real outcome
    best = initial
    stale = 0
    cost = 0.0
    episodes = 0
    for ordinal in range(run_budget.episodes_per_worker):
        if cancel.is_set():
            break
        if time.time() - started_at > run_budget.wall_clock_s:
            break
        if run_budget.usd is not None and cost >= run_budget.usd:
            break
        t0 = time.time()
        out, session = run_episode(artifact, verifier, ctx, new_session(ordinal), episode_budget, integrity)
        episodes += 1
        cost += out.cost_usd
        # Capture and store LLM reasoning
        reasoning = json.dumps(session.messages, default=str)
        ep_id = store.add_episode(
            worker_id,
            ordinal=ordinal,
            turns=out.turns,
            edits=out.edits,
            end_reason=out.end_reason,
            lessons=out.lessons,
            reasoning=reasoning,
        )
        holes = len(scan_holes(out.artifact))
        score = out.final_verdict.value if isinstance(out.final_verdict, Scored) else None
        av_id = store.add_artifact_version(
            ep_id,
            out.artifact,
            holes_remaining=holes,
            cost_usd=out.cost_usd,
            wall_time_s=time.time() - t0,
            integrity_ok=out.end_reason != "integrity_violation",
            score=score,
        )
        store.add_verdict(av_id, out.final_verdict)
        if out.solved:
            return WorkerResult(index, out.artifact, out.artifact, episodes, cost, (3, 0.0))
        rank = outcome_rank(out)
        if rank > best_rank:
            best_rank, best, stale = rank, out.artifact, 0
        else:
            stale += 1
        artifact = annotate_lessons(out.artifact, ordinal, out.lessons)
        if run_budget.plateau_patience is not None and stale >= run_budget.plateau_patience:
            break
    return WorkerResult(index, None, best, episodes, cost, best_rank)
