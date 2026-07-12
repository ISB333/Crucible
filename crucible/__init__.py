"""Crucible — verifier-grounded multi-agent search engine (v0).

PRD §9 surface:

    from crucible import Task, run, budgets
    from crucible.verifiers import Pytest

    result = run(
        task=Task.from_path("problem.py", editable=["solution"]),
        verifier=Pytest(suite="tests/"),
        model="claude-sonnet-4-6",
        workers=10,
        episode=budgets(edits=90, turns=40),
        run_budget=budgets(wall_clock="2h", usd=200),
    )
    result.solution      # Artifact | None — hole-free, verified, integrity-clean
    result.best_partial  # Artifact — highest-progress attempt if unsolved
    result.run_id        # provenance key into the append-only store

NOTE (documented deviation from PRD sketch): PRD §9 named the second budget kwarg
``run``; that would shadow the ``run()`` function, so the SDK uses ``run_budget``.
"""

from collections.abc import Callable
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from crucible.advisor import Advisor, AdvisorPolicy
from crucible.artifact import Artifact
from crucible.budgets import EpisodeBudget, RunBudget, budgets
from crucible.llm import LLMSession
from crucible.orchestrator import SearchResult as Result
from crucible.orchestrator import run_search
from crucible.sandbox import DockerSandbox, Sandbox, SubprocessSandbox
from crucible.store import Store
from crucible.task import Task
from crucible.verify import Verifier

__version__ = "0.0.1"
__all__ = ["Artifact", "Result", "Task", "AdvisorPolicy", "budgets", "run", "__version__"]


def run(
    *,
    task: Task,
    verifier: Verifier,
    model: str,
    workers: int = 10,
    episode: EpisodeBudget | None = None,
    run_budget: RunBudget | None = None,
    sandbox: str = "docker",  # "docker" | "subprocess"
    image: str | None = None,  # docker image override (e.g. "crucible-lean:0")
    base_url: str | None = None,  # OpenAI-compatible endpoint for local models
    db: str | Path = "crucible.db",
    session_factory: Callable[[int, int], LLMSession] | None = None,  # test seam
    advisor: str | AdvisorPolicy | None = None,  # LLM shepherding: model name or policy
    extra_tools: "list[dict] | None" = None,  # extra tool schemas exposed to the worker
    tool_handlers: "dict[str, Callable[[dict], str]] | None" = None,  # their execution handlers
    advisor_factory: "Callable[[], Advisor] | None" = None,  # override the policy-built advisor
) -> Result:
    """Run a verifier-grounded multi-agent search (PRD §9).

    Raises ValueError if:
    - ``verifier.deterministic`` is False (advisory-only verifier cannot be the
      sole accept signal in v0 — PRD §3).
    - ``sandbox`` is not ``"docker"`` or ``"subprocess"``.

    Args:
        advisor: Optional LLM shepherding config. If str, interpreted as model name
            with default policy (max_calls_per_episode=1, plateau_trigger=True, fail_streak=3).
            If AdvisorPolicy, used as-is. If None (default), advisor is disabled.
    """
    if not verifier.deterministic:
        raise ValueError(
            "advisory verifier (deterministic=False) cannot be the sole verifier in v0 —"
            " it can never be the accept signal (PRD §3)"
        )

    factory: Callable[[], Sandbox]
    if sandbox == "docker":
        _image = image or "crucible-py:0"

        def factory() -> Sandbox:  # noqa: E306
            return DockerSandbox(image=_image)

    elif sandbox == "subprocess":

        def factory() -> Sandbox:  # noqa: E306
            return SubprocessSandbox()

    else:
        raise ValueError(f"unknown sandbox kind {sandbox!r}; use 'docker' or 'subprocess'")

    # Normalize advisor config
    advisor_policy: AdvisorPolicy | None = None
    if isinstance(advisor, str):
        advisor_policy = AdvisorPolicy(model=advisor)
    elif isinstance(advisor, AdvisorPolicy):
        advisor_policy = advisor
    elif advisor is not None:
        raise ValueError(f"advisor must be str, AdvisorPolicy, or None, got {type(advisor).__name__}")

    # Build advisor factory from policy if the caller didn't supply one
    eff_advisor_factory = advisor_factory
    if eff_advisor_factory is None and advisor_policy is not None:
        def _adv_factory(_pol=advisor_policy, _bu=base_url) -> Advisor:
            from crucible.advisor import LLMAdvisor
            return LLMAdvisor(policy=_pol, base_url=_bu)
        eff_advisor_factory = _adv_factory

    sf: Callable[[int, int], LLMSession]
    if session_factory is not None:
        sf = session_factory
    else:
        from crucible.providers import make_session  # optional extras imported lazily

        _base_url = base_url
        _model = model

        def _make(*_: int) -> LLMSession:  # type: ignore[unused-argument]
            return make_session(_model, base_url=_base_url, extra_tools=extra_tools or ())

        sf = _make

    return run_search(
        task=task,
        verifier=verifier,
        session_factory=sf,
        store=Store(db),
        sandbox_factory=factory,
        model=model,
        workers=workers,
        episode_budget=episode,
        run_budget=run_budget,
        advisor_factory=eff_advisor_factory,
        advisor_policy=advisor_policy,
        tool_handlers=tool_handlers,
    )
