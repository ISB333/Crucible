"""The agent's harness. The worker edits ONLY the `solve` body (the crucible:region).

Wave 0 baseline: one-shot — read the spec, ask Tess to write ONLY the function body
(indented Python, no signature, no explanation), write the body to skeleton.py.
"""
from __future__ import annotations

from pathlib import Path

from agent_contract import Task, LLM, Tools


# crucible:region start name=solve
def solve(task: Task, workdir: Path, llm: LLM, tools: Tools) -> None:
    """Minimal baseline: ask Tess to complete the skeleton in one shot.

    The contract is:
    - spec.md contains code_prompt (imports + function signature) — what the agent sees.
    - skeleton.py contains the function body — what the agent writes.
    - check_solution(task_id, body) prepends code_prompt internally.
    So we ask Tess for ONLY the indented function body, no signature, no fences.
    """
    spec = task.spec  # code_prompt: imports + signature + docstring
    prompt = (
        "Complete this Python function. The signature is:\n\n"
        f"{spec}\n\n"
        "Output ONLY the function body (indented Python), "
        "no signature, no explanation, no markdown fences."
    )
    code = llm.chat([{"role": "user", "content": prompt}], max_tokens=512)
    # Strip markdown fences if the model wraps them despite instructions
    lines = code.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    body = "\n".join(lines) + "\n"
    tools.write_file(str(task.skeleton_path), body)
# crucible:region end