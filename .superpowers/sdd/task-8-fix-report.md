# Task 8 Fix Report: def solve Outside Editable Region

## Bug

The `def solve(task, workdir, llm, tools) -> None:` signature was INSIDE the
`# crucible:region start name=solve` editable region. When the Gemini worker
rewrote the region, it could drop the `def solve` line, producing an
`IndentationError` at import time, which caused the verifier to return `Fail`.

## Fix

Moved `def solve(...)` ABOVE the `# crucible:region start name=solve` marker so
it is FROZEN (outside the editable region). The worker can now only rewrite the
indented function body between the markers, never the signature line.

### New harness.py structure

```python
def solve(task: Task, workdir: Path, llm: LLM, tools: Tools) -> None:
# crucible:region start name=solve
    """docstring and body"""
    ...
# crucible:region end
```

The `# crucible:region start/end` comments are at column 0 (inside the function
body but as comments they don't affect Python indentation parsing). The first
non-comment, non-blank line after `def solve(...)` is indented, establishing the
function body.

### Files changed

1. **`examples/agentic_harness/harness.py`** -- Restructured: `def solve(...)`
   moved above the region marker. The baseline body (read spec, prompt Tess for
   body-only, strip fences, write to skeleton) remains inside the region.

2. **`tests/unit/agentic_harness/test_agentic_verifier.py`** -- Updated
   `_artifact(solve_body)` helper and all test fixtures. The `solve_body`
   parameter now takes the indented body only (e.g., `"    pass"`), not the
   `def solve` line. The helper builds harness files with `def solve(...)` outside
   the region.

3. **`examples/agentic_harness/run_agentic.py`** -- Updated the re-verify block.
   `artifact.region_text(artifact.region("solve"))` now returns only the
   indented body (without `def solve`). The re-verify block now prepends the
   frozen `def solve(...)` signature + imports when writing the best body to the
   temp harness file.

4. **`examples/agentic_harness/measure_baseline.py`** -- No change needed.
   `from harness import solve` still works because `solve` is a top-level
   function.

## Verification

### Unit tests

All 104 tests pass:

```
tests/unit/agentic_harness/ -q
104 passed, 4 warnings in 7.13s
```

### Python validity

```
python -c "import ast; ast.parse(open('examples/agentic_harness/harness.py').read())"
# No error -- valid Python
```

### Region parsing

- `artifact.region("solve")` correctly identifies the body-only region
- `artifact.region_text(region)` returns only the indented body (no `def solve`)
- `_solve_has_hole()` correctly detects `"raise NotImplementedError"` in the
  indented body text
- `replace_region("solve", "    pass")` produces valid Python with `def solve`
  at module level

### Re-verify block

- `solve_text` = region text = indented body only
- Constructed re-verify harness = `from pathlib import Path\nfrom agent_contract import ...\n\ndef solve(...):\n` + body
- Valid Python, callable `solve` function at module level

## Dry-run

The dry-run (`--workers 1 --episodes 1 --edits 1 --turns 4 --wall-clock 45m
--skip-reverify`) was started but the Gemini API call hung for 24+ minutes
without producing any database entries (0 episodes, 0 verdicts). The process
was killed (SIGBUS). This appears to be a Gemini API / rate-limit issue, not a
code bug.

The structural fix is verified through:
- 104 passing unit tests (including the `test_sys_path_fix` test that exec's
  the harness and verifies `solve` is callable)
- Valid Python syntax confirmation via `ast.parse`
- Manual verification of the region parsing, hole detection, and re-verify
  harness construction

## Concerns

- The Gemini API call timing out may need investigation (rate limits, key
  validity, or network issues). This is separate from the structural fix.
- The dry-run should be re-attempted once the API issue is resolved.