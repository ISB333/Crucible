# Crucible v0.5 Molecular Optimization — Full Test Guide

This guide walks through testing all components of the Crucible v0.5 molecular optimization extension. It covers unit tests, integration tests, and the live example.

## Prerequisites

1. **Python virtual environment**: Already set up with `uv venv .venv && uv pip install -e ".[dev,gemini]"`
2. **Docker**: Installed and running (for integration tests and the chem image)
3. **API Key** (for live example): 
   - For Anthropic models: `ANTHROPIC_API_KEY` set in the environment
   - For Google Gemini models: `GOOGLE_API_KEY` or `GEMINI_API_KEY` set in the environment (optional for tests)

## 1. Build Required Docker Images

First, build the `crucible-chem:0` image for the molecular verifier:

```bash
cd .
docker build -t crucible-chem:0 -f docker/chem.Dockerfile docker/
```

Optionally, build other images if needed:
```bash
# For Python verifier (v0 base)
docker build -t crucible-py:0 -f docker/py.Dockerfile docker/


# For Lean math ladder example
docker build -t crucible-lean:0 -f docker/lean.Dockerfile docker/
```

## 2. Run All Unit Tests

Run the full unit test suite to verify core functionality:

```bash
.venv/bin/pytest -m unit -v
```

Expected output: All 173 unit tests pass, 29 deselected.

Key v0.5 unit tests to check specifically:
- `tests/unit/test_scored_verdict.py`: Tests for the `Scored` verdict type
- `tests/unit/test_store_scores.py`: Tests for score persistence
- `tests/unit/test_outcome_rank.py`: Tests for episode ranking
- `tests/unit/test_scored_search.py`: Tests for best-score selection
- `tests/unit/test_plateau.py`: Tests for plateau patience
- `tests/unit/test_chem_verifier.py`: Tests for the Chem verifier (with fake sandbox)

## 3. Run Integration Tests

Run integration tests to verify real-world behavior:

```bash
.venv/bin/pytest -m integration -v
```

Expected output: 28 integration tests pass, 1 skipped.

Key v0.5 integration tests:
- `tests/integration/test_chem_live.py`: Molecular optimization end-to-end tests (requires `crucible-chem:0`)
  - `test_above_target_molecule_is_accepted`: Verifies molecules meeting the target are accepted as solutions
  - `test_below_target_run_returns_best_scoring`: Verifies the best-scoring molecule is returned when no target is hit

## 4. Run the Live Molecular Optimization Example

Test the real-model example with Google Gemini (requires a Google API key):

```bash
cd .
export GOOGLE_API_KEY="your-api-key-here"
.venv/bin/python examples/chem/run_chem.py
```

To use Anthropic models instead:

```bash
cd .
export ANTHROPIC_API_KEY="your-api-key-here"
sed -i 's/gemini-pro/claude-sonnet-4-6/' examples/chem/run_chem.py
.venv/bin/python examples/chem/run_chem.py
```

Expected behavior:
- 3 workers run for up to 20 episodes each (with 5-episode plateau patience)
- The initial molecule is octane (insoluble)
- The model will explore more soluble molecules
- If a molecule with logS ≥ 0.0 is found, it will be accepted as a solution
- If not, the best-scoring molecule found will be printed

## 5. Verify Linting and Typechecking

Ensure all code adheres to style guidelines and type annotations:

```bash
# Lint check
.venv/bin/ruff check crucible tests examples

# Format check (should report "68 files already formatted")
.venv/bin/ruff format --check crucible tests examples

# Typecheck
.venv/bin/pyright crucible tests examples
```

Expected outputs:
- `All checks passed!` for ruff check
- `68 files already formatted` for ruff format --check
- `0 errors, 0 warnings, 0 informations` for pyright

## 6. Verify Git Status and Branch

Ensure the working tree is clean and you're on the correct branch:

```bash
git status
git branch
```

Expected outputs:
- `nothing to commit, working tree clean`
- `* main`

## 7. Merge the Branch (Optional)

If you're ready to merge into the main branch:

```bash
git checkout main
git merge feat/exp4-molecular-opt-v05
```

## Summary of v0.5 Changes

The molecular optimization extension adds:

1. **New Verdict Type**: `Scored(value)` for valid artifacts with a numeric score
2. **Ranking System**: Episodes ranked as Ok > Scored > Partial > Fail
3. **Best-by-Rank Selection**: Orchestrator picks the best-scoring artifact when no Ok is reached
4. **Plateau Patience**: Early stopping when the best score stops improving
5. **Chem Verifier**: RDKit-based scorer for molecular solubility (logS)
6. **crucible-chem:0 Image**: Docker image with pre-installed RDKit

All changes preserve the v0 accept path (first-wins, immutable spec, fresh-reverify), making this a backward-compatible optimization extension.
