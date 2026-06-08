# Crucible — Test & Run Guide

## Prerequisites

- Python env: `uv venv .venv && uv pip install -e ".[dev,gemini]"`
- Docker running (for integration tests and the chem image)
- API key in environment (`ANTHROPIC_API_KEY` or `GOOGLE_API_KEY`)

## 1. Build Docker images

```bash
docker build -t crucible-chem:0 -f docker/chem.Dockerfile docker/
docker build -t crucible-py:0   -f docker/py.Dockerfile   docker/
docker build -t crucible-lean:0 -f docker/lean.Dockerfile docker/
```

## 2. Unit tests

```bash
uv run pytest -m unit -v
```

## 3. Integration tests

```bash
uv run pytest -m integration -v
```

## 4. Run the chemistry example

```bash
# Anthropic (default model: claude-haiku-4-5)
ANTHROPIC_API_KEY=... uv run python examples/chem/run_chem.py

# Gemini
GOOGLE_API_KEY=... uv run python examples/chem/run_chem.py --model gemini-flash
```

3 workers run for up to 20 episodes each (plateau patience 5). Starting molecule is octane.
Target: logS ≥ 0.4. If hit, prints the solution; otherwise prints the best found.

## 5. Inspect reasoning

After a run, see how the model thought through each episode:

```bash
# List all runs in the database
uv run crucible runs --db crucible-chem.db

# Show reasoning for the latest run
uv run crucible reasoning --db crucible-chem.db

# Specific run / worker / episode
uv run crucible reasoning 5 --db crucible-chem.db --worker 0 --episode 0
```

See [REASONING.md](REASONING.md) for details.

## 6. Lint and typecheck

```bash
uv run ruff check crucible tests examples
uv run ruff format --check crucible tests examples
uv run pyright crucible tests examples
```
