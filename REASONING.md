# LLM Reasoning Tracking

This feature allows you to capture and view the detailed reasoning process of LLMs as they collaborate to solve problems. You can see their conversation history, including the problem statement, steps taken to solve it, and final solution.

## Overview

The reasoning tracking feature works by:
1. Capturing the full conversation history for each LLM session
2. Storing the reasoning in the SQLite database as JSON
3. Providing a CLI command to view the captured reasoning

## How to Use

### 1. Run a Task with Reasoning Tracking

To capture reasoning, simply run a task as you normally would. The reasoning will be automatically stored in the database.

Example commands:

```bash
# Run chemistry example with reasoning tracking
GOOGLE_API_KEY=... uv run python examples/chem/run_chem.py --model gemini-3.5-flash

# Run math example with reasoning tracking
OPENAI_API_KEY=... uv run python examples/run_kata_real_model.py --model gpt-4-turbo
```

### 2. View Captured Reasoning

Use the `reasoning` CLI command to view the captured reasoning.

#### Basic Usage

```bash
uv run python -m crucible reasoning
```

This will display the reasoning for all workers and episodes.

#### Filter by Worker

```bash
uv run python -m crucible reasoning --worker 0
```

This will display reasoning only for worker 0.

#### Filter by Episode

```bash
uv run python -m crucible reasoning --episode 10
```

This will display reasoning only for episode 10.

#### Filter by Worker and Episode

```bash
uv run python -m crucible reasoning --worker 0 --episode 10
```

This will display reasoning for worker 0 and episode 10.

#### Specify Database

If you're using a custom database file (like in the chemistry example), specify it with the `--db` option:

```bash
uv run python -m crucible reasoning --db crucible-chem.db
```

## Examples

### Chemistry Example

Run the chemistry example:
```bash
GOOGLE_API_KEY=... uv run python examples/chem/run_chem.py --model gemini-3.5-flash
```

View the reasoning:
```bash
uv run python -m crucible reasoning --db crucible-chem.db
```

### Math Example

Run the math example:
```bash
OPENAI_API_KEY=... uv run python examples/run_kata_real_model.py --model gpt-4-turbo
```

View the reasoning:
```bash
uv run python -m crucible reasoning
```

## Implementation Details

The reasoning tracking feature is implemented across several files:

1. `crucible/llm.py`: Added `messages` property to the `LLMSession` protocol
2. `crucible/providers.py`: Implemented messages property for all LLM providers (Anthropic, OpenAI, Gemini)
3. `crucible/store.py`: Added `reasoning_json` column to the episodes table
4. `crucible/worker.py`: Updated to capture and store the reasoning
5. `crucible/cli.py`: Added new "reasoning" command with filtering options

## Notes

- Reasoning is automatically captured for all LLM sessions
- The reasoning is stored as JSON in the `reasoning_json` column of the `episodes` table
- The CLI command supports filtering by worker and/or episode
- All LLM providers (Anthropic, OpenAI, Gemini) are supported
