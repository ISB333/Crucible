# LLM Reasoning Inspection

After running any example, you can inspect how the model reasoned through each episode.

## List runs

```bash
uv run crucible runs
uv run crucible runs --db crucible-chem.db
```

## Show reasoning

```bash
# Latest run (all workers and episodes)
uv run crucible reasoning --db crucible-chem.db

# Specific run
uv run crucible reasoning 22 --db crucible-chem.db

# Filter to one worker or episode
uv run crucible reasoning --worker 0 --episode 0 --db crucible-chem.db
```

## What you see

Each message in the conversation is printed in order:

- `[user]` — the initial problem (artifact + verdict) and tool results
- `[model]` — the model's text reasoning and tool calls (`→ search_replace(...)`)
- `  ← ...` — verifier feedback returned after each edit

## Where it's stored

Reasoning is captured automatically in the `reasoning_json` column of the `episodes` table in the SQLite database. All providers (Anthropic, OpenAI, Gemini) are supported.
