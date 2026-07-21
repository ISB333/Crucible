#!/usr/bin/env bash
# Detached overnight Crucible agentic search with GLM-5.2:cloud worker (no shepherd).
#
# Survives this terminal closing, an SSH disconnect, AND the user's computer
# shutting down: setsid puts the run in its own session (not the terminal's
# process group), nohup ignores SIGHUP, stdin is /dev/null, stdout/stderr go to
# the log. The VPS keeps the process alive regardless of what happens upstream.
#
# Tess is served before the search and STOPPED when the search finishes (so a
# 10h run doesn't orphan Tess at ~10GB RAM). PYTHONUNBUFFERED=1 streams the
# final result + re-verification to the log.
#
# Monitor: tail -f examples/agentic_harness/overnight.log
# Status:  bash examples/agentic_harness/status_agentic.sh
# Stop:    kill "$(cat examples/agentic_harness/overnight.pid)"; bash examples/agentic_harness/serve_tess.sh --stop
set -euo pipefail

cd /home/isb/models/Crucible
DIR=examples/agentic_harness
LOG=$DIR/overnight.log
PIDF=$DIR/overnight.pid
: > "$LOG"

# Serve Tess first — the verifier needs the OpenAI-compatible endpoint.
bash "$DIR/serve_tess.sh"

# Source .env so OLLAMA_API_KEY (and GOOGLE_API_KEY for Gemini runs) reach the
# Python process. Without this, the setsid child won't inherit env vars that
# are only set in .env (the DB already has empty OPENAI_BASE_URL/OPENAI_API_KEY
# entries, and run_agentic.py overwrites them in-process, but OLLAMA_API_KEY
# must be present for the guard check and the Ollama-Cloud routing).
set -a && source .env && set +a

# Launch the search detached. When it finishes, stop Tess (frees ~10GB RAM).
setsid bash -c "echo \$\$ > '$PIDF'; env PYTHONUNBUFFERED=1 uv run python -u $DIR/run_agentic.py \
  --model glm-5.2:cloud --no-advisor --workers 3 --episodes 6 --edits 4 --turns 8 \
  --wall-clock 10h --plateau-patience 3 >> '$LOG' 2>&1; \
  bash '$DIR/serve_tess.sh' --stop >> '$LOG' 2>&1" < /dev/null &
disown

sleep 3
echo "launched detached; pid=$(cat "$PIDF" 2>/dev/null || echo '?')"
echo "log:   tail -f $LOG"
echo "stop:  kill \"\$(cat $PIDF)\"; bash $DIR/serve_tess.sh --stop"
echo "status: bash $DIR/status_agentic.sh"