#!/usr/bin/env bash
# Detached overnight Crucible speed search on the 9B.
#
# Survives this terminal closing, an SSH disconnect, AND the user's computer
# shutting down: setsid puts the run in its own session (not the terminal's
# process group), nohup ignores SIGHUP, stdin is /dev/null, stdout/stderr go to
# the log. The VPS keeps the process alive regardless of what happens upstream.
#
# Monitor: tail -f /home/isb/models/Crucible/examples/inference_speed/overnight.log
# Status:  grep -E 'SOLVED|Best partial|re-verification' .../overnight.log
# Stop:    kill "$(cat .../overnight.pid)"; pkill -f llama-server
set -euo pipefail

cd /home/isb/models/Crucible
LOG=/home/isb/models/Crucible/examples/inference_speed/overnight.log
PIDF=/home/isb/models/Crucible/examples/inference_speed/overnight.pid
: > "$LOG"

setsid bash -c 'echo $$ > "'"$PIDF"'"; exec env PYTHONUNBUFFERED=1 uv run python -u examples/inference_speed/run_speed.py \
  --workers 4 --episodes 200 --edits 25 --turns 12 \
  --wall-clock 10h --plateau-patience 12' >> "$LOG" 2>&1 < /dev/null &
disown

sleep 3
echo "launched detached; pid=$(cat "$PIDF" 2>/dev/null || echo '?')"
echo "log:   tail -f $LOG"
echo "stop:  kill \"\$(cat $PIDF)\"; pkill -f llama-server"