#!/usr/bin/env bash
# Serve Tess-4-9B with the best-aggregate config from the inference speed search.
# OpenAI-compatible endpoint for agentic harness task evals.
#
#   bash examples/agentic_harness/serve_tess.sh            # agg profile (default — 12 concurrent, ~10 tok/s total)
#   bash examples/agentic_harness/serve_tess.sh --single   # single-stream profile (one user, ~7.7 tok/s)
#   bash examples/agentic_harness/serve_tess.sh --ngram   # ngram/lookup decoding — NO draft model (~4.5GB less RAM)
#   bash examples/agentic_harness/serve_tess.sh --port 9090
#   bash examples/agentic_harness/serve_tess.sh --stop
#
# Tess-4-9B is Qwen-based (same 248K tokenizer), so the speed search's best configs
# transfer directly. Q3_K_M self-spec draft = same tokenizer = lossless speculative
# decoding. KV-cache q8_0 saves VRAM with negligible quality loss (greedy-match
# verified). Quality is byte-identical to the unoptimized model.
set -euo pipefail

# Tess-4-9B with best-aggregate config. Override via env to test another model
# with the same best settings (e.g. Qwen3.5-9B — same arch, same tokenizer):
#   MODEL=/home/isb/models/Qwen3.5-9B-Q4_K_M.gguf \
#   DRAFT=/home/isb/models/Qwen3.5-9B-Q3_K_M.gguf bash serve_tess.sh
MODEL=${MODEL:-/home/isb/models/Tess-4-9B-Q4_K_M.gguf}
DRAFT=${DRAFT:-/home/isb/models/Tess-4-9B-Q3_K_M.gguf}
PORT=9090
PROFILE=agg
PIDF=/home/isb/models/Crucible/examples/agentic_harness/serve.pid
LOG=/home/isb/models/Crucible/examples/agentic_harness/serve.log

while [ $# -gt 0 ]; do
  case "$1" in
    --single) PROFILE=single; shift;;
    --agg) PROFILE=agg; shift;;
    --ngram) PROFILE=ngram; shift;;
    --port) PORT="$2"; shift 2;;
    --stop)
      kill "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null || true
      pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
      echo "stopped"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

# Args verified by Config.to_cli_args() (same path the verifier measured).
if [ "$PROFILE" = "agg" ]; then
  # best aggregate (10.15): n_threads=10, n_concurrent=12, draft_threads=2
  ARGS=(--threads 10 --batch-size 512 --ubatch-size 512 --ctx-size 4096 --parallel 12
        --flash-attn on --model-draft "$DRAFT" --spec-draft-n-max 16 --spec-draft-threads 2
        --cache-type-k q8_0 --cache-type-v q8_0)
elif [ "$PROFILE" = "ngram" ]; then
  # ngram/lookup decoding: drafts from n-gram stats in the running context, NO draft
  # model (~4.5GB less RAM, no draft bandwidth contention — better VPS fit than the
  # Q3 draft). Lossless: the target still verifies every drafted token.
  ARGS=(--threads 6 --batch-size 512 --ubatch-size 512 --ctx-size 4096 --parallel 8
        --flash-attn on --spec-type ngram-mod --spec-ngram-mod-n-max 16 --spec-ngram-mod-n-match 24
        --cache-type-k q8_0 --cache-type-v q8_0)
else
  # best single-stream (7.67): n_threads=6, n_concurrent=8, draft_threads=6
  ARGS=(--threads 6 --batch-size 512 --ubatch-size 512 --ctx-size 4096 --parallel 8
        --flash-attn on --model-draft "$DRAFT" --spec-draft-n-max 16 --spec-draft-threads 6
        --cache-type-k q8_0 --cache-type-v q8_0)
fi

: > "$LOG"
setsid bash -c "echo \$\$ > '$PIDF'; exec llama-server --model '$MODEL' ${ARGS[*]} --port $PORT --host 0.0.0.0" \
  >> "$LOG" 2>&1 < /dev/null &
disown

# wait for readiness — curl -sf fails on 4xx/5xx; llama-server returns 503 while
# loading and 200 only when ready, so this naturally waits for the real server (and
# rejects a stale service on the port returning 404).
echo "launching (profile=$PROFILE, port=$PORT)..."
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 1
done
if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "READY — OpenAI-compatible endpoint at http://$(hostname -I | awk '{print $1}'):$PORT/v1"
  echo "pid=$(cat "$PIDF")  log=$LOG"
  echo
  echo "Quick test:"
  echo "  curl http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \\"
  echo "    -d '{\"model\":\"tess\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in 3 words\"}],\"max_tokens\":20,\"chat_template_kwargs\":{\"enable_thinking\":false}}'"
  echo
  echo "Stop:  bash examples/agentic_harness/serve_tess.sh --stop"
else
  echo "FAILED to become ready in 180s — check $LOG (is port $PORT already in use?)"; exit 1
fi