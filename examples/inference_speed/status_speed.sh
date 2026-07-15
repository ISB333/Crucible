#!/usr/bin/env bash
# Live status of the overnight Crucible speed search, straight from the DB.
# Crucible prints nothing to stdout during a run (only the final result), so the
# DB is the only live channel. Run on demand, or wrap in `watch`:
#   watch -n 30 bash examples/inference_speed/status_speed.sh
DB=/home/isb/models/Crucible/examples/inference_speed/speed.db
[ -f "$DB" ] || { echo "no db yet"; exit 0; }

RID=$(sqlite3 "$DB" "SELECT id FROM runs ORDER BY id DESC LIMIT 1")
[ -n "$RID" ] || { echo "no runs yet"; exit 0; }

START=$(sqlite3 "$DB" "SELECT started_at FROM runs WHERE id=$RID")
NOW=$(date +%s)
ELAPSED=$((NOW - ${START%.*}))
HRS=$((ELAPSED/3600)); MINS=$(((ELAPSED%3600)/60))

echo "=== Crucible speed run #$RID  (elapsed ${HRS}h${MINS}m, wall-cap 10h) ==="
sqlite3 -column "$DB" "
SELECT 'workers', (SELECT count(*) FROM workers WHERE run_id=$RID);
" 2>/dev/null
EP=$(sqlite3 "$DB" "SELECT count(*) FROM episodes e JOIN workers w ON e.worker_id=w.id WHERE w.run_id=$RID")
ADV=$(sqlite3 "$DB" "SELECT count(*) FROM events WHERE run_id=$RID AND kind='advisor_consult'")
echo "episodes: $EP | advisor consults: $ADV"

echo "--- verdict breakdown ---"
sqlite3 -header -column "$DB" "
SELECT v.kind, count(*) AS n, round(max(v.score),3) AS best_score
FROM verdicts v
JOIN artifact_versions a ON v.artifact_version_id=a.id
JOIN episodes e ON a.episode_id=e.id
JOIN workers w ON e.worker_id=w.id
WHERE w.run_id=$RID
GROUP BY v.kind ORDER BY n DESC"

echo "--- best aggregate tok/s found (target 30.0) ---"
sqlite3 -column "$DB" "
SELECT round(max(v.score),3) AS best_agg_tok_s
FROM verdicts v
JOIN artifact_versions a ON v.artifact_version_id=a.id
JOIN episodes e ON a.episode_id=e.id
JOIN workers w ON e.worker_id=w.id
WHERE w.run_id=$RID AND lower(v.kind)='scored'"

echo "--- latest 4 verdicts (what's being tried / rejected) ---"
sqlite3 -column "$DB" "
SELECT v.kind, round(v.score,2) AS sc, substr(replace(v.feedback,char(10),' '),1,95) AS feedback
FROM verdicts v
JOIN artifact_versions a ON v.artifact_version_id=a.id
JOIN episodes e ON a.episode_id=e.id
JOIN workers w ON e.worker_id=w.id
WHERE w.run_id=$RID
ORDER BY v.id DESC LIMIT 4"

echo "--- latest config tried ---"
sqlite3 "$DB" "
SELECT substr(a.files_json, instr(a.files_json,'\"config.py\"'),400)
FROM artifact_versions a
JOIN episodes e ON a.episode_id=e.id
JOIN workers w ON e.worker_id=w.id
WHERE w.run_id=$RID
ORDER BY a.id DESC LIMIT 1" 2>/dev/null | tr ',' '\n' | grep -iE "draft_model|cache_type|numa|draft_threads|n_concurrent|n_threads" | head -8