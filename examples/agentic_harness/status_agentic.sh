#!/usr/bin/env bash
# Live status of the agentic harness search, straight from the DB.
# Crucible prints little mid-run (only the final result), so the DB is the
# primary live channel. Run on demand, or wrap in `watch`:
#   watch -n 30 bash examples/agentic_harness/status_agentic.sh
DB=/home/isb/models/Crucible/examples/agentic_harness/agentic.db
[ -f "$DB" ] || { echo "no db yet"; exit 0; }

RID=$(sqlite3 "$DB" "SELECT id FROM runs ORDER BY id DESC LIMIT 1")
[ -n "$RID" ] || { echo "no runs yet"; exit 0; }

START=$(sqlite3 "$DB" "SELECT started_at FROM runs WHERE id=$RID")
NOW=$(date +%s)
ELAPSED=$((NOW - ${START%.*}))
HRS=$((ELAPSED/3600)); MINS=$(((ELAPSED%3600)/60))

echo "=== Agentic harness run #$RID  (elapsed ${HRS}h${MINS}m, wall-cap 10h) ==="
NW=$(sqlite3 "$DB" "SELECT count(*) FROM workers WHERE run_id=$RID")
echo "workers: $NW"

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

echo "--- best pass_rate (scored verdicts) ---"
sqlite3 -column "$DB" "
SELECT round(max(v.score),3) AS best_pass_rate
FROM verdicts v
JOIN artifact_versions a ON v.artifact_version_id=a.id
JOIN episodes e ON a.episode_id=e.id
JOIN workers w ON e.worker_id=w.id
WHERE w.run_id=$RID AND v.kind='SCORED'"

echo "--- latest 4 verdicts ---"
sqlite3 -header -column "$DB" "
SELECT v.kind, round(v.score,3) AS score, substr(replace(v.feedback,char(10),' '),1,95) AS feedback
FROM verdicts v
JOIN artifact_versions a ON v.artifact_version_id=a.id
JOIN episodes e ON a.episode_id=e.id
JOIN workers w ON e.worker_id=w.id
WHERE w.run_id=$RID
ORDER BY v.id DESC LIMIT 4"