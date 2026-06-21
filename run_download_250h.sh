#!/usr/bin/env bash
# Resilient outer runner for the mecka_random_250h Zarr download.
# The python script is idempotent (skips episodes whose <hash>.zarr already
# exists), so on any crash (OOM -> BrokenProcessPool, segfault, etc.) we just
# relaunch and it continues from where it stopped. Stops when the target count
# is reached, on a clean below-target exit, or after 2 no-progress iterations.


  # cd /workspace/EgoVerse
  # WORKERS=10 NUM_SHARDS=N SHARD=k TMPDIR=/root/dl_tmp \
  #   setsid bash run_download_250h.sh >/workspace/EgoVerse/runner.s${k}.out 2>&1 </dev/null &
  # disown
set -u
cd /workspace/EgoVerse
source /workspace/EgoVerse/emimic/bin/activate
set -a; . /workspace/claude/home/.egoverse_env; set +a
# Scratch dir for downloads + frame extraction. On extra nodes, override with a
# NODE-LOCAL path (e.g. TMPDIR=/local/tmp) so heavy I/O stays off the shared NFS.
TMPDIR="${TMPDIR:-/workspace/tmp}"
export TMPDIR TEMP="$TMPDIR" TMP="$TMPDIR"
mkdir -p "$TMPDIR"

OUT=/workspace/mecka_random_250h_zarr
IDS=/workspace/EgoVerse/mecka_random_250h.json
WORKERS="${WORKERS:-16}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD="${SHARD:-0}"
TARGET=10087
MAX_ITERS="${MAX_ITERS:-50}"
# Per-shard log so multiple nodes writing to the shared volume don't clobber.
if [ "$NUM_SHARDS" -gt 1 ]; then
  LOG=/workspace/EgoVerse/download_250h_resume.s${SHARD}of${NUM_SHARDS}.log
else
  LOG=/workspace/EgoVerse/download_250h_resume.log
fi

count_zarr(){ find "$OUT" -maxdepth 1 -name '*.zarr' -type d 2>/dev/null | wc -l; }

prev=$(count_zarr)
stall=0
for i in $(seq 1 "$MAX_ITERS"); do
  echo "===== RUNNER iter $i  $(date -u +%H:%M:%S)  zarr=$(count_zarr)/$TARGET  workers=$WORKERS  shard=$SHARD/$NUM_SHARDS =====" | tee -a "$LOG"
  # Drop orphaned in-flight temp dirs left by a prior crash (nothing is running now).
  find "$TMPDIR" -maxdepth 1 -name 'zarr_local_*' -type d -exec rm -rf {} + 2>/dev/null
  python download_episodes_local.py --ids-file "$IDS" --output-dir "$OUT" \
    --workers "$WORKERS" --num-shards "$NUM_SHARDS" --shard "$SHARD" >> "$LOG" 2>&1
  rc=$?
  now=$(count_zarr)
  echo "===== RUNNER iter $i done rc=$rc  zarr=$now/$TARGET (+$((now-prev))) =====" | tee -a "$LOG"
  if [ "$now" -ge "$TARGET" ]; then echo "RUNNER: target reached" | tee -a "$LOG"; break; fi
  if [ "$rc" -eq 0 ]; then echo "RUNNER: clean exit below target (remaining episodes likely unconvertible)" | tee -a "$LOG"; break; fi
  if [ "$now" -le "$prev" ]; then stall=$((stall+1)); else stall=0; fi
  if [ "$stall" -ge 2 ]; then echo "RUNNER: no progress for 2 iters; stopping for investigation" | tee -a "$LOG"; break; fi
  prev="$now"
done
echo "RUNNER: finished iter-loop. final zarr=$(count_zarr)/$TARGET  $(date -u +%H:%M:%S)" | tee -a "$LOG"
