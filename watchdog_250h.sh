#!/usr/bin/env bash
# Watchdog for the 250h download. Exits (which re-invokes the agent) the moment
# state changes: target reached, runner died/bailed, or disk getting low.
# Otherwise loops quietly. 12h safety timeout.
LOG=/workspace/EgoVerse/download_250h_resume.log
OUT=/workspace/mecka_random_250h_zarr
TARGET=10087
count_zarr(){ find "$OUT" -maxdepth 1 -name '*.zarr' -type d 2>/dev/null | wc -l; }
for i in $(seq 1 720); do
  sleep 60
  cnt=$(count_zarr)
  runner=$(pgrep -fc 'run_download_250h.sh')
  oom=$(grep -o 'oom_kill [0-9]*' /sys/fs/cgroup/memory.events | awk '{print $2}')
  freeg=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
  if [ "$cnt" -ge "$TARGET" ]; then
    echo "WATCHDOG: TARGET REACHED  zarr=$cnt/$TARGET  oom_kill=$oom  freeGB=$freeg"; exit 0
  fi
  if [ "$runner" -lt 1 ]; then
    echo "WATCHDOG: RUNNER NOT RUNNING  zarr=$cnt/$TARGET  oom_kill=$oom  (done or bailed — investigate)"; exit 2
  fi
  if [ "${freeg:-9999}" -lt 200 ]; then
    echo "WATCHDOG: LOW DISK  freeGB=$freeg  zarr=$cnt/$TARGET"; exit 3
  fi
done
echo "WATCHDOG: 12h timeout  zarr=$(count_zarr)/$TARGET"; exit 1
