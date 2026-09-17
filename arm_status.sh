#!/bin/bash
# One line per arm run: staging progress, and whether anything has failed.
MODAL=/home/mecka/.venv-modal/bin/modal
APPS="ap-N7emVJHDpRharURX0JB43B:clean-top
ap-FNlpNuPQMzYkjuw1mBEKrD:clean-bottom
ap-640FSeblWE61nfsJDXOEk0:clean-random
ap-zpqIXUoX4UNWZWuEXIZ7K4:org-top
ap-yVOGjcUta6ZNm4CkE7HDpN:org-bottom
ap-NfaPJWaxpgqAithu35Wftd:org-random
ap-q6Ip0ZteZo1sEflqqgkSbS:dish-top
ap-JTarb0KKpSNJ6DNYXfG59X:dish-bottom
ap-p14wtVksyM6eagpgVHOWzF:dish-random"

for row in $APPS; do
  app="${row%%:*}"
  name="${row##*:}"
  log=$($MODAL app logs "$app" -e robotics --since 15m 2>&1 | sed 's/\x1b\[[0-9;]*m//g')
  stage=$(echo "$log" | grep -E "PoolFiller stats" | tail -1 | grep -oE "extract_cursor=[0-9]+ .* used=[0-9.]+ GB")
  err=$(echo "$log" | grep -cE "Training failed|prepare_epoch: timeout|OutOfMemoryError|rank0\]: Traceback")
  step=$(echo "$log" | grep -oE "[0-9]+/[0-9]+ .*it/s" | tail -1)
  sync=$(echo "$log" | grep -c "all ranks synchronized")
  printf '%-14s err=%s ddp=%s %s %s\n' "$name" "$err" "$sync" "${stage:-<no filler line>}" "$step"
done
