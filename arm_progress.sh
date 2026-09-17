#!/bin/bash
# Training progress and dropped-episode rate per arm.
#
# The drop rate is the one that decides whether the experiment is sound: a flat
# ~1% loss across arms is tolerable, but if low-quality arms lose systematically
# more episodes than high-quality ones, that confounds the exact comparison
# these runs exist to make.
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
  app="${row%%:*}"; name="${row##*:}"
  log=$($MODAL app logs "$app" -e robotics --since 90m 2>&1 | sed 's/\x1b\[[0-9;]*m//g')
  bad=$(echo "$log" | grep -oE "PrefetchedMapDataset: [0-9a-f]{24} marked" | awk '{print $2}' | sort -u | wc -l)
  ep=$(echo "$log" | grep -oE "PoolFiller stats: train_cursor=[0-9]+" | tail -1 | grep -oE "[0-9]+$")
  loss=$(echo "$log" | grep -oE "Train/Loss +[0-9.]+" | tail -1 | awk '{print $2}')
  wait=$(echo "$log" | grep -oE "dataloader wait [0-9.]+s" | tail -1)
  err=$(echo "$log" | grep -cE "Training failed|OutOfMemoryError")
  printf '%-14s err=%s bad_eps=%-4s train_cursor=%-7s loss=%-8s %s\n' \
    "$name" "$err" "$bad" "${ep:-?}" "${loss:-<none yet>}" "$wait"
done
