#!/bin/bash
# Training progress and dropped-episode rate per arm.
#
# The drop rate is the one that decides whether the experiment is sound: a flat
# ~1% loss across arms is tolerable, but if low-quality arms lose systematically
# more episodes than high-quality ones, that confounds the exact comparison
# these runs exist to make.
MODAL=/home/mecka/.venv-modal/bin/modal
APPS="ap-iP3oWp6dCaBSEO6oNWqdar:clean-top
ap-QchA9nkYbHWf7yIIugSPc4:clean-bottom
ap-AMZCJbjDR26JB1SANGNU3y:clean-random
ap-MpLHe8Wq2cY15got58pzUV:org-top
ap-vb9BrpxLx1IshdkvHXGFq2:org-bottom
ap-Oktxk0qydANIhs6gjvzQ2c:org-random
ap-XOzyPTtv11JEBJE3IvvQ0W:dish-top
ap-S7kN8yeITm4EzSmcjx2eyi:dish-bottom
ap-2TCOC2DBTjWgfEQ6SKy4Q6:dish-random"

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
