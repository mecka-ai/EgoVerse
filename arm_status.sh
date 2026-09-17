#!/bin/bash
# One line per arm run: staging progress, and whether anything has failed.
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
  app="${row%%:*}"
  name="${row##*:}"
  log=$($MODAL app logs "$app" -e robotics --since 15m 2>&1 | sed 's/\x1b\[[0-9;]*m//g')
  stage=$(echo "$log" | grep -E "PoolFiller stats" | tail -1 | grep -oE "extract_cursor=[0-9]+ .* used=[0-9.]+ GB")
  err=$(echo "$log" | grep -cE "Training failed|prepare_epoch: timeout|OutOfMemoryError|rank0\]: Traceback")
  step=$(echo "$log" | grep -oE "[0-9]+/[0-9]+ .*it/s" | tail -1)
  sync=$(echo "$log" | grep -c "all ranks synchronized")
  printf '%-14s err=%s ddp=%s %s %s\n' "$name" "$err" "$sync" "${stage:-<no filler line>}" "$step"
done
