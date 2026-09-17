#!/bin/bash
# One line per arm run: staging progress, and whether anything has failed.
MODAL=/home/mecka/.venv-modal/bin/modal
APPS="ap-MbEu0J2GYJrkwI8gmqsQh0:clean-top
ap-wXxZj4SIyb9cEBxdl5NKsi:clean-bottom
ap-vD0zUakn1BkH9WLbAU9akV:clean-random
ap-s2CzLUjKco0UjOyDSOXQyc:org-top
ap-cjiIf26fPI551EE5EiYHDe:org-bottom
ap-KVFCjiAggiT0GvELzdsKmH:org-random
ap-FlIe485FUJM9ci0FY0NMEM:dish-top
ap-n8wuoEELnYfxR0ea9AcHjb:dish-bottom
ap-uYC7kV4DUsAvjI2Sd57kkw:dish-random"

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
