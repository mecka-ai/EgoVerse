#!/bin/bash
# One line per arm run: staging progress, and whether anything has failed.
MODAL=/home/mecka/.venv-modal/bin/modal
APPS="ap-KYRz9NbWEMjo7FoXccQ3kA:clean-top
ap-paWxUBuQhkKOviMNuG0dFU:clean-bottom
ap-JwuyZvZwjaAw3T87inrjH1:clean-random
ap-jallqHjxWeioJh33niPkac:org-top
ap-w8bpXsjsoPRZbO27qJU34m:org-bottom
ap-7wKtRYoKAouCXDG55FTNmu:org-random
ap-CxqacIBU9xorGWaB2tuZtj:dish-top
ap-59VDAQIFTiEUZ9qebG9ElH:dish-bottom
ap-NPvTDiiSAmUTNrWvH2tma8:dish-random"

for row in $APPS; do
  app="${row%%:*}"
  name="${row##*:}"
  log=$($MODAL app logs "$app" -e robotics --since 15m 2>&1 | sed 's/\x1b\[[0-9;]*m//g')
  stage=$(echo "$log" | grep -E "PoolFiller stats" | tail -1 | grep -oE "extract_cursor=[0-9]+ .* used=[0-9.]+ GB")
  err=$(echo "$log" | grep -cE "Training failed|prepare_epoch: timeout|rank0\]: Traceback")
  step=$(echo "$log" | grep -oE "[0-9]+/[0-9]+ .*it/s" | tail -1)
  sync=$(echo "$log" | grep -c "all ranks synchronized")
  printf '%-14s err=%s ddp=%s %s %s\n' "$name" "$err" "$sync" "${stage:-<no filler line>}" "$step"
done
