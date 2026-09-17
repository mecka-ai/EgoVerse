#!/bin/bash
# Which arm runs have produced checkpoints yet, and which ones.
# Only the newest run dir per arm is inspected -- earlier dirs are from the
# rounds that died before writing anything.
MODAL=/home/mecka/.venv-modal/bin/modal
for dom in cleaning-sanitation organization-stocking dish-handling; do
  for arm in top bottom random; do
    root="qaexp-${dom}-${arm}"
    newest=$($MODAL volume ls egoverse-training-outputs "$root" -e robotics 2>/dev/null \
             | sed 's/\x1b\[[0-9;]*m//g' | grep "$root/" | tail -1)
    [ -z "$newest" ] && { printf '%-38s <no run dir>\n' "$root"; continue; }
    ck=$($MODAL volume ls egoverse-training-outputs "$newest/checkpoints" -e robotics 2>/dev/null \
         | sed 's/\x1b\[[0-9;]*m//g' | grep -oE "epoch_epoch=[0-9]+\.ckpt|last\.ckpt" | sort -u | tr '\n' ' ')
    printf '%-38s %s\n' "${dom}/${arm}" "${ck:-<none yet>}"
  done
done
