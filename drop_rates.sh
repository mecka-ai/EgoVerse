#!/bin/bash
# Distinct episodes each round-2 run discarded as undecodable, over the whole run.
#
# This is the only measurement from the experiment that bears on whether the
# quality score tracks anything real, because no checkpoints survived and the
# per-arm training losses are computed on different data and so are not
# comparable with each other.
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
  app="${row%%:*}"; name="${row##*:}"
  log=$($MODAL app logs "$app" -e robotics --since 1440m 2>&1 | sed 's/\x1b\[[0-9;]*m//g')
  bad=$(echo "$log" | grep -oE "PrefetchedMapDataset: [0-9a-f]{24} marked" | awk '{print $2}' | sort -u | wc -l)
  lines=$(echo "$log" | wc -l)
  printf '%-14s distinct_bad_episodes=%-5s (log lines seen: %s)\n' "$name" "$bad" "$lines"
done
