#!/bin/bash
# Stop the current arm runs. Kept in sync with arm_status.sh.
MODAL=/home/mecka/.venv-modal/bin/modal
for app in \
  ap-MbEu0J2GYJrkwI8gmqsQh0 \
  ap-wXxZj4SIyb9cEBxdl5NKsi \
  ap-vD0zUakn1BkH9WLbAU9akV \
  ap-s2CzLUjKco0UjOyDSOXQyc \
  ap-cjiIf26fPI551EE5EiYHDe \
  ap-KVFCjiAggiT0GvELzdsKmH \
  ap-FlIe485FUJM9ci0FY0NMEM \
  ap-n8wuoEELnYfxR0ea9AcHjb \
  ap-uYC7kV4DUsAvjI2Sd57kkw ; do
  echo "stopping $app"
  $MODAL app stop "$app" -e robotics --yes 2>&1 | tail -1
done
