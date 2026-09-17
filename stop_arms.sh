#!/bin/bash
# Stop the current arm runs. Kept in sync with arm_status.sh.
MODAL=/home/mecka/.venv-modal/bin/modal
for app in \
  ap-N7emVJHDpRharURX0JB43B \
  ap-FNlpNuPQMzYkjuw1mBEKrD \
  ap-640FSeblWE61nfsJDXOEk0 \
  ap-zpqIXUoX4UNWZWuEXIZ7K4 \
  ap-yVOGjcUta6ZNm4CkE7HDpN \
  ap-NfaPJWaxpgqAithu35Wftd \
  ap-q6Ip0ZteZo1sEflqqgkSbS \
  ap-JTarb0KKpSNJ6DNYXfG59X \
  ap-p14wtVksyM6eagpgVHOWzF ; do
  echo "stopping $app"
  $MODAL app stop "$app" -e robotics --yes 2>&1 | tail -1
done
