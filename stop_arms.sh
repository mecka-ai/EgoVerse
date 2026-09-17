#!/bin/bash
# Stop the current arm runs. Kept in sync with arm_status.sh.
MODAL=/home/mecka/.venv-modal/bin/modal
for app in \
  ap-iP3oWp6dCaBSEO6oNWqdar \
  ap-QchA9nkYbHWf7yIIugSPc4 \
  ap-AMZCJbjDR26JB1SANGNU3y \
  ap-MpLHe8Wq2cY15got58pzUV \
  ap-vb9BrpxLx1IshdkvHXGFq2 \
  ap-Oktxk0qydANIhs6gjvzQ2c \
  ap-XOzyPTtv11JEBJE3IvvQ0W \
  ap-S7kN8yeITm4EzSmcjx2eyi \
  ap-2TCOC2DBTjWgfEQ6SKy4Q6 ; do
  echo "stopping $app"
  $MODAL app stop "$app" -e robotics --yes 2>&1 | tail -1
done
