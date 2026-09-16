#!/bin/bash
set -e
export ZARR_VOLUME_NAME=mecka-zarr-train
export EGOVERSE_REPO_OVERRIDE=/home/mecka/EgoVerse/.claude/worktrees/fix-frame-count-offbyone
export ZARR_IMAGE_TARGET_HW=224x224
cd /home/mecka/mecka
/home/mecka/.venv-modal/bin/modal run -e robotics --detach \
  modal_pull_and_convert.py::submit_train
