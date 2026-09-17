#!/bin/bash
# Does batch_size 48 fit on H200:4?
#
# The 9 arm runs all died with torch.OutOfMemoryError at their first step
# (batch 128 -> 135.2 of 139.8 GiB). GPU memory depends on batch size, model and
# rank count, NOT on how many episodes are staged -- so stage only a handful of
# episodes and take a few steps. That answers the memory question in ~10 minutes
# instead of paying 1.5h of staging nine times over to find out.
#
# Keeps the real batch_size (48, from the config) and the real GPU count.
set -e
cd /home/mecka/EgoVerse/.claude/worktrees/fix-frame-count-offbyone

python egomimic/modal/trainModal.py \
  data=qaexp_arm_6d \
  trainer=ddp_modal \
  logger=wandb \
  model=pi0.5_bc_mecka \
  name=smoke4gpu-batch48 \
  description=smoke-h200x4-batch48-memory-check \
  '~evaluator@train_viz_evaluator' \
  '+data.train_datasets.mecka_bimanual.resolver.eps_to_use=/mnt/zarr-data/_arms/dish-handling__random.json' \
  '+data.valid_datasets.mecka_bimanual.resolver.eps_to_use=/mnt/zarr-data/_arms/dish-handling__random.json' \
  data.train_datasets.mecka_bimanual.episodes_per_epoch=32 \
  data.valid_datasets.mecka_bimanual.episodes_per_epoch=8 \
  trainer.limit_train_batches=8 \
  trainer.limit_val_batches=2 \
  trainer.max_epochs=1 \
  norm_stats.sample_frac=0.005 \
  model.robomimic_model.config.pytorch_weight_path=pi_checkpoints/pi05_base_pytorch \
  +modal_gpu=H200:4 \
  +modal_cpu=64 \
  +modal_memory_gb=200 \
  +modal_volume=mecka_zarr_qaexp \
  +modal_ephemeral_disk_gb=600
