#!/bin/bash
# Smoke run: one arm, a handful of steps, just to prove the path end to end.
#
# What this is actually checking, none of which has ever run before:
#   - pi0.5_bc_mecka's 18-dim 6D action space against the staged zarr path
#     (every existing staged config is 12-dim ypr, so a shape mismatch at the
#     action converter would surface here rather than 30 minutes into a real run)
#   - ZarrDirEpisodeResolver + eps_to_use pinning an arm inside the container
#   - the directory-copy staging path under the real pool/filler machinery
#   - norm stats computing from a 2% sample without a precomputed cache
#
# episodes_per_epoch is overridden down so staging is ~2 minutes, not ~30.
set -e
cd /home/mecka/EgoVerse/.claude/worktrees/fix-frame-count-offbyone

python egomimic/modal/trainModal.py \
  data=qaexp_arm_6d \
  trainer=ddp_modal \
  logger=wandb \
  model=pi0.5_bc_mecka \
  name=smoke-cleaning-top \
  description="smoke: pi0.5 6D + zarr staging, 1 arm, few steps" \
  '~evaluator@train_viz_evaluator' \
  '+data.train_datasets.mecka_bimanual.resolver.eps_to_use=/mnt/zarr-data/_arms/cleaning-sanitation__top.json' \
  '+data.valid_datasets.mecka_bimanual.resolver.eps_to_use=/mnt/zarr-data/_arms/cleaning-sanitation__top.json' \
  data.train_datasets.mecka_bimanual.episodes_per_epoch=24 \
  data.valid_datasets.mecka_bimanual.episodes_per_epoch=8 \
  data.train_dataloader_params.mecka_bimanual.batch_size=8 \
  data.train_dataloader_params.mecka_bimanual.num_workers=4 \
  data.valid_dataloader_params.mecka_bimanual.batch_size=8 \
  data.valid_dataloader_params.mecka_bimanual.num_workers=2 \
  trainer.limit_train_batches=5 \
  trainer.limit_val_batches=2 \
  trainer.max_epochs=1 \
  norm_stats.sample_frac=0.05 \
  +modal_gpu=H200:1 \
  +modal_cpu=16 \
  +modal_memory_gb=64 \
  +modal_volume=mecka_zarr_qaexp \
  +modal_ephemeral_disk_gb=512 \
  init_submodules=false
