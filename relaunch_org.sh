#!/bin/bash
# Relaunch the two organization-stocking runs that died on the mixed-resolution
# episode (6a91bcfd87ec349d4b502e59, 360x640 among 126x224). It has been dropped
# from both arm manifests, which are now 2,297 episodes each against top's 2,298
# -- a 0.04% difference, immaterial next to the ~1-3% episodes lost to decode
# failures, and not worth restarting a healthy top run to equalise.
set -e
cd /home/mecka/EgoVerse/.claude/worktrees/fix-frame-count-offbyone

for arm in bottom random; do
  echo "=== relaunching organization-stocking / ${arm} ==="
  python egomimic/modal/trainModal.py \
    data=qaexp_arm_6d \
    trainer=ddp_modal \
    logger=wandb \
    model=pi0.5_bc_mecka \
    name="qaexp-organization-stocking-${arm}" \
    description="qaexp-scoring-organization-stocking-${arm}" \
    '~evaluator@train_viz_evaluator' \
    evaluator=eval_pi \
    "+data.train_datasets.mecka_bimanual.resolver.eps_to_use=/mnt/zarr-data/_arms/organization-stocking__${arm}.json" \
    "+data.valid_datasets.mecka_bimanual.resolver.eps_to_use=/mnt/zarr-data/_arms/organization-stocking__${arm}.json" \
    model.robomimic_model.config.pytorch_weight_path=pi_checkpoints/pi05_base_pytorch \
    +modal_gpu=H200:4 \
    +modal_cpu=64 \
    +modal_memory_gb=200 \
    +modal_volume=mecka_zarr_qaexp \
    +modal_ephemeral_disk_gb=600
  sleep 5
done
echo "=== both relaunched ==="
