#!/bin/bash
# The 9 QA_exp scoring-effectiveness runs: three task families x top/bottom/
# random, equal episode counts within each family so the only variable is which
# half of the quality ranking the data came from.
#
# Same path the smoke run verified end to end (Train/Loss 0.21512, last.ckpt
# written), with the smoke overrides dropped so qaexp_arm_6d's real settings
# apply: batch 128, 12 workers, episodes_per_epoch 2400 (>= every arm's train
# split, so each arm stages once and is then reused from local NVMe).
set -euo pipefail
cd /home/mecka/EgoVerse/.claude/worktrees/fix-frame-count-offbyone

for domain in cleaning-sanitation organization-stocking dish-handling; do
  for arm in top bottom random; do
    echo "=== launching ${domain} / ${arm} ==="
    python egomimic/modal/trainModal.py \
      data=qaexp_arm_6d \
      trainer=ddp_modal \
      logger=wandb \
      model=pi0.5_bc_mecka \
      name="qaexp-${domain}-${arm}" \
      description="qaexp-scoring-${domain}-${arm}" \
      '~evaluator@train_viz_evaluator' \
      "+data.train_datasets.mecka_bimanual.resolver.eps_to_use=/mnt/zarr-data/_arms/${domain}__${arm}.json" \
      "+data.valid_datasets.mecka_bimanual.resolver.eps_to_use=/mnt/zarr-data/_arms/${domain}__${arm}.json" \
      model.robomimic_model.config.pytorch_weight_path=pi_checkpoints/pi05_base_pytorch \
      +modal_gpu=H200:4 \
      +modal_cpu=64 \
      +modal_memory_gb=200 \
      +modal_volume=mecka_zarr_qaexp \
      +modal_ephemeral_disk_gb=600
    sleep 5
  done
done
echo "=== all 9 launched ==="
