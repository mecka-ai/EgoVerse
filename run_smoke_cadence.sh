#!/bin/bash
# Does validation run, and does a checkpoint land on cadence?
#
# Neither path had ever executed. The earlier smokes ran max_epochs=1 against
# check_val_every_n_epoch=200 and every_n_epochs=100, so validation never ran and
# no on-cadence checkpoint was ever written -- which is how nine runs reached
# epoch 200 and then died there with zero artifacts:
#   - AttributeError: 'PI' object has no attribute 'shared_ac_key' (eval_hpt)
#   - and the checkpoint, tied by Lightning to the validation loop, died with it
#
# So turn both cadences down until they actually fire: 3 epochs, validate every
# 2, checkpoint every 1. Real batch size and real GPU count are kept, because a
# smoke may only differ from the real run on axes that cannot cause the failure
# being screened for.
set -e
cd /home/mecka/EgoVerse/.claude/worktrees/fix-frame-count-offbyone

python egomimic/modal/trainModal.py \
  data=qaexp_arm_6d \
  trainer=ddp_modal \
  logger=wandb \
  model=pi0.5_bc_mecka \
  name=smoke-cadence \
  description=smoke-validation-and-checkpoint-cadence \
  '~evaluator@train_viz_evaluator' \
  '+data.train_datasets.mecka_bimanual.resolver.eps_to_use=/mnt/zarr-data/_arms/dish-handling__random.json' \
  '+data.valid_datasets.mecka_bimanual.resolver.eps_to_use=/mnt/zarr-data/_arms/dish-handling__random.json' \
  data.train_datasets.mecka_bimanual.episodes_per_epoch=32 \
  data.valid_datasets.mecka_bimanual.episodes_per_epoch=8 \
  trainer.limit_train_batches=4 \
  trainer.limit_val_batches=2 \
  trainer.max_epochs=3 \
  trainer.min_epochs=1 \
  trainer.check_val_every_n_epoch=2 \
  callbacks.model_checkpoint.every_n_epochs=1 \
  norm_stats.sample_frac=0.005 \
  model.robomimic_model.config.pytorch_weight_path=pi_checkpoints/pi05_base_pytorch \
  +modal_gpu=H200:4 \
  +modal_cpu=64 \
  +modal_memory_gb=200 \
  +modal_volume=mecka_zarr_qaexp \
  +modal_ephemeral_disk_gb=600
