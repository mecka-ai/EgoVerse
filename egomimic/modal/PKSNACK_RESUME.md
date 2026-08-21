# packaging_snacks runs — paused 2026-08-21, how to resume

Both families were **paused** (Modal apps stopped, auto-resume watchers killed) on
2026-08-21. Nothing was deleted: every run still has its `checkpoints/last.ckpt`
and its W&B run id, so a resume is just "re-run the original command".

## How resume works here (why the command is unchanged)

`trainHydra.py` looks for `last.ckpt` inside the run's `hydra.run.dir` and, if
present, that checkpoint **overrides** any launch-time `ckpt_path` / `finetune_ckpt`.
So resuming requires only that you keep

  * the same `hydra.run.dir` (→ finds `last.ckpt`), and
  * the same `wandb_run_id` (→ appends to the same W&B run instead of starting a new one).

Both are already baked into the commands below. Do **not** change them, and do not
delete the W&B run — a deleted run id can never be reused.

## State at pause

`max_epochs = 2000` for every run. `every_n_epochs=30` checkpoints.

| run | last epoch ckpt | epochs left | GPU | W&B run id |
|---|---|---|---|---|
| pksnack_ops/L1   | 1859 | ~140 | A100      | `pksnack_ops_L1` |
| pksnack_ops/L2   | 1949 | ~50  | A100      | `pksnack_ops_L2` |
| pksnack_ops/L3   | 1979 | ~20  | A100      | `pksnack_ops_L3` |
| pksnack_ops/L4   | 1949 | ~50  | A100      | `pksnack_ops_L4` |
| pksnack_size/300M | 1979 | ~20 | A100      | `pksnack_size_300M` |
| pksnack_size/600M | 1799 | ~200 | A100     | `pksnack_size_600M` |
| pksnack_size/1B   | 1199 | ~800 | A100-80GB | `pksnack_size_1B` |
| pksnack_size/1_5B | 839  | ~1160 | A100-80GB | `pksnack_size_1_5B` |

`L3`, `300M` (1979) and `L2`, `L4` (1949) are within one or two checkpoints of the
2000-epoch finish line — cheapest to just let those four run to completion.
`1B` and `1_5B` are the ones with real time left.

## Resume — operator scaling (L1..L4)

4 levels of operator diversity at a fixed 10 h data budget, all 300M, THREE val
legs (in-domain-operator holdout, fully held-out operator, trained-on train-viz).

```bash
# lvl in L1 L2 L3 L4   (all A100)
MODAL_GPU=A100 MODAL_ENVIRONMENT=robotics python egomimic/modal/trainModal.py \
  name=pksnack_ops description=<lvl> \
  data=data_pksnack_opscale_<lvl>_zarr \
  model=hpt_bc_flow_mecka_300M_shared_head \
  data_schematic.norm_mode=minmax reject_outliers=false \
  trainer=ddp_modal trainer.limit_train_batches=100 \
  trainer.check_val_every_n_epoch=30 trainer.limit_val_batches=100 \
  callbacks.model_checkpoint.every_n_epochs=30 \
  +trainer.gradient_clip_val=10 \
  norm_stats.precomputed_norm_path=data_div_smoke/D1_2026-07-22_22-05-03/norm_stats \
  evaluator@train_viz_evaluator=train_viz_hpt_oph \
  +second_val_prefix=Valid_oph \
  +evaluator@extra_val_evaluator=train_viz_hpt_trainviz \
  +third_val_prefix=Valid_trainviz \
  'logger.wandb.tags=["packaging-snacks","op-scaling"]' \
  wandb_run_id=pksnack_ops_<lvl> \
  hydra.run.dir=./logs/pksnack_ops/<lvl> \
  +modal_ephemeral_disk_gb=600 init_submodules=false
```

Val-leg → metric/video mapping (do not reorder; the prefixes are positional):
idx0 = in-domain-operator holdout (`Valid/*`, `videos/`),
idx1 = fully held-out operator (`Valid_oph/*`, `videos_oph/`),
idx2 = trained-on train-viz (`Valid_trainviz/*`, `videos_trainviz/`).

## Resume — model-size scaling (300M/600M/1B/1_5B)

Same 10 h packaging_snacks data for all four; TWO val legs.

```bash
# size in 300M 600M -> A100 ;  1B 1_5B -> A100-80GB (1B OOMs on 40GB)
MODAL_GPU=<A100|A100-80GB> MODAL_ENVIRONMENT=robotics python egomimic/modal/trainModal.py \
  name=pksnack_size description=<size> \
  data=data_pksnack_all_zarr \
  model=hpt_bc_flow_mecka_<size>_shared_head \
  data_schematic.norm_mode=minmax reject_outliers=false \
  trainer=ddp_modal trainer.limit_train_batches=100 \
  trainer.check_val_every_n_epoch=30 trainer.limit_val_batches=100 \
  callbacks.model_checkpoint.every_n_epochs=30 \
  +trainer.gradient_clip_val=10 \
  norm_stats.precomputed_norm_path=data_div_smoke/D1_2026-07-22_22-05-03/norm_stats \
  evaluator@train_viz_evaluator=train_viz_hpt_trainviz \
  +second_val_prefix=Valid_trainviz \
  'logger.wandb.tags=["packaging-snacks","size-scaling"]' \
  wandb_run_id=pksnack_size_<size> \
  hydra.run.dir=./logs/pksnack_size/<size> \
  +modal_ephemeral_disk_gb=600 init_submodules=false
```

idx0 = held-out operator (`Valid/*`, `videos/`), idx1 = trained-on train-viz
(`Valid_trainviz/*`, `videos_trainviz/`).

## Gotchas that cost time before — keep these

* `+trainer.gradient_clip_val=10` is **required on resume**. The MAD grad-clip
  history is not checkpointed, so without a hard clip the first post-resume
  outlier batch explodes 1B/1.5B.
* `init_submodules=false` on every launch (pi0.5 is the only exception).
* `1B` and `1_5B` need `A100-80GB`; 1B OOMs at ~37.7 GiB on a 40 GB A100.
* These runs are **preemption-heavy**. Without a watcher they will sit down after
  the first preemption — re-arm an auto-resume watcher if you want them unattended.
  Run the watcher with `python3 -u` so a silent death is visible in its log.
* Watchers must be killed BEFORE stopping apps, or they immediately resubmit.
