"""
Offline val-metric sweep across a folder of WAM checkpoints.

Given a training run's ``checkpoints/`` folder, enumerate every
``epoch_epoch=*.ckpt``, run a validation pass on the run's val split, and log
the resulting metrics keyed by epoch — to wandb when a key is available, and
always to ``val_sweep_metrics.json`` in the output dir. Uses the same TF
rolling (``eval_dreamzero._sample_rolling_tf``) that the offline eval uses so
the per-block reconditioning matches the training-time val loop.

Design:
  * Instantiate the datamodule + model + data schematic ONCE (expensive), then
    loop checkpoints and swap in state_dicts. A fresh trainer per ckpt so
    ``trainer.callback_metrics`` doesn't carry across.
  * Wandb logging is direct (``wandb.init`` / ``wandb.log(step=epoch)``)
    rather than through Lightning's ``WandbLogger`` so we control the step
    axis (epoch, not global_step which resets on every load).
  * Videos are moved to a ``ckpt_epoch_<N>`` subdir per pass so nothing
    overwrites.

Extra config keys (on top of ``train_zarr_human_wam_wan22_5b``):
  * ``checkpoints_dir``  — folder holding ``epoch_epoch=*.ckpt`` files
  * ``num_val_episodes`` — per-embodiment val episodes (default 5)
  * ``wandb_project`` / ``wandb_run_name`` — wandb dest (optional)

Run (repo root; on Modal use egomimic/modal/offline_val_wam.py::sweep):

    python -m egomimic.eval.val_sweep \
        --config-name=train_zarr_human_wam_wan22_5b \
        data=data_dishwashing_48h_wam \
        +checkpoints_dir=/path/to/run/checkpoints \
        <training-time overrides...>
"""

from __future__ import annotations

import copy
import gc
import glob
import json
import os
import re

import hydra
import lightning as L
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig, OmegaConf

# Importing trainHydra registers the eval/multiply OmegaConf resolvers and the
# DataLoader shm/tmpdir setup — same import environment as training.
import egomimic.trainHydra as th
from egomimic.eval.eval import Eval
from egomimic.eval.eval_dreamzero import (
    _apply_eval_trainer_overrides,
    _force_ood_split,
    _patch_algo_use_sample_rolling,
    _restrict_to_first_n_episodes,
)
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.zarr.utils import DataSchematic, set_global_seed
from egomimic.utils.aws.aws_data_utils import load_env
from egomimic.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def _list_checkpoints(ckpt_dir: str) -> list[tuple[int, str]]:
    """Return sorted ``[(epoch, path), ...]`` for every ``epoch_epoch=N.ckpt``.

    Skips ``last.ckpt`` — it's a duplicate of the highest-epoch checkpoint,
    and gets overwritten by any resume so it's not a stable point in time.
    """
    pattern = os.path.join(ckpt_dir, "epoch_epoch=*.ckpt")
    out: list[tuple[int, str]] = []
    for p in glob.glob(pattern):
        m = re.search(r"epoch=(\d+)\.ckpt$", p)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def _metrics_to_floats(metrics: dict) -> dict:
    """Flatten trainer.callback_metrics (Tensors) -> floats."""
    result = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            try:
                result[k] = float(v.detach().cpu().item())
            except Exception:
                continue
        elif isinstance(v, (int, float)):
            result[k] = float(v)
    return result


@hydra.main(
    version_base="1.3",
    config_path="../hydra_configs",
    config_name="train_zarr_human_wam_wan22_5b",
)
def main(cfg: DictConfig) -> None:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
        set_global_seed(cfg.seed)
    else:
        raise ValueError("Seed must be provided in cfg for reproducibility!")

    load_env()

    checkpoints_dir = cfg.get("checkpoints_dir")
    if not checkpoints_dir:
        raise ValueError(
            "checkpoints_dir must be provided (path to a training run's "
            "``checkpoints/`` folder holding ``epoch_epoch=*.ckpt`` files)."
        )
    ckpts = _list_checkpoints(checkpoints_dir)
    if not ckpts:
        raise ValueError(f"No ``epoch_epoch=*.ckpt`` files under {checkpoints_dir}")
    log.info(
        f"[val_sweep] Found {len(ckpts)} checkpoints: epochs {[e for e, _ in ckpts]}"
    )

    num_val_episodes: int = int(cfg.get("num_val_episodes", 5))
    wandb_project: str = str(cfg.get("wandb_project", "egoverse"))
    wandb_run_name: str = str(
        cfg.get(
            "wandb_run_name",
            f"val_sweep_{os.path.basename(os.path.dirname(checkpoints_dir.rstrip('/')))}",
        )
    )

    # ---- config surgery (optional seed-split; default = config's own valid) -
    if bool(cfg.get("force_ood_split", False)):
        _force_ood_split(
            cfg,
            valid_ratio=float(cfg.get("valid_ratio", 0.2)),
            valid_mode=str(cfg.get("valid_mode", "valid")),
        )

    # ---- datasets ------------------------------------------------------------
    train_datasets = {
        name: hydra.utils.instantiate(cfg.data.train_datasets[name])
        for name in cfg.data.train_datasets
    }
    valid_datasets = {
        name: hydra.utils.instantiate(cfg.data.valid_datasets[name])
        for name in cfg.data.valid_datasets
    }

    total_windows = 0
    for name, mds in valid_datasets.items():
        _restrict_to_first_n_episodes(mds, num_val_episodes)
        total_windows += len(mds)
    log.info(
        f"[val_sweep] Sweeping {total_windows} val windows per checkpoint "
        f"(x {num_val_episodes} episodes per dataset)."
    )

    # ---- datamodule ------------------------------------------------------------
    assert "MultiDataModuleWrapper" in cfg.data._target_
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data,
        train_datasets=train_datasets,
        valid_datasets=valid_datasets,
        train_viz_datasets={},
    )

    # ---- data schematic (same recipe as trainHydra) ----------------------------
    data_schematic: DataSchematic = hydra.utils.instantiate(cfg.data_schematic)
    for dataset_name, dataset in datamodule.train_datasets.items():
        data_schematic.infer_shapes_from_batch(dataset[0])
        instantiate_copy = copy.deepcopy(cfg.data.train_datasets[dataset_name])
        km = OmegaConf.to_container(instantiate_copy.resolver.key_map, resolve=False)
        km["norm_mode"] = True
        instantiate_copy.resolver.key_map = km
        norm_dataset = hydra.utils.instantiate(instantiate_copy)
        data_schematic.infer_norm_from_dataset(
            norm_dataset,
            dataset_name,
            sample_frac=OmegaConf.select(cfg, "norm_stats.sample_frac", default=1.0),
            num_workers=OmegaConf.select(cfg, "norm_stats.num_workers", default=4),
            precomputed_norm_path=OmegaConf.select(
                cfg, "norm_stats.precomputed_norm_path", default=None
            ),
        )

    viz_func_dict = {
        name: hydra.utils.instantiate(v) for name, v in cfg.visualization.items()
    }

    # ---- model (instantiated ONCE — reused across ckpts via load_state_dict) --
    model: LightningModule = ModelWrapper(
        config_tree=th._build_model_config_tree(cfg),
        data_schematic_state=data_schematic.to_state(),
        viz_func=viz_func_dict,
        scheduler_interval=cfg.model.get("scheduler_interval", "step"),
    )

    # ---- one-epoch validate-only trainer overrides ----------------------------
    _apply_eval_trainer_overrides(cfg, limit_val_batches=total_windows or 1)

    # ---- wandb (optional) ------------------------------------------------------
    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        import wandb

        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "checkpoints_dir": checkpoints_dir,
                "num_val_episodes": num_val_episodes,
                "n_checkpoints": len(ckpts),
                "epochs": [e for e, _ in ckpts],
            },
        )
    else:
        log.warning("[val_sweep] WANDB_API_KEY not set — metrics go to JSON only.")

    all_metrics: dict[int, dict] = {}
    out_root = None

    # ---- checkpoint loop ---------------------------------------------------------
    for epoch, ckpt_path in ckpts:
        log.info(f"[val_sweep] === epoch {epoch}: loading {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(
            checkpoint["state_dict"], strict=False
        )
        log.info(
            f"[val_sweep] load_state_dict: {len(missing)} missing / "
            f"{len(unexpected)} unexpected keys"
        )
        del checkpoint
        gc.collect()

        # Fresh trainer per ckpt so callback_metrics doesn't carry over and
        # WAMEvalVideo's per-eid buffers start empty.
        trainer: Trainer = hydra.utils.instantiate(
            cfg.trainer, callbacks=None, logger=None
        )
        out_root = trainer.default_root_dir
        os.makedirs(os.path.join(out_root, "videos"), exist_ok=True)

        eval_obj: Eval = hydra.utils.instantiate(cfg.evaluator)
        eval_obj.trainer = trainer
        eval_obj.model = model.model
        model.evaluator = eval_obj

        # Route rolling through the same TF path the offline eval uses so the
        # per-block reconditioning matches the training val loop's semantics.
        _patch_algo_use_sample_rolling(model.model, teacher_force=True)

        trainer.validate(model=model, datamodule=datamodule)

        metrics = _metrics_to_floats(trainer.callback_metrics)
        all_metrics[epoch] = metrics
        log.info(f"[val_sweep] epoch {epoch} metrics: {metrics}")
        if use_wandb:
            import wandb

            wandb.log(metrics, step=epoch)

        # Move the videos this ckpt produced into an epoch-scoped subdir so
        # subsequent ckpts don't overwrite them.
        src_videos = os.path.join(out_root, "videos", "epoch_0")
        dst_videos = os.path.join(out_root, "videos", f"ckpt_epoch_{epoch}")
        if os.path.isdir(src_videos) and not os.path.exists(dst_videos):
            os.rename(src_videos, dst_videos)

        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    if out_root:
        with open(os.path.join(out_root, "val_sweep_metrics.json"), "w") as f:
            json.dump(all_metrics, f, indent=2)
        log.info(
            f"[val_sweep] metrics saved: {os.path.join(out_root, 'val_sweep_metrics.json')}"
        )
    if use_wandb:
        import wandb

        wandb.finish()
    log.info("[val_sweep] Done.")


if __name__ == "__main__":
    main()
