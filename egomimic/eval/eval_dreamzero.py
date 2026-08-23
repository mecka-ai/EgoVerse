"""Offline evaluation for the DreamZero WAM model (Modal fork port).

Loads a hydra config (defaults to ``train_zarr_human_wam_wan22_5b``), plans a
FULL-EPISODE walk over the first N episodes of the valid split, loads a
checkpoint, and runs a single Lightning ``trainer.validate`` pass through
``WAMEvalVideo`` — emitting one ``predicted_video_<i>.mp4`` + one
``validation_video_<i>.mp4`` per episode, each spanning that whole episode, plus
the ``Valid/*`` metrics (optionally dumped as JSON for the sweep aggregator).

Differences vs ``trainHydra.py``:

  1. **Full-episode val split**: ``egomimic.eval.wam_episode.plan_full_episode_walk``
     keeps the first ``+num_val_episodes`` episodes (sorted) and tiles each into
     ``cam_horizon``-length windows at stride ``cam_horizon - 1``, in order.
     ``limit_val_batches`` is auto-set to the resulting window count and the
     valid dataloader batch size is forced to 1, so the val loop walks episode 0
     start-to-finish, then episode 1, ... A mecka dishwashing episode is ~2100
     frames, which is why the episode is streamed as windows rather than handed
     over as one sample; ``wam_rollout.EpisodeRoller`` carries the
     teacher-forcing context across the window seams so the rollout is
     continuous over the episode.

  2. **Selectable rolling mode**: the evaluator yaml picks GT teacher-forced
     (``evaluator=eval_dreamzero_tf``, dreamzero Fig-14a — each chunk conditions
     on GT, drift cannot accumulate) or fully autoregressive
     (``evaluator=eval_dreamzero_ar`` — only the episode anchor is GT). Both go
     through the SAME shared rollout (``egomimic.eval.wam_rollout``) that the
     training-time val loop uses; ``_select_rolling_mode`` just flips a flag.
     There is no longer a monkey-patched second copy of the rolling loop.

By default the config's OWN valid split is evaluated (e.g. the dw48 held-out-
operator json in ``data_dishwashing_48h_wam``). For configs whose valid split
aliases the train set (``mode: total`` + interpolation, e.g. ``mecka_wam``),
pass ``+force_ood_split=true`` to rewrite train/valid into the seed-split
(``valid_mode=valid|train|total`` picks the side).

Extra keys this script understands (all optional, pass with ``+``):
  ``num_val_episodes``         how many episodes to walk (default 3)
  ``max_windows_per_episode``  truncate each episode's walk (smoke tests only)
  ``metrics_out``              path to write the flat metrics JSON
  ``force_ood_split`` / ``valid_ratio`` / ``valid_mode``  seed-split rewrite

Run (repo root; on Modal use egomimic/modal/offline_val_wam.py or
egomimic/modal/wam_val_sweep.py):

    python -m egomimic.eval.eval_dreamzero \
        --config-name=train_zarr_human_wam_wan22_5b \
        data=data_dishwashing_48h_wam evaluator=eval_dreamzero_tf \
        ckpt_path=/path/to/checkpoints/last.ckpt \
        +num_val_episodes=3 \
        <training-time overrides...>
"""

from __future__ import annotations

import copy
import os

import hydra
import lightning as L
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig, OmegaConf, open_dict

# Importing trainHydra registers the eval/multiply OmegaConf resolvers and
# applies the DataLoader shm/tmpdir setup — same import environment as training.
import egomimic.trainHydra as th
from egomimic.eval.wam_episode import plan_full_episode_walk
from egomimic.eval.wam_rollout import WAM_VIDEO_FPS
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.zarr.utils import DataSchematic, set_global_seed
from egomimic.utils.aws.aws_data_utils import load_env
from egomimic.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def _select_rolling_mode(algo, teacher_force: bool) -> None:
    """Point the loaded algo's val loop at TF or AR rolling.

    This used to be a ~250-line monkey-patch that re-implemented the rolling
    loop (``_sample_rolling_tf`` / ``_sample_rolling_ar``) and rebound
    ``val_rollout`` + ``forward_eval`` on the algo, precisely so that
    ``wam.py`` could be left alone. That is what let the offline and
    training-time paths diverge: two copies of the same index arithmetic, only
    one of which ever got fixed. Both paths now share
    ``egomimic.eval.wam_rollout`` (via ``WAM.val_rollout`` ->
    ``EpisodeRoller``), so selecting the mode is a single flag and the offline
    eval measures exactly what training-time validation measures.
    """
    algo.val_teacher_force = bool(teacher_force)
    if hasattr(algo, "_episode_rollers"):
        algo._episode_rollers = {}  # rebuild with the new mode
    mode = (
        "GT teacher-forced (recondition every K latents)"
        if teacher_force
        else "fully-autoregressive (no GT reconditioning after the anchor)"
    )
    log.info(f"[eval_dreamzero] rolling mode: {mode}")


def _dump_metrics(trainer, out_path: str) -> None:
    """Write the val metrics as a flat JSON dict for the sweep aggregator."""
    import json

    metrics = {}
    for k, v in trainer.callback_metrics.items():
        try:
            metrics[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    log.info(f"[eval_dreamzero] wrote {len(metrics)} metrics -> {out_path}")


def _force_ood_split(
    cfg: DictConfig, valid_ratio: float, valid_mode: str = "valid"
) -> None:
    """Rewrite ``cfg.data.train_datasets`` / ``cfg.data.valid_datasets`` so the
    valid loop iterates the requested slice of the seed-split.

    For configs like ``mecka_wam.yaml`` that set ``mode: total`` on train and
    point valid at the SAME instance via interpolation, no split is applied at
    training time. This rewrites each train dataset to ``mode: train`` and
    each valid dataset to a fresh instance of the same target with
    ``mode: valid_mode`` (default ``"valid"`` -> OOD side; ``"train"`` -> the
    training side; ``"total"`` -> the exact set training used).

    NOT needed for configs whose valid split is already held out (e.g.
    ``data_dishwashing_48h_wam`` — held-out-operator eps_to_use json).
    """
    with open_dict(cfg):
        for name, ds_cfg in cfg.data.train_datasets.items():
            ds_cfg.mode = "train"
            ds_cfg.valid_ratio = valid_ratio

        new_valid: dict = {}
        for name in cfg.data.valid_datasets:
            train_ds = cfg.data.train_datasets[name]
            valid_cfg = copy.deepcopy(train_ds)
            valid_cfg.mode = valid_mode
            valid_cfg.valid_ratio = valid_ratio
            new_valid[name] = valid_cfg
        cfg.data.valid_datasets = OmegaConf.create(new_valid)


def _force_valid_batch_size_one(cfg: DictConfig) -> None:
    """Valid dataloaders at batch_size=1 so every episode window becomes its
    own val step -> its own {predicted,validation}_video_<i>.mp4 pair. MUST be
    applied BEFORE the datamodule is instantiated (the wrapper captures the
    dataloader params at construction)."""
    with open_dict(cfg):
        if OmegaConf.select(cfg, "data.valid_dataloader_params") is not None:
            for name in cfg.data.valid_dataloader_params:
                cfg.data.valid_dataloader_params[name].batch_size = 1


def _apply_eval_trainer_overrides(cfg: DictConfig, limit_val_batches: int) -> None:
    """Force the trainer into a single-GPU one-epoch validate-only run."""
    with open_dict(cfg):
        cfg.trainer.pop("_modal", None)  # Modal-submission sentinel
        cfg.trainer.strategy = "auto"
        cfg.trainer.devices = 1
        cfg.trainer.num_nodes = 1
        cfg.trainer.limit_train_batches = 0
        cfg.trainer.limit_val_batches = limit_val_batches
        cfg.trainer.check_val_every_n_epoch = 1
        cfg.trainer.max_epochs = 1
        cfg.trainer.min_epochs = 1
        cfg.trainer.num_sanity_val_steps = 0
        cfg.trainer.sync_batchnorm = False
        cfg.logger = None


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

    num_val_episodes: int = int(cfg.get("num_val_episodes", 3))
    ckpt_path: str | None = cfg.get("ckpt_path")
    if not ckpt_path:
        raise ValueError(
            "ckpt_path must be provided (pass ckpt_path=/path/to/checkpoint.ckpt)."
        )

    # ---- config surgery ----------------------------------------------------
    # Optional seed-split rewrite for `mode: total` configs (see docstring).
    if bool(cfg.get("force_ood_split", False)):
        valid_ratio = float(cfg.get("valid_ratio", 0.2))
        valid_mode = str(cfg.get("valid_mode", "valid"))
        _force_ood_split(cfg, valid_ratio=valid_ratio, valid_mode=valid_mode)
    else:
        valid_mode = "config"
    _force_valid_batch_size_one(cfg)  # before the datamodule is built

    train_datasets = {
        name: hydra.utils.instantiate(cfg.data.train_datasets[name])
        for name in cfg.data.train_datasets
    }
    valid_datasets = {
        name: hydra.utils.instantiate(cfg.data.valid_datasets[name])
        for name in cfg.data.valid_datasets
    }

    for name in valid_datasets:
        tr_names = set(getattr(train_datasets[name], "datasets", {}).keys())
        va_names = set(getattr(valid_datasets[name], "datasets", {}).keys())
        overlap = tr_names & va_names
        if valid_mode == "valid":
            assert not overlap, (
                f"OOD split violated for dataset {name!r}: {len(overlap)} episodes "
                f"appear in both train and valid ({sorted(overlap)[:3]}...)"
            )
        assert va_names, f"Valid split for {name!r} is empty."
        log.info(
            f"[eval_dreamzero] {name}: {len(tr_names)} train / {len(va_names)} "
            f"valid episodes (valid_mode={valid_mode}, overlap={len(overlap)})"
        )

    # Plan a FULL-EPISODE walk: keep the first N episodes (sorted for
    # determinism) and tile each into cam_horizon-length windows at stride
    # cam_horizon-1, in order. With batch_size forced to 1 the val loop then
    # walks episode 0 end-to-end, then episode 1, ... and the evaluator emits
    # ONE mp4 pair per episode covering the whole episode.
    total_windows = 0
    all_plans = []
    for name, mds in valid_datasets.items():
        plans = plan_full_episode_walk(
            mds,
            num_val_episodes,
            max_windows_per_episode=cfg.get("max_windows_per_episode"),
            log=log,
        )
        all_plans.extend(plans)
        total_windows += len(mds)
    log.info(
        f"[eval_dreamzero] {len(all_plans)} episodes / {total_windows} val windows"
    )
    for p in all_plans:
        # The self-check that catches "5 fps by slowing down": the mp4's playback
        # length must equal the REAL elapsed time of the span it covers.
        log.info(
            f"[eval_dreamzero] {p.episode}: {p.video_frames} frames @ "
            f"{WAM_VIDEO_FPS} fps = {p.duration_s:.1f}s "
            f"(episode {p.total_frames} frames @ {p.source_fps:g} fps = "
            f"{p.episode_duration_s:.1f}s real; predicted span "
            f"{p.pred_pixel_frames} frames = {p.predicted_span_s:.1f}s real; "
            f"video subsample x{p.frame_stride})"
        )
        p.assert_realtime()

    assert (
        "MultiDataModuleWrapper" in cfg.data._target_
    ), "cfg.data._target_ must be 'MultiDataModuleWrapper'"
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data,
        train_datasets=train_datasets,
        valid_datasets=valid_datasets,
        train_viz_datasets={},
    )

    # ---- data schematic: same recipe as trainHydra --------------------------
    data_schematic: DataSchematic = hydra.utils.instantiate(cfg.data_schematic)
    for dataset_name, dataset in datamodule.train_datasets.items():
        log.info(f"[eval_dreamzero] Inferring shapes for dataset <{dataset_name}>")
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
    save_cache_dir = OmegaConf.select(cfg, "norm_stats.save_cache_dir", default=None)
    if save_cache_dir:
        data_schematic.cache_stats(save_cache_dir=save_cache_dir)

    viz_func_dict = {
        name: hydra.utils.instantiate(v) for name, v in cfg.visualization.items()
    }

    # ---- model wrap ----------------------------------------------------------
    log.info(f"[eval_dreamzero] Instantiating model <{cfg.model._target_}>")
    model: LightningModule = ModelWrapper(
        config_tree=th._build_model_config_tree(cfg),
        data_schematic_state=data_schematic.to_state(),
        viz_func=viz_func_dict,
        scheduler_interval=cfg.model.get("scheduler_interval", "step"),
    )

    # ---- eval / trainer overrides --------------------------------------------
    _apply_eval_trainer_overrides(cfg, limit_val_batches=total_windows or 1)

    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=None, logger=None)

    # Videos land under <output_dir>/videos/epoch_0/MECKA_BIMANUAL/*.mp4
    os.makedirs(os.path.join(trainer.default_root_dir, "videos"), exist_ok=True)

    # ---- evaluator + checkpoint ----------------------------------------------
    eval_obj = hydra.utils.instantiate(cfg.evaluator)
    eval_obj.trainer = trainer
    eval_obj.model = model.model
    model.evaluator = eval_obj

    # Drive the video subsample off the DATA's frame rate (read from the episode
    # zarr metadata) rather than the class default, so playback is real time for
    # any capture rate. All planned episodes come from the same capture pipeline;
    # assert that rather than silently picking one.
    if all_plans:
        rates = {p.source_fps for p in all_plans}
        assert len(rates) == 1, (
            f"val episodes have mixed source frame rates {sorted(rates)}; one "
            "video subsample stride cannot be real-time for all of them."
        )
        eval_obj.set_source_fps(all_plans[0].source_fps, all_plans[0].data_frame_stride)
        log.info(
            f"[eval_dreamzero] evaluator source_fps={eval_obj.source_fps:g}, "
            f"video subsample stride={eval_obj.frame_stride} "
            f"-> {WAM_VIDEO_FPS} fps real-time playback"
        )

    log.info(f"[eval_dreamzero] Loading checkpoint {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    log.info(
        f"[eval_dreamzero] load_state_dict: {len(missing)} missing / "
        f"{len(unexpected)} unexpected keys"
    )
    # Free the second copy of the weights before Lightning moves the module
    # to GPU (a 5B ckpt is tens of GB on disk).
    del checkpoint
    import gc

    gc.collect()

    # Route eval through the selected rolling sampler — must happen AFTER
    # load_state_dict since we rebind bound methods on the loaded algo.
    teacher_force_rolling = bool(getattr(eval_obj, "teacher_force_rolling", False))
    _select_rolling_mode(model.model, teacher_force=teacher_force_rolling)

    log.info("[eval_dreamzero] Starting evaluation!")
    trainer.validate(model=model, datamodule=datamodule)

    videos_dir = os.path.join(trainer.default_root_dir, "videos")
    log.info(f"[eval_dreamzero] Done. Videos under: {videos_dir}")
    for k, v in sorted(trainer.callback_metrics.items()):
        try:
            log.info(f"[eval_dreamzero] metric {k} = {float(v):.6f}")
        except (TypeError, ValueError):
            pass

    # Per-episode roller stats: how much of each episode the rollout covered.
    for eid, roller in getattr(model.model, "_episode_rollers", {}).items():
        log.info(f"[eval_dreamzero] roller[{eid}] final stats: {roller.stats()}")

    metrics_out = cfg.get("metrics_out")
    if metrics_out:
        _dump_metrics(trainer, str(metrics_out))


if __name__ == "__main__":
    main()
