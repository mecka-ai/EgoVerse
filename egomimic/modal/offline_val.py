"""Offline (weights-only) validation for the 8 data_div_oss runs on Modal.

For a given run_tag + epoch, re-composes that run's EXACT training config from
its `.hydra/overrides.yaml` (stored on the egoverse-training-outputs volume),
swaps the valid episodes for 3 IN-DISTRIBUTION episodes (seen-in-training
operator — a control experiment against the operator-holdout val plateau),
loads the checkpoint weights (strict=True) and runs `trainer.validate` on a
single GPU. The run's own EvalVideo evaluator writes GT-vs-pred videos to
    <outputs volume>/offline_val_indist/<run_tag>/videos/epoch_0/MECKA_BIMANUAL/
and Valid/* metrics land in a CSVLogger + manifest.json in the same dir.

The live training runs and their dirs are NOT touched: their overrides.yaml /
checkpoints are only read, and all outputs go to the new offline_val_indist/
prefix.

Usage (single run — the debug loop):
    MODAL_ENVIRONMENT=robotics modal run egomimic/modal/offline_val.py::run \
        --run-tag 300M_mm_nobc_dw48 --epoch 539

All / several runs in parallel:
    MODAL_ENVIRONMENT=robotics modal run --detach egomimic/modal/offline_val.py::run_many \
        --run-tags 300M_mm_nobc_dw48,600M_mm_nobc_dw48,...

Notes
-----
- OSS (HPT) runs validate on A100-80GB, pi0.5-family runs on H200 (they were
  trained on 2 GPUs, but eval_video.py has no rank guard — 2 ranks corrupt the
  videos — so validation is single-GPU: trainer.devices=1, strategy=auto).
- The pi runs need the openpi submodule + its patched transformers==4.53.2
  (applied by modal_setup._prepare_repo) and TORCHDYNAMO_DISABLE=1 (the image
  ships no C compiler; pi's sample_actions is wrapped in @torch.compile).
- The in-dist episode json is written at runtime into the output dir, so this
  script does not require committing new files to the repo.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

# modal_setup.py lives next to this file locally (egomimic/modal/) and is baked
# into the image at /root/ so it is importable before the repo is cloned.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from modal_setup import (  # noqa: E402
    CFG,
    _prepare_repo,
    _resolve_git_state,
    image,
    training_outputs_volume,
    zarr_volume,
)

app = modal.App("offline-val-indist", image=image)

RUN_BASE = "data_div_oss"  # where the 8 training run dirs live on the volume
OUT_PREFIX = "offline_val_indist"  # NEW prefix at the volume root for all outputs
COMMON_EPOCH = 539  # largest checkpoint epoch present for ALL 8 runs (verified)

# In-distribution operator: 68b5da0ce7c6a693e3df941c has 4 dishwashing episodes
# in dishwashing_48h_train.json (a typical operator; most have 1-6). These are
# their 3 shortest episodes — all SEEN IN TRAINING (that's the point). Sorted by
# hash == the order episode_table_to_df resolves them in.
INDIST_OPERATOR = "68b5da0ce7c6a693e3df941c"
INDIST_EPISODES = [
    "69b2100ed99f29421f1b4a57",  # 1825 frames
    "69b335f8290f064f72218fab",  # 2083 frames
    "69b37b384af76c8acce9cc65",  # 2245 frames
]
INDIST_TOTAL_FRAMES = 1825 + 2083 + 2245  # 6153

OSS_RUNS = (
    "300M_mm_nobc_dw48",
    "600M_mm_nobc_dw48",
    "1B_mm_nobc_dw48",
    "1_5B_mm_nobc_dw48",
)
PI_RUNS = (
    "pi05_dw48",
    "pali_dw48",
    "pi05_lang_dw48",
    "pali_lang_dw48",
)

# Original-run overrides stripped for the offline pass (replaced below, or not
# applicable to a fresh weights-only validation).
_DROP_KEYS = {
    "wandb_run_id",
    "ckpt_path",
    "finetune_ckpt",
    "launch_params.gpus_per_node",
    "trainer.limit_train_batches",
    "trainer.limit_val_batches",
    "trainer.check_val_every_n_epoch",
}
_DROP_PREFIXES = ("logger.", "callbacks.")


def _fmt_limit(limit_val_batches: float) -> str:
    """Format limit_val_batches for a hydra override: fraction <= 1.0, else int."""
    if limit_val_batches <= 1.0:
        return f"{float(limit_val_batches)}"
    return f"{int(limit_val_batches)}"


def _build_overrides(
    run_tag: str, out_dir: str, eps_json: str, limit_val_batches: float
) -> list[str]:
    """Original run overrides, minus wandb/ckpt/logger keys, plus offline-val ones."""
    import yaml

    overrides_path = (
        f"{CFG.output_mount_path}/{RUN_BASE}/{run_tag}/.hydra/overrides.yaml"
    )
    with open(overrides_path) as f:
        original = yaml.safe_load(f)
    print(f"[offline_val] original overrides ({overrides_path}):")
    for ov in original:
        print(f"    {ov}")

    kept = []
    for ov in original:
        key = ov.split("=", 1)[0].lstrip("+~")
        if key in _DROP_KEYS or key == "logger" or key.startswith(_DROP_PREFIXES):
            continue
        kept.append(ov)

    kept += [
        # single GPU: eval_video.py has no rank guard, 2 ranks corrupt videos
        "launch_params.gpus_per_node=1",
        "trainer.devices=1",
        "trainer.num_nodes=1",
        "trainer.strategy=auto",
        "trainer.num_sanity_val_steps=0",
        "trainer.check_val_every_n_epoch=1",
        f"trainer.limit_val_batches={_fmt_limit(limit_val_batches)}",
        # all outputs to the offline_val_indist/<run_tag> prefix
        f"paths.output_dir={out_dir}",
        "norm_stats.save_cache_dir=null",
        # 3 in-dist episodes for the val set; train set restricted to the same 3
        # episodes — it is never iterated (validate only), but is still needed
        # for shape inference (dataset[0], same embodiment => same shapes) and
        # the norm-stats dataset construction (stats themselves load from the
        # run's precomputed_norm_path, which stays untouched in the overrides).
        f"data.train_datasets.mecka_bimanual.resolver.eps_to_use={eps_json}",
        f"data.valid_datasets.mecka_bimanual.resolver.eps_to_use={eps_json}",
    ]
    print("[offline_val] final overrides:")
    for ov in kept:
        print(f"    {ov}")
    return kept


def _run_offline_val(
    run_tag: str,
    epoch: int,
    git_remote: str,
    git_commit: str,
    submodules: frozenset,
    limit_val_batches: float,
    is_pi: bool,
) -> dict:
    """Container body: clone repo, compose the run's config, validate, commit."""
    # --- env BEFORE any heavy import ---
    os.environ["MODAL_IS_REMOTE"] = "1"
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if is_pi:
        # pi0.5's sample_actions is @torch.compile'd; the image has no C
        # compiler, so TorchInductor would crash eval (see trainModal.py).
        os.environ["TORCHDYNAMO_DISABLE"] = "1"

    # Clone the repo (+ openpi submodule and its transformers==4.53.2 overlay
    # for pi runs — must happen before `import transformers`).
    _prepare_repo(git_remote=git_remote, git_commit=git_commit, submodules=submodules)
    if CFG.remote_repo_dir not in sys.path:
        sys.path.insert(0, CFG.remote_repo_dir)
    openpi_src = f"{CFG.remote_repo_dir}/external/openpi/src"
    if is_pi and openpi_src not in sys.path:
        sys.path.insert(0, openpi_src)
    os.chdir(CFG.remote_repo_dir)

    out_dir = f"{CFG.output_mount_path}/{OUT_PREFIX}/{run_tag}"
    os.makedirs(out_dir, exist_ok=True)
    eps_json = f"{out_dir}/offline_val_indist3.json"
    with open(eps_json, "w") as f:
        json.dump(INDIST_EPISODES, f)

    ckpt_path = (
        f"{CFG.output_mount_path}/{RUN_BASE}/{run_tag}/checkpoints/"
        f"epoch_epoch={epoch}.ckpt"
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    overrides = _build_overrides(run_tag, out_dir, eps_json, limit_val_batches)

    # --- heavy imports (after repo clone / transformers swap) ---
    import copy
    import glob as _glob

    import hydra
    import lightning as L
    import torch
    from hydra import compose, initialize_config_dir
    from lightning.pytorch.loggers import CSVLogger
    from omegaconf import OmegaConf, open_dict

    # Importing trainHydra registers the eval/multiply OmegaConf resolvers and
    # applies the DataLoader shm/tmpdir setup — same environment as training.
    import egomimic.trainHydra as th
    from egomimic.pl_utils.pl_model import ModelWrapper
    from egomimic.rldb.zarr.utils import DataSchematic, set_global_seed
    from egomimic.utils.aws.aws_data_utils import load_env
    from egomimic.utils.dataloader_ipc import configure_dataloader_ipc

    with initialize_config_dir(
        config_dir=f"{CFG.remote_repo_dir}/egomimic/hydra_configs",
        version_base="1.3",
    ):
        cfg = compose(config_name="train_zarr_cartesian", overrides=overrides)

    with open_dict(cfg):
        cfg.logger = None  # no wandb; CSVLogger is attached manually below
        cfg.ckpt_path = None
        cfg.finetune_ckpt = None

    # ---- mirror trainHydra.train() (validation-relevant parts) ----
    configure_dataloader_ipc()
    L.seed_everything(cfg.seed, workers=True)
    set_global_seed(cfg.seed)
    load_env()

    data_schematic: DataSchematic = hydra.utils.instantiate(cfg.data_schematic)

    train_datasets = {
        name: hydra.utils.instantiate(cfg.data.train_datasets[name])
        for name in cfg.data.train_datasets
    }
    valid_datasets = {
        name: hydra.utils.instantiate(cfg.data.valid_datasets[name])
        for name in cfg.data.valid_datasets
    }
    datamodule = hydra.utils.instantiate(
        cfg.data,
        train_datasets=train_datasets,
        valid_datasets=valid_datasets,
        train_viz_datasets={},
    )

    for dataset_name, dataset in datamodule.train_datasets.items():
        data_schematic.infer_shapes_from_batch(dataset[0])
        instantiate_copy = copy.deepcopy(cfg.data.train_datasets[dataset_name])
        km = OmegaConf.to_container(instantiate_copy.resolver.key_map, resolve=False)
        km["norm_mode"] = True
        instantiate_copy.resolver.key_map = km
        norm_dataset = hydra.utils.instantiate(instantiate_copy)
        # Stats load from the run's precomputed_norm_path (norm_dataset unused then).
        data_schematic.infer_norm_from_dataset(
            norm_dataset,
            dataset_name,
            sample_frac=OmegaConf.select(cfg, "norm_stats.sample_frac", default=1.0),
            num_workers=OmegaConf.select(cfg, "norm_stats.num_workers", default=4),
            precomputed_norm_path=OmegaConf.select(
                cfg, "norm_stats.precomputed_norm_path", default=None
            ),
        )

    if cfg.reject_outliers:  # false in all 8 runs; kept for config fidelity
        th._propagate_data_schematic_to_datasets(
            data_schematic,
            datamodule.train_datasets,
            bounds_slack=float(
                OmegaConf.select(cfg, "reject_outliers_slack", default=0.0)
            ),
        )

    viz_func_dict = {
        name: hydra.utils.instantiate(v) for name, v in cfg.visualization.items()
    }

    model = ModelWrapper(
        config_tree=th._build_model_config_tree(cfg),
        data_schematic_state=data_schematic.to_state(),
        viz_func=viz_func_dict,
        scheduler_interval=cfg.model.get("scheduler_interval", "step"),
    )

    th._log_dataset_frame_counts(datamodule.train_datasets, datamodule.valid_datasets)

    csv_logger = CSVLogger(save_dir=out_dir, name="csv")
    with open_dict(cfg):
        cfg.trainer.pop("_modal", None)
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=[], logger=[csv_logger])
    os.makedirs(os.path.join(trainer.default_root_dir, "videos"), exist_ok=True)

    eval_obj = hydra.utils.instantiate(cfg.evaluator)
    eval_obj.trainer = trainer
    eval_obj.model = model.model
    model.evaluator = eval_obj

    print(f"[offline_val] loading checkpoint weights (strict=True): {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_epoch = checkpoint.get("epoch")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint

    print(f"[offline_val] validating {run_tag} @ epoch {epoch} (ckpt epoch counter: {ckpt_epoch})")
    trainer.validate(model=model, datamodule=datamodule)

    metrics = {}
    for k, v in trainer.callback_metrics.items():
        try:
            metrics[k] = float(v)
        except (TypeError, ValueError):
            pass

    videos = sorted(
        os.path.relpath(p, out_dir)
        for p in _glob.glob(f"{out_dir}/videos/**/*.mp4", recursive=True)
    )
    manifest = {
        "run_tag": run_tag,
        "epoch": epoch,
        "ckpt_path": ckpt_path,
        "ckpt_epoch_counter": ckpt_epoch,
        "operator": INDIST_OPERATOR,
        "episodes": INDIST_EPISODES,
        "total_frames": INDIST_TOTAL_FRAMES,
        "limit_val_batches": limit_val_batches,
        "git_commit": git_commit,
        "videos": videos,
        "metrics": metrics,
    }
    with open(f"{out_dir}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    training_outputs_volume.commit()
    print(f"[offline_val] DONE {run_tag}: {len(videos)} videos, metrics:")
    print(json.dumps(metrics, indent=2))
    return manifest


_COMMON = dict(
    cpu=12.0,
    memory=131072,
    timeout=7200,
    secrets=[modal.Secret.from_name(name) for name in CFG.secret_names],
    volumes={
        CFG.volume_mount_path: zarr_volume,
        CFG.output_mount_path: training_outputs_volume,
    },
)


@app.function(gpu="A100-80GB", **_COMMON)
def run_val_oss(
    run_tag: str,
    epoch: int,
    git_remote: str,
    git_commit: str,
    limit_val_batches: float = 1.0,
) -> dict:
    return _run_offline_val(
        run_tag,
        epoch,
        git_remote,
        git_commit,
        submodules=frozenset(),
        limit_val_batches=limit_val_batches,
        is_pi=False,
    )


@app.function(gpu="H200", **_COMMON)
def run_val_pi(
    run_tag: str,
    epoch: int,
    git_remote: str,
    git_commit: str,
    limit_val_batches: float = 1.0,
) -> dict:
    return _run_offline_val(
        run_tag,
        epoch,
        git_remote,
        git_commit,
        submodules=frozenset({"openpi"}),
        limit_val_batches=limit_val_batches,
        is_pi=True,
    )


def _fn_for(run_tag: str):
    if run_tag in PI_RUNS:
        return run_val_pi
    if run_tag in OSS_RUNS:
        return run_val_oss
    raise SystemExit(f"unknown run_tag {run_tag!r}; expected one of {OSS_RUNS + PI_RUNS}")


@app.local_entrypoint()
def run(
    run_tag: str = "300M_mm_nobc_dw48",
    epoch: int = COMMON_EPOCH,
    limit_val_batches: float = 1.0,
) -> None:
    """Validate a single run. Blocks and prints the manifest."""
    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print("Warning: local repo dirty; the container runs the last pushed commit.")
    print(f"Submitting offline val: {run_tag} @ epoch {epoch} (commit {git_commit[:12]})")
    manifest = _fn_for(run_tag).remote(
        run_tag, epoch, git_remote, git_commit, limit_val_batches
    )
    print(json.dumps(manifest, indent=2))


@app.local_entrypoint()
def run_many(
    run_tags: str,
    epoch: int = COMMON_EPOCH,
    limit_val_batches: float = 1.0,
) -> None:
    """Validate several runs in parallel (comma-separated run tags)."""
    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print("Warning: local repo dirty; containers run the last pushed commit.")
    tags = [t.strip() for t in run_tags.split(",") if t.strip()]
    handles = {}
    for tag in tags:
        handles[tag] = _fn_for(tag).spawn(
            tag, epoch, git_remote, git_commit, limit_val_batches
        )
        print(f"spawned {tag}: {handles[tag].object_id}")
    failures = {}
    for tag, handle in handles.items():
        try:
            manifest = handle.get()
            print(f"\n=== {tag}: OK, {len(manifest['videos'])} videos ===")
            print(json.dumps(manifest["metrics"], indent=2))
        except Exception as exc:  # keep gathering the rest
            failures[tag] = repr(exc)
            print(f"\n=== {tag}: FAILED: {exc!r} ===")
    if failures:
        raise SystemExit(f"failed runs: {failures}")
