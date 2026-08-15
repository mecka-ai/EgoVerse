"""Offline (weights-only) validation for the 8 data_div_oss runs on Modal.

For a given run_tag + epoch, loads that run's EXACT training config from its
stored `.hydra/config.yaml` on the egoverse-training-outputs volume (the
composed config Hydra wrote at launch; re-composing from overrides.yaml against
today's hydra_configs is NOT safe — e.g. pi0.5_bc_mecka.yaml dropped the
`model.pi05: true` flag after these runs launched, which silently builds a
non-adaRMS action expert that cannot load the checkpoints). It then swaps the
valid episodes for 3 IN-DISTRIBUTION episodes (seen-in-training operator — a
control experiment against the operator-holdout val plateau), loads the
checkpoint weights (strict=True) and runs `trainer.validate` on a single GPU.
The run's own EvalVideo evaluator writes GT-vs-pred videos to
    <outputs volume>/offline_val_indist/<run_tag>/videos/epoch_0/MECKA_BIMANUAL/
and Valid/* metrics land in a CSVLogger + manifest.json in the same dir.

The validation itself runs in a FRESH SUBPROCESS inside the container (a
runner script written at container runtime), exactly like training runs
`python -m egomimic.trainHydra` as a subprocess. Building the pi model
in-process silently produces a NON-adaRMS gemma expert (plain
input_layernorm.weight instead of input_layernorm.dense.*) even though the
transformers overlay is on disk — verified empirically: the same
PI0Pytorch(pi05=True) build in the same container has adaRMS keys in a fresh
subprocess and lacks them in-process. The subprocess also matches training's
import environment (.pth-based egomimic/openpi resolution).

The live training runs and their dirs are NOT touched: their config.yaml /
checkpoints are only read, and all outputs go to the new offline_val_indist/
prefix.

Usage (single run):
    MODAL_ENVIRONMENT=robotics modal run --detach egomimic/modal/offline_val.py::run \
        --run-tag 300M_mm_nobc_dw48 --epoch 539

Several runs in parallel:
    MODAL_ENVIRONMENT=robotics modal run --detach egomimic/modal/offline_val.py::run_many \
        --run-tags pi05_dw48,pali_dw48

Notes
-----
- OSS (HPT) runs validate on A100-80GB, pi0.5-family runs on H200 (they were
  trained on 2 GPUs, but eval_video.py has no rank guard — 2 ranks corrupt the
  videos — so validation is single-GPU: trainer.devices=1, strategy=auto).
- The runner initializes a world_size=1 gloo process group before validate:
  ModelWrapper.on_validation_end calls torch.distributed.barrier()
  unconditionally, which raises under a single-device strategy otherwise.
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


# =============================================================================
# Runner script — written to /root/offline_val_runner.py at container runtime
# and executed as a FRESH python subprocess (see module docstring for why).
# Mirrors the validation-relevant parts of egomimic/trainHydra.py train().
# =============================================================================
RUNNER_SRC = r'''
"""Offline-val runner: fresh-process validation of one run. See offline_val.py."""
import argparse
import copy
import glob
import json
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--config-path", required=True)
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eps-json", required=True)
    ap.add_argument("--limit-val-batches", type=float, default=1.0)
    ap.add_argument("--git-commit", default="")
    args = ap.parse_args()

    import hydra
    import lightning as L
    import torch
    from lightning.pytorch.loggers import CSVLogger
    from omegaconf import OmegaConf, open_dict

    # Importing trainHydra registers the eval/multiply OmegaConf resolvers and
    # applies the DataLoader shm/tmpdir setup — same environment as training.
    import egomimic.trainHydra as th
    from egomimic.pl_utils.pl_model import ModelWrapper
    from egomimic.rldb.zarr.utils import DataSchematic, set_global_seed
    from egomimic.utils.aws.aws_data_utils import load_env
    from egomimic.utils.dataloader_ipc import configure_dataloader_ipc

    # ---- load the run's stored launch config and apply offline-val edits ----
    cfg = OmegaConf.load(args.config_path)
    print(f"[offline_val] loaded launch config: {args.config_path}")
    lvb = (
        float(args.limit_val_batches)
        if args.limit_val_batches <= 1.0
        else int(args.limit_val_batches)
    )
    with open_dict(cfg):
        # all outputs to the offline_val_indist/<run_tag> prefix; this also
        # replaces the ${hydra:runtime.output_dir} interpolations, which cannot
        # resolve outside a hydra app run
        cfg.paths.output_dir = args.out_dir
        cfg.norm_stats.save_cache_dir = None
        # no wandb; a CSVLogger is attached manually below
        cfg.logger = None
        cfg.ckpt_path = None
        cfg.finetune_ckpt = None
        # single GPU: eval_video.py has no rank guard, 2 ranks corrupt videos
        cfg.launch_params.gpus_per_node = 1
        cfg.trainer.devices = 1
        cfg.trainer.num_nodes = 1
        cfg.trainer.strategy = "auto"
        cfg.trainer.num_sanity_val_steps = 0
        cfg.trainer.check_val_every_n_epoch = 1
        cfg.trainer.limit_val_batches = lvb
        # 3 in-dist episodes for the val set; train set restricted to the same
        # 3 episodes — it is never iterated (validate only), but is still
        # needed for shape inference (dataset[0], same embodiment => same
        # shapes) and the norm-stats dataset construction (stats themselves
        # load from the run's precomputed_norm_path, which stays untouched).
        cfg.data.train_datasets.mecka_bimanual.resolver.eps_to_use = args.eps_json
        cfg.data.valid_datasets.mecka_bimanual.resolver.eps_to_use = args.eps_json

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
        # Stats load from the run's precomputed_norm_path (dataset unused then).
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

    csv_logger = CSVLogger(save_dir=args.out_dir, name="csv")
    with open_dict(cfg):
        cfg.trainer.pop("_modal", None)
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=[], logger=[csv_logger])
    os.makedirs(os.path.join(trainer.default_root_dir, "videos"), exist_ok=True)

    eval_obj = hydra.utils.instantiate(cfg.evaluator)
    eval_obj.trainer = trainer
    eval_obj.model = model.model
    model.evaluator = eval_obj

    print(f"[offline_val] loading checkpoint weights (strict=True): {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    ckpt_epoch = checkpoint.get("epoch")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint

    # ModelWrapper.on_validation_end calls torch.distributed.barrier()
    # unconditionally; under a single-device strategy no process group exists
    # and it raises. A world_size=1 gloo group makes the barrier a no-op
    # without touching training code (single-device strategy ignores it).
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29513",
            rank=0,
            world_size=1,
        )
        print("[offline_val] initialized world_size=1 gloo group (barrier no-op)")

    print(
        f"[offline_val] validating {args.run_tag} @ epoch {args.epoch} "
        f"(ckpt epoch counter: {ckpt_epoch})"
    )
    trainer.validate(model=model, datamodule=datamodule)

    metrics = {}
    for k, v in trainer.callback_metrics.items():
        try:
            metrics[k] = float(v)
        except (TypeError, ValueError):
            pass

    videos = sorted(
        os.path.relpath(p, args.out_dir)
        for p in glob.glob(f"{args.out_dir}/videos/**/*.mp4", recursive=True)
    )
    with open(os.path.join(args.out_dir, "indist_meta.json")) as f:
        meta = json.load(f)
    manifest = {
        "run_tag": args.run_tag,
        "epoch": args.epoch,
        "ckpt_path": args.ckpt_path,
        "ckpt_epoch_counter": ckpt_epoch,
        "operator": meta["operator"],
        "episodes": meta["episodes"],
        "total_frames": meta["total_frames"],
        "limit_val_batches": args.limit_val_batches,
        "git_commit": args.git_commit,
        "videos": videos,
        "metrics": metrics,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("[offline_val] RUNNER_DONE " + json.dumps(metrics))


if __name__ == "__main__":
    main()
'''


def _run_offline_val(
    run_tag: str,
    epoch: int,
    git_remote: str,
    git_commit: str,
    submodules: frozenset,
    limit_val_batches: float,
    is_pi: bool,
    out_prefix: str = OUT_PREFIX,
) -> dict:
    """Container body: clone repo, write + run the runner subprocess, commit."""
    import shlex
    import subprocess

    # Clone the repo (+ openpi submodule and its transformers==4.53.2 overlay
    # for pi runs). _prepare_repo registers egomimic (and openpi) via .pth
    # files, so the runner subprocess resolves them like training's DDP ranks.
    _prepare_repo(git_remote=git_remote, git_commit=git_commit, submodules=submodules)

    out_dir = f"{CFG.output_mount_path}/{out_prefix}/{run_tag}"
    os.makedirs(out_dir, exist_ok=True)
    eps_json = f"{out_dir}/offline_val_indist3.json"
    with open(eps_json, "w") as f:
        json.dump(INDIST_EPISODES, f)
    with open(f"{out_dir}/indist_meta.json", "w") as f:
        json.dump(
            {
                "operator": INDIST_OPERATOR,
                "episodes": INDIST_EPISODES,
                "total_frames": INDIST_TOTAL_FRAMES,
            },
            f,
        )

    config_path = f"{CFG.output_mount_path}/{RUN_BASE}/{run_tag}/.hydra/config.yaml"
    ckpt_path = (
        f"{CFG.output_mount_path}/{RUN_BASE}/{run_tag}/checkpoints/"
        f"epoch_epoch={epoch}.ckpt"
    )
    for p in (config_path, ckpt_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    runner_path = "/root/offline_val_runner.py"
    with open(runner_path, "w") as f:
        f.write(RUNNER_SRC)

    env = os.environ.copy()
    env["MODAL_IS_REMOTE"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    if is_pi:
        # pi0.5's sample_actions is @torch.compile'd; the image has no C
        # compiler, so TorchInductor would crash eval (see trainModal.py).
        env["TORCHDYNAMO_DISABLE"] = "1"

    cmd = [
        CFG.python_bin,
        runner_path,
        "--run-tag", run_tag,
        "--epoch", str(epoch),
        "--config-path", config_path,
        "--ckpt-path", ckpt_path,
        "--out-dir", out_dir,
        "--eps-json", eps_json,
        "--limit-val-batches", str(limit_val_batches),
        "--git-commit", git_commit,
    ]
    print(f"[offline_val] running: {shlex.join(cmd)}")
    proc = subprocess.run(cmd, cwd=CFG.remote_repo_dir, env=env, check=False)

    training_outputs_volume.commit()
    if proc.returncode != 0:
        raise RuntimeError(f"offline val runner failed (exit {proc.returncode})")

    with open(f"{out_dir}/manifest.json") as f:
        manifest = json.load(f)
    print(f"[offline_val] DONE {run_tag}: {len(manifest['videos'])} videos, metrics:")
    print(json.dumps(manifest["metrics"], indent=2))
    return manifest


_COMMON = dict(
    cpu=12.0,
    memory=131072,
    # 4h: generous headroom — zarr reads over the volume FUSE mount degrade
    # heavily when several vals hit the same episodes concurrently (observed
    # ~10x); prefer running vals sequentially.
    timeout=14400,
    secrets=[modal.Secret.from_name(name) for name in CFG.secret_names],
    volumes={
        CFG.volume_mount_path: zarr_volume,
        CFG.output_mount_path: training_outputs_volume,
    },
)


# NOTE: originally A100-80GB; switched to H200 mid-campaign — the A100 pool's
# volume FUSE reads were ~10x slower than H200 containers on the same episodes
# (observed 2026-08-14), stalling OSS vals past their timeout.
@app.function(gpu="H200", **_COMMON)
def run_val_oss(
    run_tag: str,
    epoch: int,
    git_remote: str,
    git_commit: str,
    limit_val_batches: float = 1.0,
    out_prefix: str = OUT_PREFIX,
) -> dict:
    return _run_offline_val(
        run_tag,
        epoch,
        git_remote,
        git_commit,
        submodules=frozenset(),
        limit_val_batches=limit_val_batches,
        is_pi=False,
        out_prefix=out_prefix,
    )


@app.function(gpu="H200", **_COMMON)
def run_val_pi(
    run_tag: str,
    epoch: int,
    git_remote: str,
    git_commit: str,
    limit_val_batches: float = 1.0,
    out_prefix: str = OUT_PREFIX,
) -> dict:
    return _run_offline_val(
        run_tag,
        epoch,
        git_remote,
        git_commit,
        submodules=frozenset({"openpi"}),
        limit_val_batches=limit_val_batches,
        is_pi=True,
        out_prefix=out_prefix,
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
    out_prefix: str = OUT_PREFIX,
) -> None:
    """Validate a single run. Blocks and prints the manifest."""
    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print("Warning: local repo dirty; the container runs the last pushed commit.")
    print(
        f"Submitting offline val: {run_tag} @ epoch {epoch} -> {out_prefix}/ "
        f"(commit {git_commit[:12]})"
    )
    manifest = _fn_for(run_tag).remote(
        run_tag, epoch, git_remote, git_commit, limit_val_batches, out_prefix
    )
    print(json.dumps(manifest, indent=2))


@app.local_entrypoint()
def run_many(
    run_tags: str,
    epoch: int = COMMON_EPOCH,
    limit_val_batches: float = 1.0,
    out_prefix: str = OUT_PREFIX,
) -> None:
    """Validate several runs in parallel (comma-separated run tags)."""
    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print("Warning: local repo dirty; containers run the last pushed commit.")
    tags = [t.strip() for t in run_tags.split(",") if t.strip()]
    handles = {}
    for tag in tags:
        handles[tag] = _fn_for(tag).spawn(
            tag, epoch, git_remote, git_commit, limit_val_batches, out_prefix
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
