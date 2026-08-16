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

Operator-scaling levels (op_scaling/L1..L4) over the 8 TRAIN-set episodes that
are shared by all four levels' train splits — one episode per L1 operator:
    MODAL_ENVIRONMENT=robotics modal run --detach \
        egomimic/modal/offline_val.py::run_opscale --run-tag L1 --epoch 779
    MODAL_ENVIRONMENT=robotics modal run --detach \
        egomimic/modal/offline_val.py::run_opscale_many --run-tags L2,L3,L4 --epoch 779

The op-scaling pass differs from the dw48 pass in three parameterized ways:
run_base (op_scaling instead of data_div_oss), the episode list, and
flush_per_episode=True — see `_install_episode_flush` in the runner: instead of
eval_video.py's fixed 1000-frame video chunks it emits exactly ONE video per
val episode, so video index i is the same episode/operator in every level.

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

# ---------------------------------------------------------------------------
# op_scaling (operator-diversity) levels — TRAIN-SET offline val
# ---------------------------------------------------------------------------
# The 4 levels are supersets in OPERATORS (L1 8 ⊂ L2 24 ⊂ L3 72 ⊂ L4 160) but
# each level's train json was built by round-robin over that level's operators,
# so one operator contributes different (and fewer) episodes at higher levels.
# These 8 episodes — one per L1 operator — are in the train split of ALL FOUR
# levels, so every level has literally SEEN all 8: a like-for-like train-set
# (fit) comparison across diversity levels. Listed in episode-hash order, which
# is the order episode_table_to_df / the val dataloader resolve them in.
OPSCALE_RUNS = ("L1", "L2", "L3", "L4")
OPSCALE_RUN_BASE = "op_scaling"
OPSCALE_OUT_PREFIX = "offline_val_opscale"
OPSCALE_COMMON_EPOCH = 779  # largest epoch_epoch=N.ckpt present in all 4 runs
OPSCALE_TRAINVAL_8 = {  # episode_hash -> (operator, num_frames)
    "69b083e65a299178939432ae": ("696a8ab16adfd3c664a65c91", 3590),
    "69b08bef2e8f3cdc83df98da": ("6963a33b83a9fdf2d863cb6b", 3295),
    "69b0a0624d596b45d52ba551": ("6975db9bb393af9134ca5d21", 3598),
    "69b8b2d61cf7f6f00d4364df": ("6954b58920b100982d80f170", 2699),
    "69b8b325a52e1a2126f45ffe": ("6944be8574e27bfb2358061e", 2995),
    "69b8c0beb3cc90fa8d9ac9ef": ("6776bf817d12b76c8e1be433", 3601),
    "69b92e31e749b83b1a333011": ("6968000b0af001daaaad5168", 2605),
    "69b9f1670ed8c646a6770f85": ("6980e0c57c6b5a6b3c8cf16c", 3295),
}


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
import math
import os
import sys
import types


def _install_episode_flush(eval_obj, datamodule, cfg):
    """Emit exactly ONE validation video per val episode.

    eval_video.EvalVideo.on_validation_step flushes its frame buffer every
    1000 buffered frames, so with N multi-thousand-frame episodes the videos
    are fixed-size chunks whose boundaries have nothing to do with episodes.
    For a cross-run comparison we want video i == episode i in every run, so
    this replaces on_validation_step (on the instance only — the repo's
    evaluator classes are untouched) with the same body plus a flush at the
    batch that crosses each episode boundary, and no size-based flush.

    Boundaries are derived from the val MultiDataset's own index_map layout:
    datasets are concatenated in dict order (== episode-hash order, the order
    episode_table_to_df pins) and the val DataLoader is shuffle=False, so the
    global sample range of episode k is known exactly. A batch straddling a
    boundary is attributed to the earlier episode, i.e. each video may carry
    up to batch_size-1 frames of the next episode.
    """
    import torch
    import torchvision.io as tvio

    from egomimic.rldb.embodiment.embodiment import get_embodiment

    ds_name = next(iter(datamodule.valid_datasets))
    val_ds = datamodule.valid_datasets[ds_name]
    batch_size = int(cfg.data.valid_dataloader_params[ds_name].batch_size)

    ep_names = list(val_ds.datasets.keys())
    ep_lens = [len(val_ds.datasets[n]) for n in ep_names]
    flush_after = {}  # batch_idx -> episode hash whose video is closed here
    cum = 0
    for name, ln in zip(ep_names, ep_lens):
        cum += ln
        flush_after[math.ceil(cum / batch_size) - 1] = name
    print(
        "[offline_val] per-episode video flush: "
        + json.dumps(
            {
                "batch_size": batch_size,
                "episodes": ep_names,
                "frames": ep_lens,
                "flush_after_batch_idx": {str(k): v for k, v in flush_after.items()},
            }
        )
    )

    def _write(self, key):
        buf = self.val_image_buffer.get(key) or []
        if not buf:
            return
        out = os.path.join(
            self.video_dir(),
            f"epoch_{self.trainer.current_epoch}",
            str(get_embodiment(key)),
            f"validation_video_{self.val_counter[key]}.mp4",
        )
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tvio.write_video(out, torch.stack(buf), fps=30, video_codec="h264")
        print(f"[offline_val] wrote {out} ({len(buf)} frames)")
        buf.clear()
        self.val_counter[key] += 1

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        metrics, images_dict = self.compute_metrics_and_viz(batch)
        device = self.trainer.lightning_module.device
        metrics = {
            k: (v.to(device) if torch.is_tensor(v) else torch.tensor(v, device=device))
            for k, v in metrics.items()
        }
        for key, images in images_dict.items():
            if self.val_image_buffer.get(key) is None:
                self.val_image_buffer[key] = []
                self.val_counter[key] = 0
            self.val_image_buffer[key].extend(torch.from_numpy(images))
        if batch_idx in flush_after:
            for key in list(self.val_image_buffer):
                _write(self, key)
        self.trainer.lightning_module.log_dict(
            metrics, sync_dist=True, add_dataloader_idx=False
        )

    eval_obj.on_validation_step = types.MethodType(on_validation_step, eval_obj)
    return [{"episode": n, "frames": l} for n, l in zip(ep_names, ep_lens)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--config-path", required=True)
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eps-json", required=True)
    ap.add_argument("--meta-json", default="")
    ap.add_argument("--limit-val-batches", type=float, default=1.0)
    ap.add_argument("--flush-per-episode", action="store_true")
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
        # SINGLE val set. The op_scaling runs ship a second val loader
        # (train_viz = held-out-operator, logged as Valid_oph/* into videos_oph/
        # by train_viz_evaluator). Drop all of it so dataloader_idx=1 does not
        # exist: videos land in the standard videos/ dir and metrics keep the
        # plain Valid/* names. No-ops for the dw48 runs, which have none.
        cfg.data.pop("train_viz_datasets", None)
        cfg.data.pop("train_viz_dataloader_params", None)
        cfg.pop("train_viz_evaluator", None)
        cfg.pop("second_val_prefix", None)

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

    episode_layout = None
    if args.flush_per_episode:
        episode_layout = _install_episode_flush(eval_obj, datamodule, cfg)

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
    meta_path = args.meta_json or os.path.join(args.out_dir, "indist_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    manifest = {
        "run_tag": args.run_tag,
        "epoch": args.epoch,
        "ckpt_path": args.ckpt_path,
        "ckpt_epoch_counter": ckpt_epoch,
        "operator": meta.get("operator"),
        "episodes": meta["episodes"],
        "episode_meta": meta.get("episode_meta"),
        "episode_layout": episode_layout,
        "total_frames": meta["total_frames"],
        "limit_val_batches": args.limit_val_batches,
        "flush_per_episode": bool(args.flush_per_episode),
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
    run_base: str = RUN_BASE,
    episode_meta: dict | None = None,
    flush_per_episode: bool = False,
) -> dict:
    """Container body: clone repo, write + run the runner subprocess, commit.

    episode_meta: {episode_hash: {"operator": ..., "num_frames": ...}} for the
    val set; defaults to the dw48 in-distribution 3-episode set. run_base is
    the volume prefix the run dirs live under (data_div_oss / op_scaling).
    """
    import shlex
    import subprocess

    # Clone the repo (+ openpi submodule and its transformers==4.53.2 overlay
    # for pi runs). _prepare_repo registers egomimic (and openpi) via .pth
    # files, so the runner subprocess resolves them like training's DDP ranks.
    _prepare_repo(git_remote=git_remote, git_commit=git_commit, submodules=submodules)

    if episode_meta is None:
        episode_meta = {
            h: {"operator": INDIST_OPERATOR} for h in INDIST_EPISODES
        }
    # Hash order == the order episode_table_to_df pins, i.e. the order the
    # shuffle=False val dataloader walks the episodes in.
    episodes = sorted(episode_meta)
    total_frames = (
        sum(int(m.get("num_frames", 0)) for m in episode_meta.values())
        or INDIST_TOTAL_FRAMES
    )

    out_dir = f"{CFG.output_mount_path}/{out_prefix}/{run_tag}"
    os.makedirs(out_dir, exist_ok=True)
    eps_json = f"{out_dir}/val_episodes.json"
    with open(eps_json, "w") as f:
        json.dump(episodes, f)
    meta_json = f"{out_dir}/val_meta.json"
    with open(meta_json, "w") as f:
        json.dump(
            {
                "operator": episode_meta[episodes[0]].get("operator"),
                "episodes": episodes,
                "episode_meta": episode_meta,
                "total_frames": total_frames,
            },
            f,
        )

    config_path = f"{CFG.output_mount_path}/{run_base}/{run_tag}/.hydra/config.yaml"
    ckpt_path = (
        f"{CFG.output_mount_path}/{run_base}/{run_tag}/checkpoints/"
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
        "--meta-json", meta_json,
        "--limit-val-batches", str(limit_val_batches),
        "--git-commit", git_commit,
    ]
    if flush_per_episode:
        cmd.append("--flush-per-episode")
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
    run_base: str = RUN_BASE,
    episode_meta: dict | None = None,
    flush_per_episode: bool = False,
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
        run_base=run_base,
        episode_meta=episode_meta,
        flush_per_episode=flush_per_episode,
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


# ---------------------------------------------------------------------------
# op_scaling entrypoints — TRAIN-SET val over the 8 shared episodes
# ---------------------------------------------------------------------------


def _opscale_episode_meta() -> dict:
    return {
        h: {"operator": op, "num_frames": nf}
        for h, (op, nf) in OPSCALE_TRAINVAL_8.items()
    }


def _opscale_kwargs(limit_val_batches: float, out_prefix: str) -> dict:
    return dict(
        limit_val_batches=limit_val_batches,
        out_prefix=out_prefix,
        run_base=OPSCALE_RUN_BASE,
        episode_meta=_opscale_episode_meta(),
        flush_per_episode=True,
    )


@app.local_entrypoint()
def run_opscale(
    run_tag: str = "L1",
    epoch: int = OPSCALE_COMMON_EPOCH,
    limit_val_batches: float = 1.0,
    out_prefix: str = OPSCALE_OUT_PREFIX,
) -> None:
    """Train-set offline val for ONE op_scaling level (L1..L4). Blocks."""
    if run_tag not in OPSCALE_RUNS:
        raise SystemExit(f"unknown level {run_tag!r}; expected one of {OPSCALE_RUNS}")
    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print("Warning: local repo dirty; the container runs the last pushed commit.")
    print(
        f"Submitting op_scaling train-set offline val: {run_tag} @ epoch {epoch} "
        f"-> {out_prefix}/{run_tag} (commit {git_commit[:12]})"
    )
    manifest = run_val_oss.remote(
        run_tag, epoch, git_remote, git_commit, **_opscale_kwargs(limit_val_batches, out_prefix)
    )
    print(json.dumps(manifest, indent=2))


@app.local_entrypoint()
def run_opscale_many(
    run_tags: str = "L1,L2,L3,L4",
    epoch: int = OPSCALE_COMMON_EPOCH,
    limit_val_batches: float = 1.0,
    out_prefix: str = OPSCALE_OUT_PREFIX,
) -> None:
    """Train-set offline val for several op_scaling levels in parallel."""
    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print("Warning: local repo dirty; containers run the last pushed commit.")
    tags = [t.strip() for t in run_tags.split(",") if t.strip()]
    for tag in tags:
        if tag not in OPSCALE_RUNS:
            raise SystemExit(f"unknown level {tag!r}; expected one of {OPSCALE_RUNS}")
    kwargs = _opscale_kwargs(limit_val_batches, out_prefix)
    handles = {
        tag: run_val_oss.spawn(tag, epoch, git_remote, git_commit, **kwargs)
        for tag in tags
    }
    for tag, h in handles.items():
        print(f"spawned {tag}: {h.object_id}")
    failures = {}
    for tag, handle in handles.items():
        try:
            manifest = handle.get()
            print(f"\n=== {tag}: OK, {len(manifest['videos'])} videos ===")
            print(json.dumps(manifest["metrics"], indent=2))
        except Exception as exc:
            failures[tag] = repr(exc)
            print(f"\n=== {tag}: FAILED: {exc!r} ===")
    if failures:
        raise SystemExit(f"failed levels: {failures}")
