"""Offline WAM (dreamzero) evaluation / val-sweep on Modal.

Thin Modal wrappers around ``egomimic.eval.eval_dreamzero`` (single-checkpoint
offline eval with AR or TF rolling) and ``egomimic.eval.val_sweep`` (metric
sweep over every ``epoch_epoch=*.ckpt`` in a run's checkpoints folder). The
container clones the repo at the pushed HEAD commit and runs the eval as a
python -m subprocess with the hydra args you pass through — the same
composition path as training, so a checkpoint trained via trainModal evaluates
against the identical config stack.

Outputs land on the ``egoverse-training-outputs`` volume under
``logs/<name>/<description>_<timestamp>`` (the hydra run dir is relative to
the repo, and the volume is mounted at ``<repo>/logs``):
    <run dir>/videos/epoch_0/MECKA_BIMANUAL/{predicted,validation}_video_*.mp4
    <run dir>/.hydra/{config,overrides}.yaml
    (val_sweep: videos/ckpt_epoch_<N>/... + val_sweep_metrics.json)

Usage (offline eval of one checkpoint, TF rolling, 3 held-out episodes):
    MODAL_ENVIRONMENT=robotics modal run --detach \
        egomimic/modal/offline_val_wam.py::run --hydra-args "\
        data=data_dishwashing_48h_wam evaluator=eval_dreamzero_tf \
        data_schematic.norm_mode=minmax reject_outliers=false \
        ckpt_path=data_div_oss/wam22_dw48/checkpoints/last.ckpt \
        +num_val_episodes=3 name=wam_offline_eval description=wam22_dw48_tf"

    (evaluator=eval_dreamzero_ar for fully-autoregressive rolling; relative
    ckpt_path / checkpoints_dir / norm paths resolve against the outputs
    volume mount.)

Usage (val-metric sweep over a run's checkpoints):
    MODAL_ENVIRONMENT=robotics modal run --detach \
        egomimic/modal/offline_val_wam.py::sweep --hydra-args "\
        data=data_dishwashing_48h_wam \
        data_schematic.norm_mode=minmax reject_outliers=false \
        +checkpoints_dir=data_div_oss/wam22_dw48/checkpoints \
        +num_val_episodes=5 name=wam_val_sweep description=wam22_dw48"
"""

from __future__ import annotations

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
    WAN_CKPT_MOUNT,
    _local_wandb_key,
    _prepare_repo,
    _resolve_git_state,
    image,
    training_outputs_volume,
    wan_checkpoints_volume,
    zarr_volume,
)

app = modal.App("egomimic-offline-val-wam", image=image)

# Hydra keys whose values are paths on the outputs volume; relative values get
# prefixed with the volume mount (same convention as trainModal).
_PATH_KEYS = {
    "ckpt_path",
    "checkpoints_dir",
    "norm_stats.precomputed_norm_path",
}


def _resolve_volume_paths(hydra_args: list[str]) -> list[str]:
    fixed = []
    for arg in hydra_args:
        key, sep, val = arg.partition("=")
        if (
            sep
            and key.lstrip("+") in _PATH_KEYS
            and val
            and val != "null"
            and not val.startswith("/")
        ):
            arg = f"{key}={CFG.output_mount_path}/{val}"
        fixed.append(arg)
    return fixed


def _run_module(
    module: str,
    hydra_args: list[str],
    git_remote: str,
    git_commit: str,
    wandb_api_key: str = "",
) -> None:
    import subprocess

    _prepare_repo(git_remote=git_remote, git_commit=git_commit)

    hydra_args = _resolve_volume_paths(hydra_args)
    cmd = [CFG.python_bin, "-m", module, *hydra_args]
    env = os.environ.copy()
    env["MODAL_IS_REMOTE"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    if wandb_api_key:
        env["WANDB_API_KEY"] = wandb_api_key
        env.setdefault("WANDB_START_METHOD", "thread")

    import shlex

    print(f"[offline_val_wam] running: {shlex.join(cmd)}")
    proc = subprocess.run(cmd, cwd=CFG.remote_repo_dir, env=env, check=False)

    training_outputs_volume.commit()
    if proc.returncode != 0:
        raise RuntimeError(f"{module} failed (exit {proc.returncode})")
    print(f"[offline_val_wam] {module} done; outputs committed to volume.")


_COMMON = dict(
    cpu=12.0,
    memory=131072,
    timeout=14400,
    secrets=[modal.Secret.from_name(name) for name in CFG.secret_names],
    volumes={
        CFG.volume_mount_path: zarr_volume,
        CFG.output_mount_path: training_outputs_volume,
        WAN_CKPT_MOUNT: wan_checkpoints_volume,
    },
)


@app.function(gpu="H200", **_COMMON)
def run_eval_dreamzero(
    hydra_args: list[str],
    git_remote: str,
    git_commit: str,
    wandb_api_key: str = "",
) -> None:
    _run_module(
        "egomimic.eval.eval_dreamzero", hydra_args, git_remote, git_commit,
        wandb_api_key=wandb_api_key,
    )


@app.function(gpu="H200", **_COMMON)
def run_val_sweep(
    hydra_args: list[str],
    git_remote: str,
    git_commit: str,
    wandb_api_key: str = "",
) -> None:
    _run_module(
        "egomimic.eval.val_sweep", hydra_args, git_remote, git_commit,
        wandb_api_key=wandb_api_key,
    )


def _split_args(hydra_args: str) -> list[str]:
    import shlex

    args = shlex.split(hydra_args or "")
    if not args:
        raise SystemExit(
            "--hydra-args is required, e.g. --hydra-args "
            "'data=data_dishwashing_48h_wam evaluator=eval_dreamzero_tf "
            "ckpt_path=data_div_oss/wam22_dw48/checkpoints/last.ckpt'"
        )
    return args


@app.local_entrypoint()
def run(hydra_args: str = "") -> None:
    """Offline eval of one WAM checkpoint (eval_dreamzero)."""
    args = _split_args(hydra_args)
    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print("Warning: local repo dirty; the container runs the last pushed commit.")
    print(f"Submitting eval_dreamzero (commit {git_commit[:12]}): {args}")
    run_eval_dreamzero.remote(args, git_remote, git_commit, _local_wandb_key())


@app.local_entrypoint()
def sweep(hydra_args: str = "") -> None:
    """Val-metric sweep over a run's checkpoints folder (val_sweep)."""
    args = _split_args(hydra_args)
    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print("Warning: local repo dirty; the container runs the last pushed commit.")
    print(f"Submitting val_sweep (commit {git_commit[:12]}): {args}")
    run_val_sweep.remote(args, git_remote, git_commit, _local_wandb_key())
