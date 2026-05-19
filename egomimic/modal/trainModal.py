"""Modal training entrypoints for EgoVerse.

Usage
-----
Submit a training run (fire-and-forget):
    python egomimic/modal/trainModal.py \\
        data=mecka_all_zarr trainer=ddp_modal logger=wandb model=hpt_bc_flow_mecka \\
        name=<run> description=<desc> [+modal_gpu=H100] [+modal_cpu=32] [+modal_memory_gb=128]

Blocking run (streams logs, downloads artifacts when done):
    modal run --env robotics egomimic/modal/trainModal.py::run -- \\
        data=mecka_all_zarr trainer=ddp_modal logger=wandb model=hpt_bc_flow_mecka

Verify container health:
    modal run --env robotics egomimic/modal/trainModal.py::verify
"""

from __future__ import annotations

import glob
import json as _json
import os
import shlex
import subprocess
import sys
import time as _time
from pathlib import Path

import modal

# modal_setup.py lives next to this file locally (egomimic/modal/) and is baked
# into the image at /root/ so it is importable before the repo is cloned.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from modal_setup import (  # noqa: E402
    CFG,
    REPO_ROOT,
    _git_commit_and_push,
    _local_wandb_key,
    _prepare_repo,
    _resolve_git_state,
    app,
    training_outputs_volume,
    zarr_volume,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_train_cmd(hydra_args: tuple[str, ...]) -> list[str]:
    return [CFG.python_bin, CFG.train_script, *hydra_args]


def _resolve_volume_paths(hydra_args: tuple[str, ...]) -> tuple[str, ...]:
    """Rewrite relative path overrides to absolute container paths."""
    _PATH_KEYS = {"ckpt_path", "norm_stats.precomputed_norm_path"}
    fixed = []
    for arg in hydra_args:
        key, sep, val = arg.partition("=")
        if sep and key in _PATH_KEYS and val and val != "null" and not val.startswith("/"):
            val = f"{CFG.output_mount_path}/{val}"
            arg = f"{key}={val}"
        fixed.append(arg)
    return tuple(fixed)


def _download_run_artifacts(output_rel_path: str) -> None:
    local_dest = REPO_ROOT / "modal-outputs" / output_rel_path
    local_dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading artifacts to {local_dest} ...")
    result = subprocess.run(
        [
            sys.executable, "-m", "modal", "volume", "get",
            "--env", "robotics",
            "egoverse-training-outputs",
            output_rel_path,
            str(local_dest),
        ],
        cwd=REPO_ROOT,
    )
    if result.returncode == 0:
        print(f"Artifacts saved to: {local_dest.resolve()}")
    else:
        print(
            f"Download failed — pull manually:\n"
            f"  modal volume get --env robotics egoverse-training-outputs "
            f'"{output_rel_path}" "{local_dest}"'
        )


# ---------------------------------------------------------------------------
# Remote functions
# ---------------------------------------------------------------------------


@app.function(
    gpu=CFG.gpu,
    cpu=CFG.cpu,
    memory=CFG.memory_mb,
    timeout=CFG.timeout_seconds,
    secrets=[modal.Secret.from_name(name) for name in CFG.secret_names],
    volumes={
        CFG.volume_mount_path: zarr_volume,
        CFG.output_mount_path: training_outputs_volume,
    },
)
def run_hydra_train(
    hydra_args: tuple[str, ...],
    git_remote: str,
    git_commit: str,
    wandb_api_key: str = "",
) -> str:
    """Clone the repo at *git_commit* and run trainHydra.py with *hydra_args*.

    Returns the path relative to the output volume where artifacts were written.
    """
    _prepare_repo(git_remote=git_remote, git_commit=git_commit)

    hydra_args = _resolve_volume_paths(hydra_args)
    cmd = _build_train_cmd(hydra_args)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    if wandb_api_key:
        env["WANDB_API_KEY"] = wandb_api_key

    env["MODAL_IS_REMOTE"] = "1"
    env["MODAL_TIMEOUT_SECONDS"] = str(CFG.timeout_seconds)
    env["MODAL_START_TIME"] = str(_time.time())
    env["MODAL_HYDRA_ARGS"] = _json.dumps(list(hydra_args))
    env["MODAL_GIT_REMOTE"] = git_remote
    env["MODAL_GIT_COMMIT"] = git_commit

    print(f"Running: {shlex.join(cmd)}")
    process = subprocess.run(cmd, cwd=CFG.remote_repo_dir, env=env, check=False)

    zarr_volume.commit()

    all_run_dirs = sorted(
        glob.glob(f"{CFG.output_mount_path}/*/*"),
        key=os.path.getmtime,
    )
    output_rel_path = (
        os.path.relpath(all_run_dirs[-1], CFG.output_mount_path) if all_run_dirs else ""
    )
    training_outputs_volume.commit()

    if process.returncode != 0:
        raise RuntimeError(
            f"Training failed (exit {process.returncode}): {shlex.join(cmd)}"
        )

    return output_rel_path


@app.function(
    secrets=[modal.Secret.from_name(name) for name in CFG.secret_names],
    volumes={CFG.volume_mount_path: zarr_volume},
    timeout=120,
)
def _health_check() -> dict:
    """Verify secrets, volume mount, and s5cmd from inside the container."""
    results = {}

    for key in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT_URL"):
        results[key] = "OK" if os.environ.get(key) else "MISSING"

    results["MONGODB_URI"] = "OK" if os.environ.get("MONGODB_URI") else "MISSING"

    probe = f"{CFG.volume_mount_path}/.modal_health_probe"
    try:
        open(probe, "w").close()
        os.remove(probe)
        results["volume"] = f"OK — mounted at {CFG.volume_mount_path}"
    except Exception as e:
        results["volume"] = f"ERROR: {e}"

    import subprocess as _sp
    r = _sp.run(["s5cmd", "version"], capture_output=True, text=True)
    results["s5cmd"] = f"OK — {r.stdout.strip()}" if r.returncode == 0 else "MISSING"

    return results


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def verify() -> None:
    """Boot the container and verify all secrets, volume, and s5cmd."""
    print("Running container health check...")
    results = _health_check.remote()
    all_ok = True
    for k, v in results.items():
        symbol = "✓" if v.startswith("OK") else "✗"
        print(f"  {symbol}  {k}: {v}")
        if not v.startswith("OK"):
            all_ok = False
    print()
    if all_ok:
        print("All checks passed — Modal setup is ready.")
    else:
        raise SystemExit("One or more checks failed.")


@app.local_entrypoint()
def submit(*hydra_args: str) -> None:
    """Fire-and-forget: spawn a training job from already-pushed commit."""
    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print("Warning: local repo has uncommitted changes. Modal will run the last committed state only.")
    print(f"Submitting commit {git_commit[:12]} from {git_remote}")
    handle = run_hydra_train.spawn(
        tuple(hydra_args), git_remote, git_commit, _local_wandb_key()
    )
    print(f"Submitted Modal job: {handle.object_id}")
    print("Monitor at: https://modal.com/apps/egomimic-training")
    print(
        "After completion, download artifacts:\n"
        "  modal volume get --env robotics egoverse-training-outputs <run-path> ./modal-outputs/"
    )


if __name__ == "__main__":
    """
    Main submission entrypoint: python egomimic/modal/trainModal.py <hydra_args>

    Parses +modal_gpu=, +modal_cpu=, +modal_memory_gb= from args, auto-commits
    and pushes, then re-invokes `modal run trainModal.py::submit` as a subprocess
    so CFG resource values are resolved after the env vars are set.

    Examples:
        python egomimic/modal/trainModal.py \\
            data=mecka_all_zarr trainer=ddp_modal logger=wandb \\
            model=hpt_bc_flow_mecka name=my_run description=test

        python egomimic/modal/trainModal.py \\
            data=mecka_all_zarr trainer=ddp_modal logger=wandb \\
            model=hpt_bc_flow_mecka name=my_run description=test \\
            +modal_gpu=H100 +modal_cpu=32 +modal_memory_gb=128
    """
    import subprocess

    _MODAL_KEY_MAP = {
        "modal_gpu": "MODAL_GPU",
        "modal_cpu": "MODAL_CPU",
        "modal_memory_gb": "MODAL_MEMORY_GB",
        "modal_memory_mb": "MODAL_MEMORY_MB",
        "modal_volume": "MODAL_ZARR_VOLUME",
    }

    modal_env = os.environ.copy()
    container_overrides = []
    gpu_count = 1

    for arg in sys.argv[1:]:
        key, sep, val = arg.lstrip("+").partition("=")
        if sep and key in _MODAL_KEY_MAP:
            modal_env[_MODAL_KEY_MAP[key]] = val
            if key == "modal_gpu":
                gpu_count = int(val.split(":")[1]) if ":" in val else 1
        else:
            container_overrides.append(arg)

    # Keep DDP launcher in sync with requested GPU count
    container_overrides = [
        a for a in container_overrides
        if not a.lstrip("+").startswith("launch_params.gpus_per_node=")
    ]
    container_overrides.append(f"launch_params.gpus_per_node={gpu_count}")

    gpu = modal_env.get("MODAL_GPU", "A100")
    cpu = modal_env.get("MODAL_CPU", "12")
    mem = modal_env.get("MODAL_MEMORY_GB") or str(
        int(modal_env.get("MODAL_MEMORY_MB", "65536")) // 1024
    )
    print(f"Modal resources: gpu={gpu}  cpu={cpu}  memory={mem}GB")

    _git_commit_and_push(REPO_ROOT)

    cmd = [
        sys.executable, "-m", "modal", "run",
        "--detach", "--env", "robotics",
        str(Path(__file__).resolve()) + "::submit",
        "--", *container_overrides,
    ]
    print(f"Dispatching: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=modal_env)
    sys.exit(result.returncode)


@app.local_entrypoint()
def run(*hydra_args: str) -> None:
    """Blocking run: streams logs and downloads artifacts when complete."""
    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print("Warning: local repo has uncommitted changes. Modal will run the last committed state only.")
    print(f"Running commit {git_commit[:12]} from {git_remote}")
    output_rel_path = run_hydra_train.remote(
        tuple(hydra_args), git_remote, git_commit, _local_wandb_key()
    )
    print(f"Remote run completed. Output path in volume: {output_rel_path}")
    if output_rel_path:
        _download_run_artifacts(output_rel_path)
