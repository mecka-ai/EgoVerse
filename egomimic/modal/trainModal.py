"""Modal training entrypoints for EgoVerse.

Usage
-----
Submit a training run (fully detached — survives local disconnects):
    python egomimic/modal/trainModal.py \\
        data=mecka_all_zarr trainer=ddp_modal logger=wandb model=hpt_bc_flow_mecka \\
        name=<run> description=<desc> [+modal_gpu=H100] [+modal_cpu=32] [+modal_memory_gb=128] \\
        [init_submodules=false]

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
    GS_MOUNT_PATH,
    REPO_ROOT,
    VOLUME_MAP,
    _local_wandb_key,
    _prepare_repo,
    _resolve_git_state,
    _uses_pi_model,
    app,
    app_name_from_hydra_args,
    gs_volume,
    launch_detached,
    pop_init_submodules,
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
    _PATH_KEYS = {
        "ckpt_path",
        "norm_stats.precomputed_norm_path",
        "model.robomimic_model.config.paligemma_weight_path",
        "model.robomimic_model.config.pytorch_weight_path",
    }
    fixed = []
    for arg in hydra_args:
        key, sep, val = arg.partition("=")
        if (
            sep
            and key in _PATH_KEYS
            and val
            and val != "null"
            and not val.startswith("/")
        ):
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
            sys.executable,
            "-m",
            "modal",
            "volume",
            "get",
            "--env",
            "robotics",
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


def _build_volumes() -> dict:
    """Build the volumes dict for the training function.

    Respects +modal_volume=<name> override (e.g. mecka_data_zip).
    Falls back to the default zarr volume (mecka_data_v2).
    """
    vol_name = os.environ.get("MODAL_VOLUME", "mecka_data_v2")
    vol_obj, mount_path = VOLUME_MAP.get(vol_name, (zarr_volume, CFG.volume_mount_path))
    return {
        mount_path: vol_obj,
        CFG.output_mount_path: training_outputs_volume,
        GS_MOUNT_PATH: gs_volume,
    }


def _ephemeral_disk_mib() -> int | None:
    """Return ephemeral disk size in MiB if +modal_ephemeral_disk_gb was set.

    Modal's ephemeral_disk parameter is in MiB. Minimum allowed is 524288 MiB (512 GiB).
    Use +modal_ephemeral_disk_gb=600 or more when extracting shards to /tmp at training.
    """
    gb = os.environ.get("MODAL_EPHEMERAL_DISK_GB")
    return int(float(gb) * 1024) if gb else None


_ephemeral = _ephemeral_disk_mib()
_volumes = _build_volumes()


@app.function(
    gpu=CFG.gpu,
    cpu=CFG.cpu,
    memory=CFG.memory_mb,
    timeout=CFG.timeout_seconds,
    ephemeral_disk=_ephemeral,
    secrets=[modal.Secret.from_name(name) for name in CFG.secret_names],
    volumes=_volumes,
)
def run_hydra_train(
    hydra_args: tuple[str, ...],
    git_remote: str,
    git_commit: str,
    wandb_api_key: str = "",
    init_submodules: bool = True,
) -> str:
    """Clone the repo at *git_commit* and run trainHydra.py with *hydra_args*.

    Returns the path relative to the output volume where artifacts were written.
    """
    _prepare_repo(
        git_remote=git_remote,
        git_commit=git_commit,
        init_submodules=init_submodules,
    )

    # (openpi's patched transformers==4.53.2 for pi0.5 is applied inside
    # _prepare_repo when the openpi submodule is present.)
    hydra_args = _resolve_volume_paths(hydra_args)
    cmd = _build_train_cmd(hydra_args)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env.setdefault("WANDB_START_METHOD", "thread")
    # openpi's pi0.5 sampler (nets["policy"].sample_actions, used in
    # PI.forward_eval) is wrapped in @torch.compile. The Modal image ships no C
    # compiler, so TorchInductor crashes eval with "Failed to find C compiler".
    # Force eager for pi runs — training (forward_training) is uncompiled and
    # unaffected; only eval sampling runs slightly slower.
    if _uses_pi_model(hydra_args):
        env.setdefault("TORCHDYNAMO_DISABLE", "1")
    # Tensor IPC temp files → NVMe /cache (trainHydra enables file_system when TMPDIR is set).
    # Prefer /cache/torch_tmp whenever /cache is writable (ephemeral disk or not).
    _torch_tmp = "/cache/torch_tmp"
    try:
        os.makedirs(_torch_tmp, exist_ok=True)
        test_file = os.path.join(_torch_tmp, ".write_test")
        with open(test_file, "w") as f:
            f.write("1")
        os.remove(test_file)
        env["TMPDIR"] = _torch_tmp
        print(
            f"DataLoader IPC: TMPDIR={_torch_tmp} (use +modal_ephemeral_disk_gb=600 if /cache is missing)"
        )
    except OSError:
        if _ephemeral:
            raise
        print(
            "WARNING: /cache/torch_tmp not writable — DataLoader will use /dev/shm. "
            "Add +modal_ephemeral_disk_gb=600 to avoid SIGBUS with num_workers>8."
        )
    # Prevent thread explosion: many DataLoader workers × many CPU cores otherwise.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    if wandb_api_key:
        env["WANDB_API_KEY"] = wandb_api_key

    env["MODAL_IS_REMOTE"] = "1"
    env["MODAL_TIMEOUT_SECONDS"] = str(CFG.timeout_seconds)
    env["MODAL_START_TIME"] = str(_time.time())
    env["MODAL_HYDRA_ARGS"] = _json.dumps(list(hydra_args))
    env["MODAL_GIT_REMOTE"] = git_remote
    env["MODAL_GIT_COMMIT"] = git_commit
    # Carried into the ModalAutoRestart continuation spawn so a run launched
    # with init_submodules=false doesn't re-init submodules after restarting.
    env["MODAL_INIT_SUBMODULES"] = "1" if init_submodules else "0"

    # openpi (used by egomimic.algo.pi for pi0.5 models) lives in the
    # external/openpi git submodule at external/openpi/src; put it on PYTHONPATH
    # so `import openpi` resolves. Its jax/flax deps are baked into the image.
    _openpi_src = f"{CFG.remote_repo_dir}/external/openpi/src"
    env["PYTHONPATH"] = (
        f"{_openpi_src}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else _openpi_src
    )

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


@app.function(
    cpu=4.0,
    memory=49152,
    timeout=1200,
    secrets=[modal.Secret.from_name(name) for name in CFG.secret_names],
)
def _verify_pi_import(git_remote: str, git_commit: str) -> dict:
    """CPU-only: clone + apply pi transformers swap, then CONSTRUCT PI0Pytorch.

    Mirrors run_hydra_train's setup (clone + .pth + transformers==4.53.2 overlay)
    and actually builds the pi0.5 pytorch model — which is where the
    transformers_replace check fires — so we validate the full pi path without a
    GPU. No PYTHONPATH: relies on the site-packages .pth (the DDP-child path).
    """
    # _prepare_repo applies the pi transformers swap (openpi submodule present).
    _prepare_repo(git_remote=git_remote, git_commit=git_commit, init_submodules=True)

    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    script = (
        "import transformers; print('transformers', transformers.__version__);"
        "import openpi.models.pi0_config as c;"
        "import openpi.models_pytorch.pi0_pytorch as p;"
        "cfg=c.Pi0Config(dtype='bfloat16', action_dim=32, action_horizon=100,"
        " max_token_len=180, paligemma_variant='gemma_2b',"
        " action_expert_variant='gemma_300m', pi05=True);"
        "m=p.PI0Pytorch(cfg);"
        "print('PI0_CONSTRUCT_OK params=', sum(x.numel() for x in m.parameters()))"
    )
    proc = subprocess.run(
        [CFG.python_bin, "-c", script],
        cwd="/",  # from / so CWD doesn't accidentally add the repo to sys.path
        env=env,
        capture_output=True,
        text=True,
    )
    return {
        "returncode": proc.returncode,
        "ok": "PI0_CONSTRUCT_OK" in proc.stdout,
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-4000:],
    }


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
def verify_pi() -> None:
    """Clone on a cheap CPU container and confirm egomimic.algo.pi imports."""
    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print("Warning: local repo dirty; Modal verifies the last committed state.")
    print(f"Verifying pi import for commit {git_commit[:12]} ...")
    res = _verify_pi_import.remote(git_remote, git_commit)
    if res["ok"]:
        print("✓ PI import OK:", res["stdout"].strip().splitlines()[-1])
    else:
        print(f"✗ PI import FAILED (exit {res['returncode']})")
        print("--- stderr (tail) ---")
        print(res["stderr"])
        raise SystemExit("pi import failed — see traceback above.")


@app.local_entrypoint()
def submit(*hydra_args: str) -> None:
    """Fire-and-forget: spawn a training job from already-pushed commit."""
    hydra_args, init_submodules = pop_init_submodules(hydra_args)
    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print(
            "Warning: local repo has uncommitted changes. Modal will run the last committed state only."
        )
    print(f"Submitting commit {git_commit[:12]} from {git_remote}")
    if not init_submodules:
        print("Skipping git submodule init (init_submodules=false)")
    handle = run_hydra_train.spawn(
        tuple(hydra_args),
        git_remote,
        git_commit,
        _local_wandb_key(),
        init_submodules=init_submodules,
    )
    print(f"Submitted Modal job: {handle.object_id}")
    print("Monitor at: https://modal.com/apps/egomimic-training")
    print(
        "After completion, download artifacts:\n"
        "  modal volume get --env robotics egoverse-training-outputs <run-path> ./modal-outputs/"
    )


if __name__ == "__main__":
    """
    Usage:
        python egomimic/modal/trainModal.py \\
            data=mecka_all_zarr trainer=ddp_modal logger=wandb \\
            model=hpt_bc_flow_mecka name=my_run description=test

        python egomimic/modal/trainModal.py \\
            data=mecka_all_zarr trainer=ddp_modal logger=wandb \\
            model=hpt_bc_flow_mecka name=my_run description=test \\
            +modal_gpu=H100 +modal_cpu=32 +modal_memory_gb=128
    """
    _MODAL_KEY_MAP = {
        "modal_gpu": "MODAL_GPU",
        "modal_cpu": "MODAL_CPU",
        "modal_memory_gb": "MODAL_MEMORY_GB",
        "modal_memory_mb": "MODAL_MEMORY_MB",
        "modal_volume": "MODAL_VOLUME",
        "modal_ephemeral_disk_gb": "MODAL_EPHEMERAL_DISK_GB",
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

    container_overrides = [
        a
        for a in container_overrides
        if not a.lstrip("+").startswith("launch_params.gpus_per_node=")
    ]
    container_overrides.append(f"launch_params.gpus_per_node={gpu_count}")

    modal_env["MODAL_APP_NAME"] = app_name_from_hydra_args(container_overrides)

    gpu = modal_env.get("MODAL_GPU", "A100")
    cpu = modal_env.get("MODAL_CPU", "12")
    mem = modal_env.get("MODAL_MEMORY_GB") or str(
        int(modal_env.get("MODAL_MEMORY_MB", "65536")) // 1024
    )
    print(f"Modal app:       {modal_env['MODAL_APP_NAME']}")
    print(f"Modal resources: gpu={gpu}  cpu={cpu}  memory={mem}GB")

    launch_detached(Path(__file__).resolve(), "submit", container_overrides, modal_env)
