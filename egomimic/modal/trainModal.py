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
    REPO_ROOT,
    VOLUME_MAP,
    _local_wandb_key,
    _prepare_repo,
    _resolve_git_state,
    app_name_from_hydra_args,
    launch_detached,
    pop_init_submodules,
    app,
    pause_precompute_shard,
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
# Pre-train pause precompute
# ---------------------------------------------------------------------------
#
# run_hydra_train spawns trainHydra in a subprocess; that subprocess can't
# .map() a Modal function (no hydration crosses the process boundary).
# So we materialize the hydra config here, resolve which episodes the
# resolvers will request, fan out pause_precompute_shard.map() from this
# function (which IS hydrated), and write the keep_indices to a cache JSON
# on disk. trainHydra's resolver reads the cache via
# EGOMIMIC_PAUSE_PRECOMPUTE_CACHE env var.

PAUSE_PRECOMPUTE_CACHE_PATH = "/tmp/pause_precompute_cache.json"


def _precompute_pause_to_cache(hydra_args: tuple[str, ...]) -> str | None:
    """Fan out pause precompute for every ModalEpisodeResolver in the data config.

    Returns the cache-file path if anything was precomputed, else None.
    Idempotent and best-effort — any failure logs and returns None so the
    in-process fallback inside the trainHydra resolver still runs.
    """
    import json as _json_local
    import time as _time_local

    try:
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
    except Exception as e:
        print(f"[pause-precompute] hydra not available: {e}; skipping fan-out")
        return None

    config_dir = str(Path(CFG.remote_repo_dir) / "egomimic" / "hydra_configs")
    if not Path(config_dir).is_dir():
        print(f"[pause-precompute] config dir missing ({config_dir}); skipping")
        return None

    try:
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="train_zarr_cartesian",
                overrides=list(hydra_args),
            )
    except Exception as e:
        print(f"[pause-precompute] hydra compose failed: {e}; skipping fan-out")
        return None

    # Walk train_datasets + valid_datasets, build (resolver, filters) pairs
    from egomimic.rldb.zarr.zarr_dataset_multi import ModalEpisodeResolver

    work_by_eps: dict[float, dict[str, str]] = {}
    seen_blocks = 0
    for block_name in ("train_datasets", "valid_datasets"):
        block = cfg.data.get(block_name)
        if block is None:
            continue
        for ds_name, ds_cfg in block.items():
            resolver_cfg = ds_cfg.get("resolver")
            if resolver_cfg is None:
                continue
            seen_blocks += 1
            try:
                resolver = instantiate(resolver_cfg)
            except Exception as e:
                print(
                    f"[pause-precompute] failed to instantiate resolver "
                    f"for {block_name}.{ds_name}: {e}"
                )
                continue
            if not isinstance(resolver, ModalEpisodeResolver):
                continue
            eps = resolver.pause_removal_epsilon
            if eps is None:
                continue
            filters_cfg = ds_cfg.get("filters")
            filters = instantiate(filters_cfg) if filters_cfg is not None else None
            try:
                meta = resolver._resolve_episode_meta(filters)
            except Exception as e:
                print(
                    f"[pause-precompute] _resolve_episode_meta failed for "
                    f"{block_name}.{ds_name}: {e}"
                )
                continue
            bucket = work_by_eps.setdefault(float(eps), {})
            for episode_hash, local_path, _num_frames, _robot in meta:
                # Dedup across train/valid: same hash + same eps = same work.
                bucket[episode_hash] = str(local_path)

    if not work_by_eps:
        print(
            f"[pause-precompute] no ModalEpisodeResolver with "
            f"pause_removal_epsilon found across {seen_blocks} dataset block(s); "
            "skipping fan-out (trainHydra will use in-process precompute if needed)"
        )
        return None

    t0 = _time_local.monotonic()
    cache: dict[str, dict] = {}
    for eps, hash_to_path in work_by_eps.items():
        episodes = list(hash_to_path.items())
        n = len(episodes)
        n_shards = min(
            int(os.environ.get("EGOMIMIC_PAUSE_PRECOMPUTE_SHARDS", "100")), n
        )
        shards = [episodes[i::n_shards] for i in range(n_shards)]
        shards = [s for s in shards if s]
        epsilons = [eps] * len(shards)
        print(
            f"[pause-precompute] eps={eps}: {n} episodes across {len(shards)} shards "
            f"(same-app worker, no deploy)"
        )
        completed = 0
        for shard_result in pause_precompute_shard.map(shards, epsilons):
            completed += 1
            for episode_hash, raw_total, indices in shard_result:
                cache[episode_hash] = {
                    "raw_total": int(raw_total),
                    "keep_indices": list(indices),
                }
            if completed % max(1, len(shards) // 10) == 0 or completed == len(shards):
                elapsed = _time_local.monotonic() - t0
                print(
                    f"[pause-precompute] eps={eps}: shard {completed}/{len(shards)} "
                    f"done | episodes={len(cache)} | elapsed {elapsed:.0f}s"
                )

    elapsed = _time_local.monotonic() - t0
    n_kept = sum(len(v["keep_indices"]) for v in cache.values())
    n_total = sum(v["raw_total"] for v in cache.values())
    pct = (100.0 * n_kept / n_total) if n_total else 100.0
    print(
        f"[pause-precompute] complete: {len(cache)} episodes "
        f"kept {n_kept}/{n_total} ({pct:.1f}%) in {elapsed:.1f}s"
    )

    Path(PAUSE_PRECOMPUTE_CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(PAUSE_PRECOMPUTE_CACHE_PATH, "w") as f:
        _json_local.dump(cache, f)
    return PAUSE_PRECOMPUTE_CACHE_PATH


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

    hydra_args = _resolve_volume_paths(hydra_args)

    # Fan out pause precompute now while we're inside a hydrated function;
    # the trainHydra subprocess can't reach a Modal worker without a deploy.
    cache_path = _precompute_pause_to_cache(hydra_args)

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
    if cache_path:
        env["EGOMIMIC_PAUSE_PRECOMPUTE_CACHE"] = cache_path

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
