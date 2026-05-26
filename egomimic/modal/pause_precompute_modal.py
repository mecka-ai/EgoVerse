"""Modal pause-filter precompute fan-out for the tar-shard dataset.

The on-disk training data is a directory of tar shards
(``shard-<sha16>.tar``) on the ``mecka_data_wds_v2`` volume mounted at
``/mnt/zarr-wds``. Each shard bundles ~20 zarr episode directories. This
script fans containers out over shards (not individual episodes): every
worker streams one shard at a time onto local NVMe, opens the contained
zarr stores, and emits the per-episode keep-mask. The aggregated cache
is written to ``/mnt/zarr-wds/pause_cache/<run>_eps<epsilon>/cache.json``
and the trainer picks it up automatically via
``pause_precompute_cache:`` in its data config (or
``EGOMIMIC_PAUSE_PRECOMPUTE_CACHE``).

Pipeline
--------
run_pause_precompute (orchestrator, CPU — main training image)
  1. Hydra-compose the training config from the supplied overrides.
  2. Walk ``data.train_datasets`` + ``data.valid_datasets``, find every
     ``TarShardIterableDataset`` block, instantiate it (which globs the
     shard volume + honors mode/valid_ratio/debug), and union the
     resulting shard paths.
  3. Fan out ``_pause_precompute_tar_shard.map(...)`` across up to 500
     CPU containers — one shard per container.
  4. Aggregate the returned dicts and write
     ``/mnt/zarr-wds/pause_cache/<run>_eps<epsilon>/cache.json``.

_pause_precompute_tar_shard (CPU worker, up to 500 concurrent)
  • Extracts the tar shard onto local NVMe (~3 GB sequential read).
  • Opens each zarr inside; builds the keep-mask with the SAME algorithm
    as ``_build_pause_keep_mask`` in zarr_dataset_multi (kept in sync).
  • Returns ``{episode_hash: {"raw_total": int, "keep_indices": [int]}}``.
  • Cleans up the extracted directory so /tmp stays bounded.

Output JSON shape (matches ``_apply_pause_precompute_cache`` consumer)::

    {episode_hash: {"raw_total": int, "keep_indices": [int, ...]}}

Usage
-----
    python egomimic/modal/pause_precompute_modal.py \\
        name=pause_run description=mecka_10k \\
        pause_config_name=train_zarr_cartesian_pi \\
        pause_epsilon=0.005 \\
        data=mecka_10k_pause_filter

``pause_epsilon=<float>`` is required.

After completion the cache path is printed; wire it into your training
data config::

    pause_precompute_cache: /mnt/zarr-wds/pause_cache/<run>_eps<eps>/cache.json
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from modal_setup import (  # noqa: E402
    CFG,
    MODAL_COMPUTE_ARG_MAP,
    ModalCompute,
    _local_hf_token,
    _prepare_repo,
    _resolve_git_state,
    app,
    app_name_from_hydra_args,
    launch_detached,
    pop_init_submodules,
    training_outputs_volume,
    wds_volume,
    zarr_volume,
)

WDS_MOUNT_PATH = "/mnt/zarr-wds"

# Orchestrator is CPU-only (config walk + shard enumeration + result aggregation).
PAUSE_ORCHESTRATOR = ModalCompute(gpu=None, cpu=4.0, memory_mb=16384)

# Worker streams one tar shard (~3 GB) onto local NVMe and reads zarr stores
# locally. Bumped vs. the loose-zarr era because extraction needs RAM + tmpdir
# headroom for a full shard.
PAUSE_WORKER = ModalCompute(gpu=None, cpu=2.0, memory_mb=8192)

PAUSE_MAX_CONTAINERS = int(os.environ.get("EGOMIMIC_PAUSE_MAX_CONTAINERS", "500"))

_SHARED_SECRETS = [modal.Secret.from_name(name) for name in CFG.secret_names]


# ---------------------------------------------------------------------------
# Shard worker
# ---------------------------------------------------------------------------
#
# Inherits the main app image — the worker module deserializes
# ``pause_precompute_modal.py`` which imports ``modal_setup``; that file is
# only present in the main image. Tar extraction + numpy delta math; no heavy
# deps. The keep-mask logic must stay in sync with ``_build_pause_keep_mask``
# in ``egomimic.rldb.zarr.zarr_dataset_multi``.


@app.function(
    cpu=PAUSE_WORKER.cpu,
    memory=PAUSE_WORKER.memory_mb,
    timeout=1800,
    volumes={WDS_MOUNT_PATH: wds_volume},
    max_containers=PAUSE_MAX_CONTAINERS,
)
def _pause_precompute_tar_shard(
    shard_id: int,
    epsilon: float,
    tar_path: str,
) -> tuple[int, dict[str, dict]]:
    """Extract one tar shard, compute keep-indices per episode inside.

    Returns ``(shard_id, {episode_hash: entry})`` where
    ``entry = {"raw_total": int, "keep_indices": [int, ...]}``. Failures
    collapse to ``raw_total == 0`` so the consumer surfaces them as
    cache-miss errors at training time.
    """
    import shutil
    import tarfile
    import time as _time
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import zarr

    LEFT_EE = "left.obs_ee_pose"
    RIGHT_EE = "right.obs_ee_pose"
    LEFT_KP = "left.obs_keypoints"
    RIGHT_KP = "right.obs_keypoints"

    # Catch any shard writes that landed since the container last warmed.
    wds_volume.reload()

    def _keypoint_max_delta(kp: np.ndarray) -> np.ndarray:
        T = kp.shape[0]
        if T < 2 or kp.ndim != 2 or kp.shape[1] % 3 != 0 or kp.shape[1] == 0:
            return np.zeros(max(T - 1, 0))
        n_landmarks = kp.shape[1] // 3
        diff = np.diff(kp.reshape(T, n_landmarks, 3), axis=0)
        per_landmark_norm = np.linalg.norm(diff, axis=-1)
        return per_landmark_norm.max(axis=-1)

    def _build_keep_mask(left_pose, right_pose, left_kp, right_kp) -> np.ndarray:
        T = len(left_pose)
        if T < 2:
            return np.ones(T, dtype=bool)
        left_d = np.linalg.norm(np.diff(left_pose, axis=0), axis=-1)
        right_d = np.linalg.norm(np.diff(right_pose, axis=0), axis=-1)
        is_paused = (left_d < epsilon) & (right_d < epsilon)
        if left_kp is not None and len(left_kp) == T:
            is_paused = is_paused & (_keypoint_max_delta(np.asarray(left_kp)) < epsilon)
        if right_kp is not None and len(right_kp) == T:
            is_paused = is_paused & (
                _keypoint_max_delta(np.asarray(right_kp)) < epsilon
            )
        keep = np.ones(T, dtype=bool)
        in_pause = False
        for t in range(1, T):
            if is_paused[t - 1]:
                if in_pause:
                    keep[t] = False
                else:
                    in_pause = True
            else:
                in_pause = False
        return keep

    def _process_episode(ep_path: Path) -> tuple[str, int, list[int]]:
        episode_hash = ep_path.name
        if episode_hash.endswith(".zarr"):
            episode_hash = episode_hash[: -len(".zarr")]
        try:
            store = zarr.open_group(str(ep_path), mode="r")
        except Exception:
            return (episode_hash, 0, [])
        try:
            left = np.asarray(store[LEFT_EE][:])
            right = np.asarray(store[RIGHT_EE][:])
        except KeyError:
            try:
                sample = next(iter(store.array_keys()), None)
                total = int(store[sample].shape[0]) if sample else 0
            except Exception:
                total = 0
            return (episode_hash, total, list(range(total)))
        except Exception:
            return (episode_hash, 0, [])
        left_kp = right_kp = None
        try:
            left_kp = np.asarray(store[LEFT_KP][:])
        except Exception:
            pass
        try:
            right_kp = np.asarray(store[RIGHT_KP][:])
        except Exception:
            pass
        try:
            keep = _build_keep_mask(left, right, left_kp, right_kp)
        except Exception:
            return (episode_hash, 0, [])
        indices = np.flatnonzero(keep).astype(np.int64).tolist()
        return (episode_hash, int(left.shape[0]), indices)

    tar_p = Path(tar_path)
    scratch = Path("/tmp") / f"shard_{shard_id}_{tar_p.stem}"
    if scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)

    t0 = _time.monotonic()
    out: dict[str, dict] = {}
    try:
        try:
            with tarfile.open(str(tar_p), "r") as tar:
                tar.extractall(path=str(scratch))
        except Exception as e:
            print(
                f"[pause-shard {shard_id}] tar extract FAILED for {tar_p.name}: {e!r}",
                file=sys.stderr,
            )
            return shard_id, out

        ep_dirs = [p for p in scratch.iterdir() if p.is_dir()]
        if not ep_dirs:
            print(
                f"[pause-shard {shard_id}] empty shard {tar_p.name} — 0 episodes",
                file=sys.stderr,
            )
            return shard_id, out

        with ThreadPoolExecutor(max_workers=8) as ex:
            for episode_hash, raw_total, indices in ex.map(_process_episode, ep_dirs):
                out[episode_hash] = {"raw_total": raw_total, "keep_indices": indices}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    n_kept = sum(len(v["keep_indices"]) for v in out.values())
    n_total = sum(v["raw_total"] for v in out.values())
    n_err = sum(1 for v in out.values() if v["raw_total"] == 0)
    pct = (100.0 * n_kept / n_total) if n_total else 100.0
    print(
        f"[pause-shard {shard_id}] {tar_p.name} | {len(out)} eps, eps={epsilon} | "
        f"kept {n_kept}/{n_total} ({pct:.1f}%) | errors={n_err} | "
        f"{_time.monotonic() - t0:.1f}s"
    )
    return shard_id, out


# ---------------------------------------------------------------------------
# Orchestrator (main training image — needs hydra + egomimic for discovery)
# ---------------------------------------------------------------------------


@app.function(
    cpu=PAUSE_ORCHESTRATOR.cpu,
    memory=PAUSE_ORCHESTRATOR.memory_mb,
    timeout=CFG.timeout_seconds,
    secrets=_SHARED_SECRETS,
    volumes={
        CFG.volume_mount_path: zarr_volume,
        CFG.output_mount_path: training_outputs_volume,
        WDS_MOUNT_PATH: wds_volume,
    },
)
def run_pause_precompute(
    hydra_args: tuple[str, ...],
    git_remote: str,
    git_commit: str,
    config_name: str,
    out_subdir: str,
    epsilon: float,
    init_submodules: bool = True,
    hf_token: str = "",
) -> str:
    """Orchestrator: hydra-compose → enumerate tar shards → fan-out → aggregate."""
    import json
    import sys as _sys
    import time as _time

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    _prepare_repo(
        git_remote=git_remote, git_commit=git_commit, init_submodules=init_submodules
    )
    _sys.path.insert(0, CFG.remote_repo_dir)
    os.chdir(CFG.remote_repo_dir)
    os.environ["MODAL_IS_REMOTE"] = "1"
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from egomimic.utils.aws.aws_data_utils import load_env

    load_env()

    # Make sure the orchestrator sees any shard writes that landed since the
    # container last warmed (otherwise dataset.__init__'s glob can come up
    # empty on a freshly-populated volume).
    wds_volume.reload()

    with initialize_config_dir(
        config_dir=f"{CFG.remote_repo_dir}/egomimic/hydra_configs",
        version_base=None,
    ):
        cfg = compose(config_name=config_name, overrides=list(hydra_args))

    # ── Walk dataset blocks, collect tar-shard paths ─────────────────────────
    # Instantiating ``TarShardIterableDataset`` runs its shard-discovery glob
    # against ``shard_dir`` and applies the same train/valid split + debug
    # truncation as training. We union the resulting ``_shards`` across blocks
    # so the cache covers every shard the trainer will read.
    tar_shards: dict[str, str] = {}
    seen_blocks = 0
    data_cfg = cfg.get("data") if cfg is not None else None
    if data_cfg is None:
        raise RuntimeError(
            f"pause-precompute: composed config '{config_name}' has no `data` group"
        )

    for block_name in ("train_datasets", "valid_datasets"):
        block = data_cfg.get(block_name)
        if block is None:
            continue
        for ds_name, ds_cfg in block.items():
            target = ds_cfg.get("_target_") if ds_cfg is not None else None
            if target is None or "TarShardIterableDataset" not in str(target):
                continue
            seen_blocks += 1
            try:
                dataset = instantiate(ds_cfg, _recursive_=True)
            except Exception as e:
                print(
                    f"[pause-precompute] {block_name}.{ds_name}: dataset "
                    f"instantiation failed: {e}",
                    file=sys.stderr,
                )
                continue

            shards = getattr(dataset, "_shards", None)
            if not shards:
                print(
                    f"[pause-precompute] {block_name}.{ds_name}: dataset has "
                    "no `_shards` — skipping",
                    file=sys.stderr,
                )
                continue

            for shard_path in shards:
                p = str(shard_path)
                tar_shards[Path(p).name] = p
            print(
                f"[pause-precompute] {block_name}.{ds_name}: "
                f"{len(shards)} shard(s) resolved"
            )

    if not tar_shards:
        raise RuntimeError(
            f"pause-precompute: no TarShardIterableDataset blocks found across "
            f"{seen_blocks} candidate block(s) in '{config_name}'."
        )

    total_shards = len(tar_shards)
    print(
        f"[pause-precompute] {total_shards} unique shard(s) "
        f"(epsilon={epsilon}, max_containers={PAUSE_MAX_CONTAINERS})"
    )

    # ── Fan out: one container per shard ─────────────────────────────────────
    shard_args: list[tuple[int, float, str]] = [
        (i, float(epsilon), path)
        for i, path in enumerate(sorted(tar_shards.values()))
    ]

    cache: dict[str, dict] = {}
    completed = 0
    n_failures = 0
    log_every = max(1, total_shards // 20)
    t0 = _time.time()
    for result in _pause_precompute_tar_shard.starmap(
        shard_args, return_exceptions=True
    ):
        completed += 1
        if isinstance(result, Exception):
            n_failures += 1
            print(f"[pause-precompute] shard FAILED: {result!r}", file=sys.stderr)
        else:
            _, shard_dict = result
            cache.update(shard_dict)
        if completed % log_every == 0 or completed == total_shards:
            elapsed = _time.time() - t0
            print(
                f"[pause-precompute] {completed}/{total_shards} shards done | "
                f"failures={n_failures} | episodes={len(cache)} | "
                f"elapsed {elapsed:.0f}s"
            )

    elapsed = _time.time() - t0
    n_kept = sum(len(v["keep_indices"]) for v in cache.values())
    n_total = sum(v["raw_total"] for v in cache.values())
    n_miss = sum(1 for v in cache.values() if v["raw_total"] == 0)
    pct = (100.0 * n_kept / n_total) if n_total else 100.0
    print(
        f"[pause-precompute] complete — {len(cache)} episodes | "
        f"kept {n_kept}/{n_total} ({pct:.1f}%) | misses={n_miss} | "
        f"{elapsed:.1f}s"
    )

    # ── Write cache.json next to the shards on the wds volume ───────────────
    out_dir = Path(WDS_MOUNT_PATH) / "pause_cache" / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "cache.json"
    tmp_path = cache_path.with_suffix(".json.tmp")
    with tmp_path.open("w") as f:
        json.dump(cache, f)
    tmp_path.replace(cache_path)
    wds_volume.commit()

    print(f"\n=== DONE ===\npause_precompute_cache: {cache_path}\n")
    return str(cache_path)


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def submit_pause_precompute(*hydra_args: str) -> None:
    """Fire-and-forget: spawn a pause-precompute job from an already-pushed commit.

    Recognized non-hydra args (stripped before composition):
      - ``pause_config_name=<name>`` — Hydra training config (required, e.g.
        ``train_zarr_cartesian`` or ``train_zarr_cartesian_pi``).
      - ``pause_epsilon=<float>`` — pause-classification epsilon (required).
        Appended to the output subdir as ``_eps<value>``.
      - ``init_submodules=<bool>`` — clone with --recurse-submodules.
      - ``name=<str>`` / ``description=<str>`` — used to label the output subdir.
    """
    config_name: str | None = None
    epsilon_str: str | None = None
    name = description = ""
    args: list[str] = []
    for arg in hydra_args:
        bare = arg.lstrip("+")
        k, sep, v = bare.partition("=")
        if sep and k == "pause_config_name":
            config_name = v
        elif sep and k == "pause_epsilon":
            epsilon_str = v
        else:
            if sep and k == "name":
                name = v
            elif sep and k == "description":
                description = v
            args.append(arg)
    args, init_submodules = pop_init_submodules(args)

    if not config_name:
        raise SystemExit(
            "pause-precompute: pause_config_name=<hydra-config-name> is required "
            "(e.g. pause_config_name=train_zarr_cartesian)"
        )
    if epsilon_str is None:
        raise SystemExit(
            "pause-precompute: pause_epsilon=<float> is required "
            "(e.g. pause_epsilon=0.005)"
        )
    try:
        epsilon = float(epsilon_str)
    except ValueError:
        raise SystemExit(
            f"pause-precompute: pause_epsilon must be a float, got {epsilon_str!r}"
        )

    import time as _time
    timestamp = _time.strftime("%Y-%m-%d_%H-%M-%S")
    out_subdir = (
        f"{name or 'pause'}_{description or config_name}_eps{epsilon_str}_{timestamp}"
    )

    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print(
            "Warning: local repo has uncommitted changes. "
            "Modal will run the last committed state only."
        )
    print(
        f"Submitting pause-precompute at commit {git_commit[:12]} from {git_remote}\n"
        f"  config_name={config_name}  epsilon={epsilon}\n"
        f"  out_subdir={out_subdir}"
    )

    handle = run_pause_precompute.spawn(
        tuple(args),
        git_remote,
        git_commit,
        config_name,
        out_subdir,
        epsilon,
        init_submodules=init_submodules,
        hf_token=_local_hf_token(),
    )
    _env = os.environ.get("MODAL_ENVIRONMENT", "robotics")
    _app = os.environ.get("MODAL_APP_NAME", "egomimic-training")
    print(f"Submitted Modal pause-precompute job: {handle.object_id}")
    print(f"Monitor: https://modal.com/apps/mecka/{_env}/apps/{_app}")


# ---------------------------------------------------------------------------
# python egomimic/modal/pause_precompute_modal.py \
#     name=… description=… pause_config_name=train_zarr_cartesian_pi \
#     pause_epsilon=0.005 \
#     data=mecka_10k_pause_filter
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    modal_env = os.environ.copy()
    hydra_args: list[str] = []
    for arg in sys.argv[1:]:
        key, sep, val = arg.lstrip("+").partition("=")
        if sep and key in MODAL_COMPUTE_ARG_MAP:
            modal_env[MODAL_COMPUTE_ARG_MAP[key]] = val
        else:
            hydra_args.append(arg)

    modal_env["MODAL_APP_NAME"] = app_name_from_hydra_args(hydra_args)
    print(f"Modal app:                                    {modal_env['MODAL_APP_NAME']}")
    print(f"Modal pause-precompute orchestrator:          {PAUSE_ORCHESTRATOR.summary()}")
    print(
        f"Modal pause-precompute shard worker:          {PAUSE_WORKER.summary()} "
        f"(max_containers={PAUSE_MAX_CONTAINERS})"
    )

    launch_detached(
        Path(__file__).resolve(), "submit_pause_precompute", hydra_args, modal_env
    )
