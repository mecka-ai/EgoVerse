"""Modal pause-filter precompute fan-out.

Mirrors egomimic/scripts/nebius/pause_precompute_driver.py but dispatches
Modal containers instead of an sbatch array. Designed to run once before
training so the trainer can consume the resulting cache.json via
``pause_precompute_cache:`` in its data config (or
``EGOMIMIC_PAUSE_PRECOMPUTE_CACHE``).

Pipeline
--------
run_pause_precompute (orchestrator, CPU — main training image)
  1. Hydra-compose the training config from the supplied overrides.
  2. Walk ``data.train_datasets`` + ``data.valid_datasets``, picking out
     every resolver with ``pause_removal_epsilon`` set.
  3. Call ``discover_episode_paths(filters)`` on each to enumerate
     ``[(episode_hash, local_path)]`` under ``/mnt/zarr-data`` — no
     ZarrDataset construction, no in-process precompute.
  4. Group by epsilon, partition round-robin into N shards (default 500,
     overridable via ``pause_shards=N``).
  5. Fan out ``_pause_precompute_shard.map(...)`` across up to 500 CPU
     containers.
  6. Aggregate the returned dicts and write
     ``/mnt/zarr-data/pause_cache/<run>/cache.json`` on the zarr volume.

_pause_precompute_shard (CPU worker, slim image, up to 500 concurrent)
  • Opens each zarr; builds the keep-mask with the SAME algorithm as
    ``_build_pause_keep_mask`` in zarr_dataset_multi (kept in sync; the
    image is intentionally slim so cold starts are ~5s).
  • Returns ``{episode_hash: {"raw_total": int, "keep_indices": [int]}}``.

Output JSON shape (matches ``_apply_pause_precompute_cache`` consumer)::

    {episode_hash: {"raw_total": int, "keep_indices": [int, ...]}}

Usage
-----
    python egomimic/modal/pause_precompute_modal.py \\
        name=pause_run description=mecka_50k \\
        pause_config_name=train_zarr_cartesian \\
        data=mecka_50k_20k

Optional: pass ``pause_shards=N`` (default 500) on its own — do NOT wrap
in brackets; bash forwards them as literal characters and Hydra rejects
them as an unparseable override.

After completion the cache path is printed; wire it into your training
data config::

    pause_precompute_cache: /mnt/zarr-data/pause_cache/<run>/cache.json
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
    zarr_volume,
)

# Orchestrator is CPU-only (SQL + path enumeration + result aggregation).
PAUSE_ORCHESTRATOR = ModalCompute(gpu=None, cpu=4.0, memory_mb=16384)

# Worker is mostly I/O bound — small zarr reads + numpy delta math.
PAUSE_WORKER = ModalCompute(gpu=None, cpu=2.0, memory_mb=4096)

PAUSE_MAX_CONTAINERS = int(os.environ.get("EGOMIMIC_PAUSE_MAX_CONTAINERS", "500"))
DEFAULT_SHARDS = int(os.environ.get("EGOMIMIC_PAUSE_PRECOMPUTE_SHARDS", "500"))

_SHARED_SECRETS = [modal.Secret.from_name(name) for name in CFG.secret_names]

# Slim image for the shard workers — no repo clone, fast cold start. The
# keep-mask algorithm below is duplicated from
# egomimic.rldb.zarr.zarr_dataset_multi._build_pause_keep_mask; keep them
# in sync.  The Nebius worker, by contrast, imports from zarr_dataset_multi
# directly because Slurm has cheap "full egomimic" via the cluster venv —
# Modal would require a repo clone per container, which kills cold-start
# at 500-way fan-out.
pause_worker_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "zarr==3.1.5", "numpy"
)


# ---------------------------------------------------------------------------
# Shard worker (slim image)
# ---------------------------------------------------------------------------


@app.function(
    image=pause_worker_image,
    cpu=PAUSE_WORKER.cpu,
    memory=PAUSE_WORKER.memory_mb,
    timeout=1800,
    volumes={CFG.volume_mount_path: zarr_volume},
    max_containers=PAUSE_MAX_CONTAINERS,
)
def _pause_precompute_shard(
    shard_id: int,
    epsilon: float,
    episodes: list[tuple[str, str]],
) -> tuple[int, dict[str, dict]]:
    """Compute keep-indices for one shard. Returns ``(shard_id, {hash: entry})``.

    ``entry = {"raw_total": int, "keep_indices": [int, ...]}``. Failures
    collapse to ``raw_total == 0`` so the consumer surfaces them as
    cache-miss errors at training time.
    """
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import zarr

    LEFT_EE = "left.obs_ee_pose"
    RIGHT_EE = "right.obs_ee_pose"
    LEFT_KP = "left.obs_keypoints"
    RIGHT_KP = "right.obs_keypoints"

    # Make sure we see the latest volume state (writers may have run since
    # the worker container was last warm).
    zarr_volume.reload()

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

    def _one(item: tuple[str, str]) -> tuple[str, int, list[int]]:
        episode_hash, path_str = item
        try:
            store = zarr.open_group(path_str, mode="r")
        except Exception:
            return (episode_hash, 0, [])
        try:
            left = np.asarray(store[LEFT_EE][:])
            right = np.asarray(store[RIGHT_EE][:])
        except KeyError:
            # Episode lacks ee_pose — keep all frames (matches the in-process
            # precompute_pause_filter fallback in zarr_dataset_multi).
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

    import time as _time

    t0 = _time.monotonic()
    out: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        for episode_hash, raw_total, indices in ex.map(_one, episodes):
            out[episode_hash] = {"raw_total": raw_total, "keep_indices": indices}

    n_kept = sum(len(v["keep_indices"]) for v in out.values())
    n_total = sum(v["raw_total"] for v in out.values())
    n_err = sum(1 for v in out.values() if v["raw_total"] == 0)
    pct = (100.0 * n_kept / n_total) if n_total else 100.0
    print(
        f"[pause-shard {shard_id}] {len(episodes)} eps, eps={epsilon} | "
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
    },
)
def run_pause_precompute(
    hydra_args: tuple[str, ...],
    git_remote: str,
    git_commit: str,
    config_name: str,
    out_subdir: str,
    n_shards: int = DEFAULT_SHARDS,
    init_submodules: bool = True,
    hf_token: str = "",
) -> str:
    """Orchestrator: hydra-compose → discover → fan-out → aggregate."""
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

    with initialize_config_dir(
        config_dir=f"{CFG.remote_repo_dir}/egomimic/hydra_configs",
        version_base=None,
    ):
        cfg = compose(config_name=config_name, overrides=list(hydra_args))

    # ── Walk resolvers and enumerate (hash, path) by epsilon ──────────────────
    work_by_eps: dict[float, dict[str, str]] = {}
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
            resolver_cfg = ds_cfg.get("resolver") if ds_cfg is not None else None
            if resolver_cfg is None:
                continue
            seen_blocks += 1
            try:
                resolver = instantiate(resolver_cfg)
            except Exception as e:
                print(
                    f"[pause-precompute] {block_name}.{ds_name}: resolver "
                    f"instantiation failed: {e}",
                    file=sys.stderr,
                )
                continue
            if not hasattr(resolver, "discover_episode_paths"):
                continue
            eps = getattr(resolver, "pause_removal_epsilon", None)
            if eps is None:
                continue

            filters_cfg = ds_cfg.get("filters")
            filters = instantiate(filters_cfg) if filters_cfg is not None else None

            try:
                pairs = resolver.discover_episode_paths(filters)
            except Exception as e:
                print(
                    f"[pause-precompute] {block_name}.{ds_name}: "
                    f"discover_episode_paths failed: {e}",
                    file=sys.stderr,
                )
                continue

            bucket = work_by_eps.setdefault(float(eps), {})
            for episode_hash, local_path in pairs:
                bucket[episode_hash] = local_path
            print(
                f"[pause-precompute] {block_name}.{ds_name}: "
                f"epsilon={eps}, {len(pairs)} episodes resolved"
            )

    if not work_by_eps:
        raise RuntimeError(
            f"pause-precompute: no resolvers with pause_removal_epsilon found "
            f"across {seen_blocks} dataset block(s). Set pause_removal_epsilon "
            "on at least one resolver in your data config and re-submit."
        )

    total_eps = sum(len(v) for v in work_by_eps.values())
    print(
        f"[pause-precompute] {total_eps} episodes across {len(work_by_eps)} "
        "epsilon group(s): "
        + ", ".join(f"eps={e}({len(v)})" for e, v in work_by_eps.items())
    )

    # ── Partition: round-robin per epsilon → list[(shard_id, eps, [(h, p)])] ──
    # Round-robin (episodes[i::eps_shards]) spreads similar storage layouts
    # across shards — keeps tail latency tight. Mirrors the Nebius driver's
    # _write_manifest algorithm.
    shards: list[tuple[int, float, list[tuple[str, str]]]] = []
    for eps, hash_to_path in work_by_eps.items():
        episodes = list(hash_to_path.items())
        if not episodes:
            continue
        eps_shards = min(n_shards, len(episodes))
        for i in range(eps_shards):
            shard_eps = episodes[i::eps_shards]
            if shard_eps:
                shards.append((len(shards), float(eps), shard_eps))

    total_shards = len(shards)
    print(
        f"[pause-precompute] fanning out {total_shards} shard(s) "
        f"(max_containers={PAUSE_MAX_CONTAINERS}) …"
    )
    t0 = _time.time()

    # ── Map across shards ─────────────────────────────────────────────────────
    cache: dict[str, dict] = {}
    completed = 0
    n_failures = 0
    log_every = max(1, total_shards // 20)
    for result in _pause_precompute_shard.starmap(shards, return_exceptions=True):
        completed += 1
        if isinstance(result, Exception):
            n_failures += 1
            # Modal doesn't surface the failed shard's id with return_exceptions,
            # so just log the exception. Affected episodes will trigger a
            # cache-miss error at training time, which names them explicitly.
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

    # ── Write cache.json to the zarr volume so training can read it back ──────
    out_dir = Path(CFG.volume_mount_path) / "pause_cache" / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "cache.json"
    tmp_path = cache_path.with_suffix(".json.tmp")
    with tmp_path.open("w") as f:
        json.dump(cache, f)
    tmp_path.replace(cache_path)
    zarr_volume.commit()

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
      - ``pause_shards=<int>`` — number of shards (default 500).
      - ``init_submodules=<bool>`` — clone with --recurse-submodules.
      - ``name=<str>`` / ``description=<str>`` — used to label the output subdir.
    """
    # Strip pause_config_name / pause_shards (used here) and collect name/
    # description (for the output subdir). Everything else stays in `args`
    # to forward to Hydra compose inside the orchestrator.
    config_name: str | None = None
    shards_str: str = str(DEFAULT_SHARDS)
    name = description = ""
    args: list[str] = []
    for arg in hydra_args:
        bare = arg.lstrip("+")
        k, sep, v = bare.partition("=")
        if sep and k == "pause_config_name":
            config_name = v
        elif sep and k == "pause_shards":
            shards_str = v
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
    n_shards = int(shards_str)

    import time as _time
    timestamp = _time.strftime("%Y-%m-%d_%H-%M-%S")
    out_subdir = f"{name or 'pause'}_{description or config_name}_{timestamp}"

    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print(
            "Warning: local repo has uncommitted changes. "
            "Modal will run the last committed state only."
        )
    print(
        f"Submitting pause-precompute at commit {git_commit[:12]} from {git_remote}\n"
        f"  config_name={config_name}  shards={n_shards}  out_subdir={out_subdir}"
    )

    handle = run_pause_precompute.spawn(
        tuple(args),
        git_remote,
        git_commit,
        config_name,
        out_subdir,
        n_shards=n_shards,
        init_submodules=init_submodules,
        hf_token=_local_hf_token(),
    )
    _env = os.environ.get("MODAL_ENVIRONMENT", "robotics")
    _app = os.environ.get("MODAL_APP_NAME", "egomimic-training")
    print(f"Submitted Modal pause-precompute job: {handle.object_id}")
    print(f"Monitor: https://modal.com/apps/mecka/{_env}/apps/{_app}")


# ---------------------------------------------------------------------------
# python egomimic/modal/pause_precompute_modal.py \
#     name=… description=… pause_config_name=train_zarr_cartesian \
#     data=mecka_50k_20k   (optional: pause_shards=N, default 500)
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
