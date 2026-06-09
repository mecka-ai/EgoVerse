"""Modal entrypoints for cross-episode k-NN action-consistency grading.

Architecture
------------
run_knn_grade (orchestrator, CPU)
  • SQL task lookup partitions episode hashes by task.
  • Fans out one _grade_task_split container per task.

_grade_task_split (CPU worker, one per task)
  • Reloads the outputs volume and diffs against the feature cache.
  • Spawns _featurize_task_shard GPU containers for *missing* episodes only.
  • Loads cached features, runs grade_task (pure numpy — see
    egomimic/curation/knn_grading.py), writes tasks/<task>.json.
  • Optionally spawns _render_retrieval_grids for visual spot-checks of the
    worst-disagreement states (calibrate retrieval before trusting scores!).

_featurize_task_shard (GPU worker, one per shard of ≤ max_episodes_per_shard)
  • collect_grading_episode (frame stride) → pooled DINOv3 features →
    one npz per episode on the egoverse-training-outputs volume.

Feature caches live at
    knn_grading/features/<feature_version>/<task>/<hash>.npz
and are keyed by ``feature_version`` (not run name), so metric iteration
(k, weights, thresholds…) re-runs CPU-only. Run outputs land in
    knn_grading/<name>/<description>_<timestamp>/
        knn_scores_by_task.json   {task: {hash: primary_score}} worst-first
        knn_report.json           per-episode metrics + task summaries
        tasks/<task>.json         full per-task result incl. debug states
        review/<task>.csv         ranked sheet with empty label column
        grids/<task>/*.jpg        query + neighbor frames for spot-checks

Usage
-----
    python egomimic/modal/knnGradeModal.py name=my_run description=test

    python egomimic/modal/knnGradeModal.py name=my_run description=test \\
        +modal_gpu=L40S:1 +modal_cpu=32 +modal_memory_gb=128 \\
        knn.k=20 featurize.stride=6

    modal run --env robotics egomimic/modal/knnGradeModal.py::submit_knn_grade -- \\
        name=my_run description=test
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from modal_setup import (  # noqa: E402
    CFG,
    CURATE_ORCHESTRATOR,
    MODAL_COMPUTE_ARG_MAP,
    ModalCompute,
    _local_hf_token,
    _prepare_repo,
    _prepare_repo_light,
    _resolve_git_state,
    app,
    app_name_from_hydra_args,
    launch_detached,
    pop_init_submodules,
    training_outputs_volume,
    zarr_volume,
)

# GPU featurize-shard workers — override at launch via +modal_gpu / +modal_cpu /
# +modal_memory_gb.
TASK_COMPUTE = ModalCompute.from_environ(
    default_gpu="L40S",
    default_cpu=16.0,
    default_memory_mb=131072,
)

# Per-task CPU scorer (cache diff + k-NN grading) — fixed, no GPU.
GRADE_SCORE_COMPUTE = ModalCompute(gpu=None, cpu=16.0, memory_mb=65536)

# Grid renderer — light CPU, needs the zarr volume for frame decode.
GRID_RENDER_COMPUTE = ModalCompute(gpu=None, cpu=8.0, memory_mb=32768)

_SHARED_SECRETS = [modal.Secret.from_name(name) for name in CFG.secret_names]

FEATURES_SUBDIR = "knn_grading/features"


# ---------------------------------------------------------------------------
# Shared container boot helpers (entrypoint-local: the repo is cloned only
# inside _boot_container, so these cannot live in egomimic.*)
# ---------------------------------------------------------------------------


def _boot_container(git_remote: str, git_commit: str, hf_token: str) -> None:
    """Set up the remote container: env vars, repo clone, sys.path."""
    import sys as _sys

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    _prepare_repo_light(git_remote=git_remote, git_commit=git_commit)
    _sys.path.insert(0, CFG.remote_repo_dir)
    os.chdir(CFG.remote_repo_dir)
    os.environ["MODAL_IS_REMOTE"] = "1"
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")


def _load_cfg(hydra_args: tuple[str, ...]):
    """Compose the knn_grade Hydra config inside an already-booted container."""
    import hydra as _hydra

    with _hydra.initialize_config_dir(
        config_dir=f"{CFG.remote_repo_dir}/egomimic/hydra_configs",
        version_base="1.3",
    ):
        return _hydra.compose("knn_grade", overrides=list(hydra_args))


def _inject_episode_ids(cfg, hashes: list[str]):
    """Deep-copy cfg and inject allowed_episode_ids into every resolver node."""
    import copy

    from omegaconf import open_dict

    task_cfg = copy.deepcopy(cfg)
    with open_dict(task_cfg):
        for ds_name in task_cfg.data.train_datasets:
            ds_node = task_cfg.data.train_datasets[ds_name]
            if "resolver" in ds_node:
                ds_node.resolver.allowed_episode_ids = list(hashes)
    return task_cfg


def _build_episode_map(task_cfg, hydra) -> dict:
    """Instantiate MultiDatasets from task_cfg; return {hash: ZarrDataset}."""
    episodes: dict = {}
    for ds_name in task_cfg.data.train_datasets:
        ds_node = task_cfg.data.train_datasets[ds_name]
        try:
            md = hydra.utils.instantiate(ds_node)
            episodes.update(md.datasets)
            print(f"  [{ds_name}] {len(md.datasets)} episodes loaded")
        except Exception as exc:
            print(f"  [{ds_name}] dataset build FAILED: {exc}")
    return episodes


def _features_task_dir(feature_version: str, task_name: str) -> Path:
    return Path(CFG.output_mount_path) / FEATURES_SUBDIR / feature_version / task_name


# ---------------------------------------------------------------------------
# GPU featurize shard (one per shard of ≤ max_episodes_per_shard episodes)
# ---------------------------------------------------------------------------


@app.function(
    gpu=TASK_COMPUTE.gpu,
    cpu=TASK_COMPUTE.cpu,
    memory=TASK_COMPUTE.memory_mb,
    timeout=86400,
    secrets=_SHARED_SECRETS,
    volumes={
        CFG.volume_mount_path: zarr_volume,
        CFG.output_mount_path: training_outputs_volume,
    },
)
def _featurize_task_shard(
    task_name: str,
    shard_idx: int,
    shard_hashes: list[str],
    git_remote: str,
    git_commit: str,
    hydra_args: tuple[str, ...],
    hf_token: str = "",
) -> tuple[int, list[dict]]:
    """GPU worker: featurize one shard of episodes into the npz cache."""
    import time as _time

    import torch

    t0 = _time.perf_counter()
    _boot_container(git_remote, git_commit, hf_token)
    import hydra as _hydra

    from egomimic.utils.aws.aws_data_utils import load_env

    load_env()
    cfg = _load_cfg(hydra_args)
    tag = f"[{task_name}][shard {shard_idx}]"
    print(f"{tag} {len(shard_hashes)} episodes — featurizing")

    task_cfg = _inject_episode_ids(cfg, shard_hashes)
    all_episodes = _build_episode_map(task_cfg, _hydra)
    if not all_episodes:
        print(f"{tag} No episodes loaded — returning empty shard")
        return shard_idx, []

    from egomimic.curation.config import (
        apply_curation_seed,
        select_curation_loader,
        select_seed,
    )
    from egomimic.curation.grading_pipeline import (
        GradingFeaturizeSettings,
        build_grading_featurizer,
        run_featurize_episodes,
    )

    apply_curation_seed(select_seed(cfg))
    settings = GradingFeaturizeSettings.from_cfg(cfg)
    ds_name = next(iter(task_cfg.data.train_datasets))
    loader_cfg = select_curation_loader(cfg, ds_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    featurizer = build_grading_featurizer(
        settings,
        device,
        image_batch_size=loader_cfg.global_frame_batch_size,
        seed=select_seed(cfg),
    )
    out_dir = _features_task_dir(settings.feature_version, task_name)
    manifest = run_featurize_episodes(
        all_episodes,
        featurizer,
        loader_cfg,
        settings,
        out_dir,
        progress=tag,
    )
    training_outputs_volume.commit()
    n_ok = sum(1 for m in manifest if not m.get("skipped"))
    print(
        f"{tag} done: {n_ok}/{len(shard_hashes)} episodes cached "
        f"in {_time.perf_counter() - t0:.1f}s"
    )
    return shard_idx, manifest


# ---------------------------------------------------------------------------
# Retrieval-grid renderer (CPU, zarr volume for frame decode)
# ---------------------------------------------------------------------------


@app.function(
    gpu=GRID_RENDER_COMPUTE.gpu,
    cpu=GRID_RENDER_COMPUTE.cpu,
    memory=GRID_RENDER_COMPUTE.memory_mb,
    timeout=86400,
    secrets=_SHARED_SECRETS,
    volumes={
        CFG.volume_mount_path: zarr_volume,
        CFG.output_mount_path: training_outputs_volume,
    },
)
def _render_retrieval_grids(
    task_name: str,
    grid_specs: list[dict],
    output_dir: str,
    git_remote: str,
    git_commit: str,
    hydra_args: tuple[str, ...],
    hf_token: str = "",
) -> int:
    """
    Render query + neighbor frame strips for the worst-disagreement states.

    Each spec: {"query": {"hash", "frame_idx"}, "z_spatial",
                "neighbors": [{"hash", "frame_idx", "sim"}, …]}.
    Saves JPEGs + a sidecar json under <output_dir>/grids/<task>/.
    """
    import numpy as _np

    _boot_container(git_remote, git_commit, hf_token)
    import cv2
    import hydra as _hydra
    import simplejpeg

    from egomimic.utils.aws.aws_data_utils import load_env

    load_env()
    cfg = _load_cfg(hydra_args)
    tag = f"[{task_name}][grids]"

    # Collect every (hash, frame) we need, grouped by episode.
    frames_by_hash: dict[str, set[int]] = {}
    for spec in grid_specs:
        frames_by_hash.setdefault(spec["query"]["hash"], set()).add(
            int(spec["query"]["frame_idx"])
        )
        for nbr in spec["neighbors"]:
            frames_by_hash.setdefault(nbr["hash"], set()).add(int(nbr["frame_idx"]))

    task_cfg = _inject_episode_ids(cfg, list(frames_by_hash))
    episodes = _build_episode_map(task_cfg, _hydra)
    print(f"{tag} decoding frames from {len(episodes)} episodes")

    image_key = str(cfg.featurize.image_key)
    side = 224
    decoded: dict[tuple[str, int], _np.ndarray] = {}
    for ep_hash, frame_set in frames_by_hash.items():
        ds = episodes.get(ep_hash)
        if ds is None:
            continue
        try:
            if ds.pause_removal_epsilon is not None and ds.keep_indices is None:
                ds.precompute_pause_filter()
            ds.preload_zarr_arrays()
            img_zarr_key = ds.key_map[image_key]["zarr_key"]
            jpeg_frames = ds._zarr_bulk_cache[img_zarr_key]
            if ds.keep_indices is not None:
                jpeg_frames = jpeg_frames[ds.keep_indices]
            for logical in sorted(frame_set):
                if logical >= len(jpeg_frames):
                    continue
                chw = ds._decode_jpeg_to_chw(jpeg_frames[logical])
                hwc = (_np.transpose(chw, (1, 2, 0)) * 255.0).astype(_np.uint8)
                decoded[(ep_hash, logical)] = cv2.resize(hwc, (side, side))
        except Exception as exc:
            print(f"{tag} decode failed for {ep_hash[:8]}: {exc}")
        finally:
            ds._zarr_bulk_cache = None

    grid_dir = Path(output_dir) / "grids" / task_name
    grid_dir.mkdir(parents=True, exist_ok=True)
    n_saved = 0
    sidecar = []
    for i, spec in enumerate(grid_specs):
        q = spec["query"]
        tiles = [decoded.get((q["hash"], int(q["frame_idx"])))]
        for nbr in spec["neighbors"]:
            tiles.append(decoded.get((nbr["hash"], int(nbr["frame_idx"]))))
        tiles = [t for t in tiles if t is not None]
        if len(tiles) < 2:
            continue
        # Query tile gets a thick border so it reads at a glance.
        tiles[0] = cv2.copyMakeBorder(
            tiles[0][8:-8, 8:-8], 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(255, 0, 0)
        )
        strip = _np.concatenate(tiles, axis=1)
        fname = f"state_{i:03d}_z{spec['z_spatial']:.1f}_{q['hash'][:8]}.jpg"
        with open(grid_dir / fname, "wb") as f:
            f.write(simplejpeg.encode_jpeg(_np.ascontiguousarray(strip), quality=88))
        sidecar.append({"file": fname, **spec})
        n_saved += 1

    with open(grid_dir / "grids.json", "w") as f:
        json.dump(sidecar, f, indent=2)
    training_outputs_volume.commit()
    print(f"{tag} saved {n_saved}/{len(grid_specs)} grids → {grid_dir}")
    return n_saved


# ---------------------------------------------------------------------------
# Per-task CPU scorer: cache diff → featurize fan-out → k-NN grading
# ---------------------------------------------------------------------------


@app.function(
    gpu=GRADE_SCORE_COMPUTE.gpu,
    cpu=GRADE_SCORE_COMPUTE.cpu,
    memory=GRADE_SCORE_COMPUTE.memory_mb,
    timeout=86400,
    secrets=_SHARED_SECRETS,
    volumes={CFG.output_mount_path: training_outputs_volume},
)
def _grade_task_split(
    task_name: str,
    task_episode_hashes: list[str],
    output_dir: str,
    git_remote: str,
    git_commit: str,
    hydra_args: tuple[str, ...],
    hf_token: str = "",
) -> tuple[str, dict]:
    """CPU orchestrator for one task: featurize missing episodes, then grade."""
    import time as _time

    from omegaconf import OmegaConf

    t0 = _time.perf_counter()
    _boot_container(git_remote, git_commit, hf_token)
    from egomimic.utils.aws.aws_data_utils import load_env

    load_env()
    cfg = _load_cfg(hydra_args)
    tag = f"[{task_name}]"

    if not task_episode_hashes:
        print(f"{tag} No episodes — skipping")
        return task_name, {}

    from egomimic.curation.config import apply_curation_seed, select_seed
    from egomimic.curation.grading_pipeline import (
        GradingFeaturizeSettings,
        load_episode_features,
    )
    from egomimic.curation.knn_grading import KnnGradeSettings, grade_task

    apply_curation_seed(select_seed(cfg))
    feat_settings = GradingFeaturizeSettings.from_cfg(cfg)
    knn_settings = KnnGradeSettings.from_cfg(OmegaConf.select(cfg, "knn"))
    feat_dir = _features_task_dir(feat_settings.feature_version, task_name)

    # ── Diff against the feature cache; featurize only what's missing ───────
    training_outputs_volume.reload()
    hashes = sorted(set(task_episode_hashes))
    missing = [h for h in hashes if not (feat_dir / f"{h}.npz").is_file()]
    print(f"{tag} {len(hashes)} episodes ({len(missing)} missing from cache)")

    if missing:
        max_per_shard = int(
            OmegaConf.select(cfg, "max_episodes_per_shard", default=100)
        )
        shards = [
            missing[i : i + max_per_shard]
            for i in range(0, len(missing), max_per_shard)
        ]
        print(f"{tag} spawning {len(shards)} featurize shard(s)")
        handles = [
            _featurize_task_shard.spawn(
                task_name, idx, shard, git_remote, git_commit, hydra_args, hf_token
            )
            for idx, shard in enumerate(shards)
        ]
        n_failures = 0
        for idx, handle in enumerate(handles):
            try:
                handle.get(timeout=CFG.timeout_seconds)
            except Exception as exc:
                n_failures += 1
                print(f"{tag}[shard {idx}] FAILED: {exc}")
        print(f"{tag} featurize done ({n_failures} shard failure(s))")
        training_outputs_volume.reload()

    # ── Load caches and grade (pure numpy) ──────────────────────────────────
    t_load = _time.perf_counter()
    episodes = []
    n_load_failures = 0
    for h in hashes:
        path = feat_dir / f"{h}.npz"
        if not path.is_file():
            n_load_failures += 1
            continue
        try:
            episodes.append(load_episode_features(path, h, knn_settings))
        except Exception as exc:
            n_load_failures += 1
            print(f"{tag} cache load failed for {h[:8]}: {exc}")
    print(
        f"{tag} loaded {len(episodes)} episodes "
        f"({n_load_failures} unavailable) in {_time.perf_counter() - t_load:.1f}s"
    )

    t_grade = _time.perf_counter()
    result = grade_task(episodes, knn_settings)
    result["task_summary"]["n_unavailable"] = n_load_failures
    print(
        f"{tag} graded {len(result['per_episode'])} episodes "
        f"in {_time.perf_counter() - t_grade:.1f}s"
    )

    # ── Persist the full per-task result (incl. debug states) ───────────────
    tasks_dir = Path(output_dir) / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    with open(tasks_dir / f"{task_name}.json", "w") as f:
        json.dump(result, f, indent=2)
    training_outputs_volume.commit()

    # ── Optional retrieval-grid spot-checks on the worst states ─────────────
    grids_cfg = OmegaConf.select(cfg, "grids", default=None)
    if grids_cfg is not None and bool(grids_cfg.get("enabled", False)):
        specs = _build_grid_specs(
            result,
            per_task=int(grids_cfg.get("per_task", 24)),
            neighbors_shown=int(grids_cfg.get("neighbors_shown", 6)),
        )
        if specs:
            try:
                _render_retrieval_grids.spawn(
                    task_name,
                    specs,
                    output_dir,
                    git_remote,
                    git_commit,
                    hydra_args,
                    hf_token,
                ).get(timeout=CFG.timeout_seconds)
            except Exception as exc:
                print(f"{tag} grid render FAILED (non-fatal): {exc}")

    print(f"{tag} task total: {_time.perf_counter() - t0:.1f}s")
    return task_name, result


def _build_grid_specs(result: dict, per_task: int, neighbors_shown: int) -> list[dict]:
    """Worst debug states across all episodes of one task → renderer specs."""
    states = []
    for ep_hash, dbg_list in result.get("debug_states", {}).items():
        for dbg in dbg_list:
            states.append(
                {
                    "query": {"hash": ep_hash, "frame_idx": dbg["frame_idx"]},
                    "z_spatial": dbg["z_spatial"],
                    "neighbors": dbg["neighbors"][:neighbors_shown],
                }
            )
    states.sort(key=lambda s: -s["z_spatial"])
    return states[:per_task]


# ---------------------------------------------------------------------------
# Orchestrator: SQL task grouping + per-task fan-out
# ---------------------------------------------------------------------------


@app.function(
    gpu=CURATE_ORCHESTRATOR.gpu,
    cpu=CURATE_ORCHESTRATOR.cpu,
    memory=CURATE_ORCHESTRATOR.memory_mb,
    timeout=CFG.timeout_seconds,
    secrets=_SHARED_SECRETS,
    volumes={
        CFG.volume_mount_path: zarr_volume,
        CFG.output_mount_path: training_outputs_volume,
    },
)
def run_knn_grade(
    hydra_args: tuple[str, ...],
    git_remote: str,
    git_commit: str,
    init_submodules: bool = True,
    hf_token: str = "",
) -> str:
    """Orchestrator: SQL task grouping + per-task container fan-out."""
    import csv
    import io
    import sys as _sys
    import time as _time

    import numpy as _np

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    _prepare_repo(
        git_remote=git_remote,
        git_commit=git_commit,
        init_submodules=init_submodules,
    )
    _sys.path.insert(0, CFG.remote_repo_dir)
    os.chdir(CFG.remote_repo_dir)
    os.environ["MODAL_IS_REMOTE"] = "1"
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")

    import hydra as _hydra

    from egomimic.curation.config import apply_curation_seed, select_seed
    from egomimic.utils.aws.aws_data_utils import load_env

    load_env()
    cfg = _load_cfg(hydra_args)
    apply_curation_seed(select_seed(cfg))

    # ── 1. SQL task lookup: episode_hash → task_name ──────────────────────────
    print("Running SQL task lookup …")
    from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df

    engine = create_default_engine()
    full_df = episode_table_to_df(engine)
    if "is_deleted" in full_df.columns:
        full_df = full_df[full_df["is_deleted"] != True]  # noqa: E712

    hash_to_task: dict[str, str] = {}
    if "task" in full_df.columns:
        hash_to_task = dict(
            zip(full_df["episode_hash"], full_df["task"].fillna("unknown"))
        )

    # ── 2. Resolve episodes via data-config resolvers ─────────────────────────
    by_task: dict[str, list[str]] = {}
    for ds_name, ds_cfg in cfg.data.train_datasets.items():
        resolver = _hydra.utils.instantiate(ds_cfg.resolver)
        dataset_filter = (
            _hydra.utils.instantiate(ds_cfg.filters) if "filters" in ds_cfg else None
        )
        resolved = resolver.resolve(filters=dataset_filter)
        print(f"[{ds_name}] {len(resolved)} episodes after resolver")
        for episode_hash in resolved:
            task = hash_to_task.get(episode_hash) or "unknown"
            if str(task) in ("nan", "None", ""):
                task = "unknown"
            by_task.setdefault(task, []).append(episode_hash)

    total_episodes = sum(len(v) for v in by_task.values())
    print(
        f"Episode partition: {total_episodes} episodes across {len(by_task)} tasks — "
        + ", ".join(f"{t}:{len(h)}" for t, h in sorted(by_task.items())[:5])
        + ("…" if len(by_task) > 5 else "")
    )
    if total_episodes == 0:
        print("No episodes found — check data config resolver settings")
        return ""

    # ── 3. Output dir ─────────────────────────────────────────────────────────
    timestamp = _time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = (
        Path(CFG.output_mount_path)
        / "knn_grading"
        / cfg.name
        / f"{cfg.description}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 4. Fan-out per-task grading containers ────────────────────────────────
    print(f"Spawning {len(by_task)} per-task graders …")
    t0 = _time.time()
    handles = [
        (
            task_name,
            _grade_task_split.spawn(
                task_name,
                episode_hashes,
                str(output_dir),
                git_remote,
                git_commit,
                hydra_args,
                hf_token,
            ),
        )
        for task_name, episode_hashes in sorted(by_task.items())
    ]

    results_by_task: dict[str, dict] = {}
    n_failures = 0
    for task_name, handle in handles:
        try:
            _label, result = handle.get(timeout=CFG.timeout_seconds)
            if result:
                results_by_task[task_name] = result
        except Exception as exc:
            n_failures += 1
            print(f"[task] FAILED ({task_name}): {exc}")

    elapsed = _time.time() - t0
    print(
        f"Grading done in {elapsed:.1f}s — "
        f"{len(results_by_task)}/{len(by_task)} tasks succeeded ({n_failures} failed)"
    )

    # ── 5. Merge + save outputs ───────────────────────────────────────────────
    def _score_of(metrics: dict) -> float:
        v = metrics.get("primary_score")
        return float(v) if v is not None and _np.isfinite(v) else float("nan")

    def _sort_key(metrics: dict) -> float:
        v = _score_of(metrics)
        return v if _np.isfinite(v) else -1.0

    scores_by_task: dict[str, dict[str, float]] = {}
    report: dict[str, dict] = {}
    review_cols = [
        "rank",
        "hash",
        "primary_score",
        "frac_flagged_spatial",
        "frac_flagged_velocity",
        "longest_flagged_run",
        "mean_z_spatial",
        "coverage_frac",
        "mean_ambiguity_pctile",
        "n_states",
        "label",
    ]
    (output_dir / "review").mkdir(exist_ok=True)
    for task_name, result in results_by_task.items():
        per_ep = result.get("per_episode", {})
        ranked = sorted(per_ep.items(), key=lambda kv: _sort_key(kv[1]), reverse=True)
        scores_by_task[task_name] = {h: _score_of(m) for h, m in ranked}
        report[task_name] = {
            "task_summary": result.get("task_summary", {}),
            "per_episode": dict(ranked),
        }
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=review_cols, extrasaction="ignore")
        writer.writeheader()
        for rank, (h, m) in enumerate(ranked):
            writer.writerow({"rank": rank, "hash": h, "label": "", **m})
        (output_dir / "review" / f"{task_name}.csv").write_text(buf.getvalue())

    stats = {
        "total_input": total_episodes,
        "n_tasks": len(results_by_task),
        "n_task_failures": n_failures,
        "graded": sum(len(s) for s in scores_by_task.values()),
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(output_dir / "knn_scores_by_task.json", "w") as f:
        json.dump(scores_by_task, f, indent=2)
    with open(output_dir / "knn_report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(output_dir / "grading_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    training_outputs_volume.commit()
    print(
        f"k-NN grading done — graded={stats['graded']} "
        f"n_tasks={len(results_by_task)} output={output_dir}"
    )
    print("NOTE: primary_score is sorted worst-first (higher = more off-mode).")
    return str(output_dir)


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def submit_knn_grade(*hydra_args: str) -> None:
    """Fire-and-forget: spawn a grading job from an already-pushed commit."""
    hydra_args, init_submodules = pop_init_submodules(hydra_args)
    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print(
            "Warning: local repo has uncommitted changes. "
            "Modal will run the last committed state only."
        )
    print(f"Submitting k-NN grading at commit {git_commit[:12]} from {git_remote}")
    if not init_submodules:
        print("Skipping git submodule init (init_submodules=false)")
    handle = run_knn_grade.spawn(
        tuple(hydra_args),
        git_remote,
        git_commit,
        init_submodules=init_submodules,
        hf_token=_local_hf_token(),
    )
    _env = os.environ.get("MODAL_ENVIRONMENT", "robotics")
    _app = os.environ.get("MODAL_APP_NAME", "egomimic-training")
    print(f"Submitted Modal k-NN grading job: {handle.object_id}")
    print(f"Monitor: https://modal.com/apps/mecka/{_env}/apps/{_app}")


# ---------------------------------------------------------------------------
# python egomimic/modal/knnGradeModal.py name=my_run description=test [overrides…]
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

    task_compute = ModalCompute.from_mapping(
        modal_env,
        default_gpu="L40S",
        default_cpu=16.0,
        default_memory_mb=131072,
    )
    print(f"Modal app:                                  {modal_env['MODAL_APP_NAME']}")
    print(
        f"Modal grading orchestrator (fixed):         {CURATE_ORCHESTRATOR.summary()}"
    )
    print(
        f"Modal grading per-task CPU scorer:          {GRADE_SCORE_COMPUTE.summary()}"
    )
    print(f"Modal grading featurize-shard GPU worker:   {task_compute.summary()}")

    launch_detached(Path(__file__).resolve(), "submit_knn_grade", hydra_args, modal_env)
