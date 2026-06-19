"""Global-shuffle shard builder for fast distributed training.

WHY
---
The zarr volume has high random-read latency. During training with episode-level
shuffling each batch touches many different zarr files on the network volume —
each with ~312ms seek cost. True global frame shuffling (sampling uniformly from
all frames in the dataset) would require N random seeks per batch, which is
prohibitively slow.

Solution: pre-build globally-shuffled shards in an ABCDL-inspired format.
Each shard is a fixed-size random sample of frames drawn from across all episodes.
At training time, shards are loaded sequentially from ephemeral local disk (fast)
rather than random-access on the network volume.

FORMAT
------
Each shard is a pair of files:
  {shard_id}.mp4   — camera frames, H.264 (+faststart, no B-frames, fixed GOP)
  {shard_id}.npz   — numpy arrays: action (T, H, D) and meta (episode_hash, frame_idx)

The MP4 uses:
  +faststart   moves the moov atom to the file front so playback/decoding can
               begin without reading the full file first.
  bf=0         disables B-frames so each frame depends only on the most recent
               keyframe, never on future frames. This allows seeking to any frame
               with at most one keyframe read.
  g=<gop>      fixed GOP (one keyframe every `gop` frames at default fps=30,
               i.e. one keyframe per second). Combined with bf=0 and a fixed fps,
               keyframe positions are at deterministic byte offsets — the file
               index can be reconstructed without reading metadata, enabling fast
               partial reads via torchcodec.

TRANSFORMS & FILTERING
-----------------------
Transforms (action chunking, coordinate frame conversion) are applied at shard
build time using the same Mecka.get_transform_list() used at training time.
Invalid frames (zero quaternion, cut rows) are filtered out using the same
_curation_valid_logical_indices logic used in curation.
This means shards contain only clean, pre-transformed frames.

VALIDATION
----------
Validation episodes are excluded from shards (pass --val-hashes-path or provide
val_hashes inside Modal). Validation uses normal zarr access via ModalEpisodeResolver.

USAGE
-----
# Dry run — list plan without building
modal run --env robotics egomimic/modal/build_global_shuffle_shards.py -- --dry-run

# Build all episodes on the zarr volume (legacy; no data config)
modal run --env robotics egomimic/modal/build_global_shuffle_shards.py

# Production build (default data config mecka_all_zarr, 1.5x coverage, 6k frames/shard)
modal run --env robotics egomimic/modal/build_global_shuffle_shards.py -- \\
    --coverage-multiplier 1.5 \\
    --frames-per-shard 6000 \\
    --output-subdir global_shuffle_v2

# Debug smoke build (300 episodes via mecka_all_zarr_debug300)
modal run --env robotics egomimic/modal/build_global_shuffle_shards.py -- \\
    --data-config mecka_all_zarr_debug300 \\
    --output-subdir global_shuffle_debug300

# With extra validation exclusion list (merged with valid_datasets from config)
modal run --env robotics egomimic/modal/build_global_shuffle_shards.py -- \\
    --data-config mecka_all_zarr \\
    --val-hashes-path /path/to/val_hashes.json

OUTPUT
------
gs_volume (global_shuffle, version=2) mounted at /mnt/zarr-gs:
    /mnt/zarr-gs/<output_subdir>/
        000000_0000.mp4
        000000_0000.npz
        ...
        index.json   (shard manifest with coverage stats)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Honor +modal_volume=mecka_data_v2 before volume mounts are resolved below.
for _arg in sys.argv[1:]:
    _key, _sep, _val = _arg.lstrip("+").partition("=")
    if _sep and _key == "modal_volume":
        os.environ["MODAL_VOLUME"] = _val

import modal
from modal_setup import VOLUME_MAP, app, zarr_volume, CFG, image

_SHARED_SECRETS = [modal.Secret.from_name(name) for name in CFG.secret_names]

DEFAULT_ZARR_VOLUME = "mecka_data_v2"


def _build_zarr_volume_spec() -> tuple[modal.Volume, str, str]:
    """Return (volume_obj, mount_path, volume_name) for the zarr episode volume."""
    vol_name = os.environ.get("MODAL_VOLUME", DEFAULT_ZARR_VOLUME)
    if vol_name == DEFAULT_ZARR_VOLUME:
        return zarr_volume, CFG.volume_mount_path, vol_name
    if vol_name in VOLUME_MAP:
        vol_obj, mount_path = VOLUME_MAP[vol_name]
        if mount_path != CFG.volume_mount_path:
            raise ValueError(
                "Global shuffle shard builder reads zarr episodes from "
                f"{CFG.volume_mount_path!r}. Volume {vol_name!r} mounts at "
                f"{mount_path!r}. Use +modal_volume={DEFAULT_ZARR_VOLUME}."
            )
        return vol_obj, mount_path, vol_name
    raise ValueError(
        f"Unknown Modal volume {vol_name!r}. "
        f"Supported zarr volume: {DEFAULT_ZARR_VOLUME}."
    )


_zarr_input_volume, INPUT_MOUNT, ZARR_VOLUME_NAME = _build_zarr_volume_spec()

# ---------------------------------------------------------------------------
# Global-shuffle volume
# ---------------------------------------------------------------------------

gs_volume = modal.Volume.from_name("global_shuffle", create_if_missing=True, version=2)
GS_MOUNT = "/mnt/zarr-gs"

EPISODES_PER_WORKER = 50
DEFAULT_NUM_WORKERS = 50
DEFAULT_FRAMES_PER_SHARD = 2000
DEFAULT_COVERAGE_MULTIPLIER = 2.0
DEFAULT_GOP = 30
DEFAULT_FPS = 30
DEFAULT_OUTPUT_SUBDIR = "global_shuffle_v1"
DEFAULT_DATA_CONFIG = "mecka_all_zarr"
DEBUG_DATA_CONFIG = "mecka_all_zarr_debug300"

ACTION_KEY = "actions_cartesian"
IMAGE_KEY = "observations.images.front_img_1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _batch_episodes_for_workers(
    episode_paths: list[str],
    num_workers: int,
) -> list[list[str]]:
    """Split episodes into ``num_workers`` batches (one Modal container each)."""
    n = len(episode_paths)
    if n == 0:
        return []
    if num_workers <= 0:
        return [
            episode_paths[i : i + EPISODES_PER_WORKER]
            for i in range(0, n, EPISODES_PER_WORKER)
        ]
    chunk_size = max(1, -(-n // num_workers))
    return [
        episode_paths[i : i + chunk_size]
        for i in range(0, n, chunk_size)
    ]


def _write_mp4(
    frames_iter,
    path: Path,
    fps: int,
    gop: int,
    n_frames: int | None = None,
) -> None:
    """Write frames from an iterable of (C, H, W) float32 [0,1] arrays to H.264 MP4.

    Accepts an iterable (not a pre-stacked array) so callers can decode frames
    on-the-fly without holding all decoded data in RAM simultaneously.

    Options applied:
      +faststart  — moov atom at file front for streaming-friendly decoding.
      bf=0        — no B-frames; each frame depends only on its nearest keyframe.
      g=<gop>     — fixed GOP size; keyframe positions are deterministic.
    """
    import av
    import numpy as np

    container = av.open(
        str(path),
        mode="w",
        format="mp4",
        options={"movflags": "+faststart"},
    )
    stream = None

    for t, frame_chw in enumerate(frames_iter):
        if stream is None:
            _, H, W = frame_chw.shape
            stream = container.add_stream("libx264", rate=fps)
            stream.width = W
            stream.height = H
            stream.pix_fmt = "yuv420p"
            stream.options = {
                "bf": "0",
                "g": str(gop),
                "keyint_min": str(gop),
                "preset": "fast",
                "crf": "23",
            }
        frame_rgb = (frame_chw.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
        av_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        av_frame.pts = t
        for packet in stream.encode(av_frame):
            container.mux(packet)

    if stream is not None:
        for packet in stream.encode():
            container.mux(packet)
    container.close()


# ---------------------------------------------------------------------------
# Remote: resolve episodes from a Hydra data config (or list entire volume)
# ---------------------------------------------------------------------------


def _resolve_dataset_episodes(ds_cfg) -> dict[str, str]:
    """Resolve one train/valid dataset block → {episode_hash: local_path}."""
    import random

    import hydra.utils as hydra_utils
    from omegaconf import OmegaConf

    from egomimic.rldb.zarr.zarr_dataset_multi import SEED, split_dataset_names

    resolver_cfg = OmegaConf.select(ds_cfg, "resolver")
    if resolver_cfg is None:
        raise ValueError("dataset config missing resolver (use ModalEpisodeResolver)")

    resolver = hydra_utils.instantiate(resolver_cfg)
    filters = (
        hydra_utils.instantiate(ds_cfg.filters) if "filters" in ds_cfg else None
    )
    resolved = resolver.resolve(filters=filters)

    mode = OmegaConf.select(ds_cfg, "mode", default="train")
    valid_ratio = float(OmegaConf.select(ds_cfg, "valid_ratio", default=0.2))
    train_coll, valid_coll = split_dataset_names(
        resolved.keys(), valid_ratio=valid_ratio, seed=SEED
    )

    if mode == "train":
        chosen = train_coll
    elif mode == "valid":
        chosen = valid_coll
    elif mode == "total":
        chosen = set(resolved.keys())
    elif mode == "percent":
        percent = float(OmegaConf.select(ds_cfg, "percent", default=0.1))
        all_names = sorted(resolved.keys())
        rng = random.Random(SEED)
        rng.shuffle(all_names)
        n_keep = int(len(all_names) * percent)
        if percent > 0.0:
            n_keep = max(1, n_keep)
        chosen = set(all_names[:n_keep])
    else:
        raise ValueError(f"Unknown dataset mode: {mode}")

    return {
        ep_hash: str(resolved[ep_hash].episode_path)
        for ep_hash in chosen
        if ep_hash in resolved
    }


@app.function(
    image=image,
    volumes={INPUT_MOUNT: _zarr_input_volume},
    secrets=_SHARED_SECRETS,
    cpu=2,
    memory=8192,
    timeout=600,
)
def resolve_episodes_from_config_fn(
    data_config: str,
    git_remote: str,
    git_commit: str,
) -> dict:
    """Resolve train episode paths and val hashes from a Hydra data config."""
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root")
    from modal_setup import CFG, _prepare_repo_light

    _prepare_repo_light(git_remote, git_commit, init_submodules=False)
    remote_repo = Path(CFG.remote_repo_dir)
    sys.path.insert(0, str(remote_repo))
    os.chdir(remote_repo)
    os.environ["MODAL_IS_REMOTE"] = "1"

    from omegaconf import OmegaConf

    from egomimic.utils.aws.aws_data_utils import load_env

    load_env()
    _zarr_input_volume.reload()

    cfg_path = remote_repo / "egomimic/hydra_configs/data" / f"{data_config}.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Data config not found: {data_config}")

    data_cfg = OmegaConf.load(cfg_path)
    train_datasets = OmegaConf.select(data_cfg, "train_datasets")
    if not train_datasets:
        raise ValueError(f"{data_config}: no train_datasets section")

    train_paths: dict[str, str] = {}
    for ds_name, ds_cfg in train_datasets.items():
        resolved = _resolve_dataset_episodes(ds_cfg)
        print(f"[{data_config}/{ds_name}] {len(resolved)} train episodes resolved")
        train_paths.update(resolved)

    if not train_paths:
        raise ValueError(f"{data_config}: resolver matched no train episodes")

    val_hashes: set[str] = set()
    valid_datasets = OmegaConf.select(data_cfg, "valid_datasets", default=None)
    if valid_datasets:
        for ds_name, ds_cfg in valid_datasets.items():
            resolved = _resolve_dataset_episodes(ds_cfg)
            print(f"[{data_config}/{ds_name}] {len(resolved)} valid episodes resolved")
            val_hashes.update(resolved.keys())

    return {
        "episode_paths": sorted(train_paths.values()),
        "val_hashes": sorted(val_hashes),
    }


@app.function(
    image=image,
    volumes={INPUT_MOUNT: _zarr_input_volume},
    cpu=1,
    memory=4096,
    timeout=120,
)
def list_episodes_fn() -> list[str]:
    """Return sorted list of all episode directory paths from the zarr volume."""
    _zarr_input_volume.reload()
    input_root = Path(INPUT_MOUNT)
    return sorted(str(p) for p in input_root.iterdir() if p.is_dir())


# ---------------------------------------------------------------------------
# Remote: process episodes → write globally-shuffled shards
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    volumes={
        INPUT_MOUNT: _zarr_input_volume,
        GS_MOUNT: gs_volume,
    },
    cpu=8,
    memory=65536,
    timeout=7200,
    max_containers=DEFAULT_NUM_WORKERS,
)
def build_shard_worker(job: dict) -> dict:
    """Process a batch of episodes, globally shuffle their frames, write shards.

    Memory-efficient two-phase approach to avoid OOM:

    Phase 1 (per episode, no image decoding):
      - Load only pose/action zarr arrays via ZarrDataset with non-image key_map.
      - Apply _curation_valid_logical_indices + batched transforms → actions_cartesian.
      - Load JPEG bytes (compressed) for valid frames from zarr — ~30 KB/frame vs
        ~1 MB decoded. 50 eps × 2000 frames × 30 KB ≈ 3 GB total (vs 100 GB+).

    Phase 2: build global shuffle plan (index arithmetic only, no data).

    Phase 3 (per shard):
      - Decode JPEG bytes one frame at a time into the MP4 encoder.
      - Never hold more than one decoded (C, H, W) float32 frame in RAM.

    Args:
        job: dict with keys: worker_id, episode_paths, output_subdir, git_remote,
             git_commit, coverage_multiplier, frames_per_shard, fps, gop, seed, val_hashes,
             data_config (optional Hydra data config for key_map/transform_list).
    """
    import sys
    import traceback
    import numpy as np
    from pathlib import Path

    worker_id: int = job["worker_id"]
    episode_paths: list[str] = job["episode_paths"]
    output_subdir: str = job["output_subdir"]
    git_remote: str = job["git_remote"]
    git_commit: str = job["git_commit"]
    coverage_multiplier: float = job["coverage_multiplier"]
    frames_per_shard: int = job["frames_per_shard"]
    fps: int = job["fps"]
    gop: int = job["gop"]
    seed: int = job["seed"]
    val_hashes: list[str] = job["val_hashes"]
    data_config: str = job.get("data_config", "")

    sys.path.insert(0, "/root")
    from modal_setup import _prepare_repo_light

    _prepare_repo_light(git_remote, git_commit, init_submodules=False)

    remote_repo = Path("/root/EgoVerse")
    sys.path.insert(0, str(remote_repo))

    _zarr_input_volume.reload()

    import zarr as zarr_lib
    from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset
    from egomimic.rldb.embodiment.human import Mecka

    if data_config:
        import hydra.utils as hydra_utils
        from omegaconf import OmegaConf

        cfg_path = remote_repo / "egomimic/hydra_configs/data" / f"{data_config}.yaml"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Data config not found: {data_config}")
        data_cfg = OmegaConf.load(cfg_path)
        first_ds = next(iter(data_cfg.train_datasets.values()))
        resolver = hydra_utils.instantiate(first_ds.resolver)
        full_key_map = resolver.key_map
        transform_list = resolver.transform_list
    else:
        full_key_map = Mecka.get_keymap(mode="cartesian")
        transform_list = Mecka.get_transform_list(mode="cartesian")

    # Camera keys are loaded directly from zarr as JPEG bytes — exclude from ZarrDataset
    # so preload_zarr_arrays() doesn't decode them into float32 arrays.
    non_img_key_map = {
        k: v
        for k, v in full_key_map.items()
        if v.get("key_type") not in ("camera_keys", "annotation_keys")
    }

    val_set = set(val_hashes)
    rng = np.random.default_rng(seed + worker_id)

    # =========================================================================
    # Phase 1: per-episode — compute valid actions + cache compressed JPEG bytes
    # =========================================================================
    episode_data: list[dict] = []

    for ep_path in episode_paths:
        ep_hash = Path(ep_path).name
        if ep_hash.endswith(".zarr"):
            ep_hash = ep_hash[:-5]
        if ep_hash in val_set:
            print(f"[worker {worker_id}] skipping val episode {ep_hash}")
            continue

        try:
            ds = ZarrDataset(
                Path(ep_path),
                key_map=non_img_key_map,
                transform_list=transform_list,
            )
            # Loads pose + action arrays only — no images, no float32 CHW blowup
            ds.preload_zarr_arrays()

            valid_logical = ds._curation_valid_logical_indices()
            if len(valid_logical) == 0:
                print(f"[worker {worker_id}] skipped {ep_hash}: no valid frames")
                continue

            batch_in = ds._build_batched_transform_batch(valid_logical)
            batch_out, ok_pos = ds._apply_transform_list_batched(batch_in)
            if batch_out is None or ACTION_KEY not in batch_out:
                print(f"[worker {worker_id}] skipped {ep_hash}: transform failed")
                continue

            actions = batch_out[ACTION_KEY].astype(np.float32)  # (T, H, D)
            valid_logical = valid_logical[ok_pos]

            real_indices = np.array(
                [ds._logical_to_real_index(int(li)) for li in valid_logical],
                dtype=np.int64,
            )

            # Read all JPEG bytes sequentially, keep only valid frames, free the rest.
            # Compressed JPEG: ~30 KB/frame × 2000 frames = 60 MB per episode.
            z = zarr_lib.open_group(str(ep_path), mode="r")
            jpeg_all = np.asarray(z["images.front_1"][:])
            valid_jpegs = jpeg_all[real_indices].copy()
            del jpeg_all

            episode_data.append({
                "episode_hash": ep_hash,
                "actions": actions,
                "jpegs": valid_jpegs,
            })
            print(f"[worker {worker_id}] {ep_hash}: {len(actions)} valid frames")

        except Exception as e:
            print(f"[worker {worker_id}] ERROR {ep_hash}: {e}")
            traceback.print_exc()
            continue

    if not episode_data:
        print(f"[worker {worker_id}] no valid episodes — skipping")
        return {
            "worker_id": worker_id,
            "n_shards": 0,
            "n_frames": 0,
            "n_source_frames": 0,
            "shard_ids": [],
            "episodes": [],
        }

    ep_stats = [
        {"episode_hash": ep["episode_hash"], "n_valid_frames": len(ep["actions"])}
        for ep in episode_data
    ]

    # =========================================================================
    # Phase 2: build global shuffle plan (index arithmetic only)
    # =========================================================================
    all_frames: list[tuple[int, int]] = []
    for ep_idx, ep in enumerate(episode_data):
        T = len(ep["actions"])
        all_frames.extend((ep_idx, t) for t in range(T))

    N = len(all_frames)
    n_target = int(np.ceil(coverage_multiplier * N))

    index_parts: list[np.ndarray] = []
    remaining = n_target
    while remaining > 0:
        perm = rng.permutation(N)
        index_parts.append(perm[: min(remaining, N)])
        remaining -= N
    global_order = np.concatenate(index_parts)[:n_target]

    # =========================================================================
    # Phase 3: write shards — decode JPEG bytes one frame at a time
    # =========================================================================
    output_root = Path(GS_MOUNT) / output_subdir
    output_root.mkdir(parents=True, exist_ok=True)

    n_shards = int(np.ceil(n_target / frames_per_shard))
    shard_ids: list[str] = []

    for shard_local in range(n_shards):
        start = shard_local * frames_per_shard
        end = min(start + frames_per_shard, n_target)
        sel = global_order[start:end]

        shard_id = f"{worker_id:05d}_{shard_local:04d}"
        mp4_path = output_root / f"{shard_id}.mp4"
        npz_path = output_root / f"{shard_id}.npz"

        if mp4_path.exists() and npz_path.exists():
            print(f"[worker {worker_id}] shard {shard_id} already exists, skipping")
            shard_ids.append(shard_id)
            continue

        shard_actions = np.stack([
            episode_data[all_frames[fi][0]]["actions"][all_frames[fi][1]]
            for fi in sel
        ])  # (T, H, D)

        def _frame_iter(sel_indices):
            for fi in sel_indices:
                ep_idx, frame_pos = all_frames[fi]
                jpeg_bytes = episode_data[ep_idx]["jpegs"][frame_pos]
                yield ZarrDataset._decode_jpeg_to_chw(jpeg_bytes)

        try:
            _write_mp4(_frame_iter(sel), mp4_path, fps=fps, gop=gop)
        except Exception as e:
            print(f"[worker {worker_id}] MP4 write failed for shard {shard_id}: {e}")
            mp4_path.unlink(missing_ok=True)
            continue

        np.savez_compressed(npz_path, action=shard_actions)
        shard_ids.append(shard_id)
        print(f"[worker {worker_id}] shard {shard_id}: {len(sel)} frames → {mp4_path.name} + {npz_path.name}")

    gs_volume.commit()

    return {
        "worker_id": worker_id,
        "n_shards": len(shard_ids),
        "n_frames": n_target,
        "n_source_frames": N,
        "shard_ids": shard_ids,
        "episodes": ep_stats,
    }


# ---------------------------------------------------------------------------
# Remote: write shard index
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    volumes={GS_MOUNT: gs_volume},
    cpu=1,
    memory=4096,
    timeout=120,
)
def write_index_fn(
    output_subdir: str,
    results: list[dict],
    coverage_multiplier: float,
    frames_per_shard: int,
    gop: int,
    fps: int,
) -> None:
    """Write index.json to the output directory."""
    import json
    from pathlib import Path

    all_shards = []
    total_source_frames = 0
    total_shards = 0
    episode_manifest: list[dict] = []

    for r in results:
        if not isinstance(r, dict):
            continue
        all_shards.extend(r.get("shard_ids", []))
        total_source_frames += r.get("n_source_frames", 0)
        total_shards += r.get("n_shards", 0)
        episode_manifest.extend(r.get("episodes", []))

    index = {
        "coverage_multiplier": coverage_multiplier,
        "frames_per_shard": frames_per_shard,
        "gop": gop,
        "fps": fps,
        "action_key": ACTION_KEY,
        "image_key": IMAGE_KEY,
        "n_shards": total_shards,
        "n_source_frames": total_source_frames,
        "n_covered_frames": int(total_source_frames * coverage_multiplier),
        "shard_ids": sorted(all_shards),
        "episodes": episode_manifest,
    }

    out = Path(GS_MOUNT) / output_subdir / "index.json"
    out.write_text(json.dumps(index, indent=2))
    gs_volume.commit()
    print(f"Wrote index.json: {total_shards} shards from {len(episode_manifest)} episodes")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    dry_run: bool = False,
    data_config: str = DEFAULT_DATA_CONFIG,
    coverage_multiplier: float = DEFAULT_COVERAGE_MULTIPLIER,
    frames_per_shard: int = DEFAULT_FRAMES_PER_SHARD,
    num_workers: int = DEFAULT_NUM_WORKERS,
    gop: int = DEFAULT_GOP,
    fps: int = DEFAULT_FPS,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    val_hashes_path: str = "",
    seed: int = 42,
) -> None:
    """Orchestrate global-shuffle shard building.

    Args:
        dry_run:             Print plan without launching shard workers.
        data_config:         Hydra data config name (default mecka_all_zarr).
                             Debug smoke: mecka_all_zarr_debug300 (300 episodes).
                             Episodes are resolved via train_datasets.*.resolver;
                             valid_datasets are excluded from shards.
        coverage_multiplier: Total frames = coverage_multiplier * dataset_frames.
                             Default 2.0 means 2x coverage (each frame appears ~2x).
        frames_per_shard:    Frames per shard file (default 2000).
        num_workers:         Number of parallel Modal containers (default 50).
        gop:                 MP4 GOP size / keyframe interval (default 30).
        fps:                 MP4 frame rate for encoding (default 30).
        output_subdir:       Sub-directory under /mnt/zarr-gs (default global_shuffle_v1).
        val_hashes_path:     Path to JSON list of extra validation episode hashes to exclude.
        seed:                Random seed for reproducibility (default 42).
    """
    import subprocess

    def _git(args):
        return subprocess.check_output(args, cwd=str(Path(__file__).parent.parent.parent), text=True).strip()

    git_remote = _git(["git", "config", "--get", "remote.origin.url"])
    git_commit = _git(["git", "rev-parse", "HEAD"])

    val_hashes: list[str] = []
    if val_hashes_path:
        import json as _json

        val_hashes = _json.loads(Path(val_hashes_path).read_text())
        print(f"Loaded {len(val_hashes)} validation episode hashes from {val_hashes_path}")

    if data_config:
        print(f"Resolving episodes from data config {data_config!r}...")
        resolved = resolve_episodes_from_config_fn.remote(
            data_config, git_remote, git_commit
        )
        all_episodes = resolved["episode_paths"]
        config_val_hashes = resolved.get("val_hashes", [])
        if config_val_hashes:
            merged = set(val_hashes) | set(config_val_hashes)
            val_hashes = sorted(merged)
            print(
                f"Excluding {len(config_val_hashes)} val episodes from "
                f"{data_config} valid_datasets"
            )
    else:
        print("Listing episodes from zarr volume (no data_config)...")
        all_episodes = list_episodes_fn.remote()

    n_episodes = len(all_episodes)
    print(f"Found {n_episodes} episodes to convert")

    # Batch episodes across num_workers parallel containers
    batches = _batch_episodes_for_workers(all_episodes, num_workers)
    n_workers = len(batches)
    episodes_per_worker = len(batches[0]) if batches else 0

    est_source_frames = n_episodes * 150  # rough estimate ~150 frames/episode
    est_output_frames = int(est_source_frames * coverage_multiplier)
    est_shards = int(est_output_frames / frames_per_shard)

    print(f"\nPlan:")
    print(f"  Zarr volume:       {ZARR_VOLUME_NAME} @ {INPUT_MOUNT}")
    print(f"  Data config:       {data_config or '(all volume episodes)'}")
    print(f"  Episodes:          {n_episodes:,}")
    print(f"  Workers:           {n_workers} (target {num_workers})")
    print(f"  Episodes/worker:   ~{episodes_per_worker}")
    print(f"  Coverage:          {coverage_multiplier}x")
    print(f"  Frames/shard:      {frames_per_shard:,}")
    print(f"  Est. shards:       ~{est_shards:,}")
    print(f"  GOP:               {gop} frames")
    print(f"  FPS:               {fps}")
    print(f"  Output:            global_shuffle (v2) / {output_subdir}")
    print(f"  Val excluded:      {len(val_hashes)} episodes")

    if dry_run:
        print("\nDry run — not launching workers.")
        return

    print(f"\nLaunching {n_workers} shard workers...")

    jobs = [
        {
            "worker_id": i,
            "episode_paths": batch,
            "output_subdir": output_subdir,
            "git_remote": git_remote,
            "git_commit": git_commit,
            "coverage_multiplier": coverage_multiplier,
            "frames_per_shard": frames_per_shard,
            "fps": fps,
            "gop": gop,
            "seed": seed,
            "val_hashes": val_hashes,
            "data_config": data_config,
        }
        for i, batch in enumerate(batches)
    ]

    results = list(build_shard_worker.map(jobs, return_exceptions=True, wrap_returned_exceptions=False))

    ok = [r for r in results if isinstance(r, dict)]
    errs = [r for r in results if isinstance(r, Exception)]

    total_shards = sum(r.get("n_shards", 0) for r in ok)
    total_frames = sum(r.get("n_frames", 0) for r in ok)
    total_episodes = sum(len(r.get("episodes", [])) for r in ok)

    print(f"\nShard building complete:")
    print(f"  Workers ok:        {len(ok)} / {n_workers}")
    print(f"  Workers failed:    {len(errs)}")
    print(f"  Episodes:          {total_episodes:,}")
    print(f"  Shards written:    {total_shards:,}")
    print(f"  Total frames:      {total_frames:,}")
    if errs:
        print(f"\nFirst error: {errs[0]}")

    print("\nWriting index.json...")
    write_index_fn.remote(
        output_subdir=output_subdir,
        results=ok,
        coverage_multiplier=coverage_multiplier,
        frames_per_shard=frames_per_shard,
        gop=gop,
        fps=fps,
    )
    print("Done.")
