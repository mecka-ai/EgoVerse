"""
Pull a handful of robot tactile episodes from the T-Rex open-source dataset
and write them to Zarr.

T-Rex ("T-Rex: Tactile-Reactive Dexterous Manipulation", ZhuoyangLiu2005/T-Rex,
arXiv:2606.17055) is a 100-hour bimanual tactile-teleop dataset; ~50 hours /
5464 episodes are open-sourced on HuggingFace as `zekaiwang/trex_dataset` in
LeRobot v3.0 format (data/*.parquet + per-stream videos/*.mp4, chunked into
~100-200MB shard files, indexed by `meta/episodes/*.parquet`).

Per frame it has, among other things:
  - observation.state / action: 58-dim bimanual arm+hand joint positions
  - observation.tactile_force: 60-dim, 6D wrench x 5 fingers x 2 hands
  - 10 per-fingertip raw grayscale tactile-sensor video streams
    (tactile_{left,right}_raw_{thumb,index,middle,ring,pinky})
  - 10 per-fingertip "deform" video streams (derived; NOT pulled here)
  - 3 RGB camera videos (head_left, left_wrist, right_wrist; NOT pulled here)

This script only pulls the raw tactile video streams + tactile_force (+
state/action for context) for a handful of the earliest episodes, so the
download stays small: episodes are picked from the first data shard
(chunk-000/file-000) and only episodes whose *every* required stream also
falls in file-000 of its own chunk are used (true for the dataset's first
few episodes). That means ~1 data parquet (~70MB) + 10 video shards
(~200MB each, ~2GB total) rather than any part of the full ~50-hour set --
deform maps and RGB video are skipped entirely for this pilot to keep it
disk/time-light.

Usage:
    python trex_robot_to_zarr.py --out-dir /home/mecka/EgoVerse/zarr/tactile_robot --n-episodes 3
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ID = "zekaiwang/trex_dataset"

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
RAW_STREAMS = [f"tactile_{side}_raw_{finger}" for side in ("left", "right") for finger in FINGERS]


def hf_download(filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=filename)


def load_episode_frames(video_path: str, from_ts: float, to_ts: float, fps: float) -> np.ndarray:
    """Decode the [from_ts, to_ts) frame range of a shard video as grayscale (T, H, W) uint8."""
    import cv2

    start_frame = int(round(from_ts * fps))
    end_frame = int(round(to_ts * fps))
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    cap.release()
    return np.stack(frames, axis=0) if frames else np.zeros((0, 0, 0), dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description="T-Rex robot tactile -> Zarr (pilot)")
    ap.add_argument("--out-dir", required=True, help="Output directory for per-episode .zarr")
    ap.add_argument("--n-episodes", type=int, default=3, help="How many episodes to pull")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching dataset metadata (info.json, episodes index, tasks)...")
    info_path = hf_download("meta/info.json")
    import json

    info = json.loads(Path(info_path).read_text())
    fps = float(info["fps"])

    episodes_meta_path = hf_download("meta/episodes/chunk-000/file-000.parquet")
    episodes_df = pd.read_parquet(episodes_meta_path)

    # Only keep episodes whose data + all 10 raw tactile video streams live in
    # chunk 0 / file 0 -- true for the dataset's earliest episodes, and keeps
    # this pilot to a single shard per stream.
    def _all_in_shard0(row) -> bool:
        if row["data/chunk_index"] != 0 or row["data/file_index"] != 0:
            return False
        for stream in RAW_STREAMS:
            if row[f"videos/observation.images.{stream}/chunk_index"] != 0:
                return False
            if row[f"videos/observation.images.{stream}/file_index"] != 0:
                return False
        return True

    candidates = episodes_df[episodes_df.apply(_all_in_shard0, axis=1)]
    episodes = candidates.head(args.n_episodes)
    logger.info(
        f"{len(episodes_df)} episodes total; {len(candidates)} fit entirely in "
        f"shard 0; converting {len(episodes)}"
    )
    if len(episodes) == 0:
        raise RuntimeError("No episodes found fully within chunk-000/file-000 for all streams")

    logger.info("Downloading data shard (state/action/tactile_force)...")
    data_path = hf_download("data/chunk-000/file-000.parquet")
    data_df = pd.read_parquet(data_path)

    video_paths = {}
    for stream in RAW_STREAMS:
        logger.info(f"Downloading video shard for {stream}...")
        video_paths[stream] = hf_download(
            f"videos/observation.images.{stream}/chunk-000/file-000.mp4"
        )

    for _, ep in episodes.iterrows():
        ep_idx = int(ep["episode_index"])
        i0, i1 = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        rows = data_df.iloc[i0:i1]

        state = np.stack(rows["observation.state"].to_numpy()).astype(np.float32)
        action = np.stack(rows["action"].to_numpy()).astype(np.float32)
        tactile_force = np.stack(rows["observation.tactile_force"].to_numpy()).astype(np.float32)
        timestamp = rows["timestamp"].to_numpy().astype(np.float32).reshape(-1)

        per_stream_frames = []
        for stream in RAW_STREAMS:
            from_ts = float(ep[f"videos/observation.images.{stream}/from_timestamp"])
            to_ts = float(ep[f"videos/observation.images.{stream}/to_timestamp"])
            frames = load_episode_frames(video_paths[stream], from_ts, to_ts, fps)
            per_stream_frames.append(frames)

        t_min = min(f.shape[0] for f in per_stream_frames)
        if t_min == 0:
            logger.warning(f"[ep {ep_idx}] no frames decoded for one or more streams, skipping")
            continue
        tactile_raw = np.stack([f[:t_min] for f in per_stream_frames], axis=1)  # (T, 10, H, W)

        dest = out_dir / f"{ep_idx:06d}.zarr"
        if dest.exists():
            shutil.rmtree(dest)
        root = zarr.open_group(str(dest), mode="w")
        root.create_array("tactile_raw", data=np.ascontiguousarray(tactile_raw, dtype=np.uint8))
        root.create_array("tactile_force", data=tactile_force[:t_min])
        root.create_array("state", data=state[:t_min])
        root.create_array("action", data=action[:t_min])
        root.create_array("timestamp", data=timestamp[:t_min])

        root.attrs["episode_index"] = ep_idx
        root.attrs["tasks"] = list(ep["tasks"]) if ep["tasks"] is not None else []
        root.attrs["motor_primitive"] = ep.get("motor_primitive", "")
        root.attrs["object"] = ep.get("object", "")
        root.attrs["fps"] = fps
        root.attrs["tactile_raw_channel_order"] = RAW_STREAMS
        root.attrs["source"] = f"huggingface:{REPO_ID} (robot tactile, pilot pull, shard 0 only)"
        logger.info(f"[ep {ep_idx}] wrote {dest} tactile_raw={tactile_raw.shape}")


if __name__ == "__main__":
    main()
