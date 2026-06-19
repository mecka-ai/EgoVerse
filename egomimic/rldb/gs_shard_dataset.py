"""GlobalShuffleShardDataset — loads pre-built globally-shuffled shards for training.

Each shard pair ({id}.mp4, {id}.npz) was built by build_global_shuffle_shards.py:
  - MP4: H.264 camera frames (single front camera), +faststart, no B-frames, fixed GOP.
    Frames are decoded lazily via torchcodec for fast local reads.
  - NPZ: pre-transformed action arrays (T, horizon, action_dim).

At training time, shards are loaded from a locally-mounted volume or copied to
ephemeral disk per worker. This replaces the network-volume random-seek pattern
of the zarr dataloader with sequential local reads.

Usage in Hydra config:
  _target_: egomimic.rldb.gs_shard_dataset.GlobalShuffleShardDataset
  shard_dir: /mnt/zarr-gs/global_shuffle_v1
  image_size: [224, 224]
  action_key: actions_cartesian
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset


class GlobalShuffleShardDataset(IterableDataset):
    """Iterable dataset that streams frames from pre-built globally-shuffled MP4+npz shards.

    Each worker loads one shard at a time and iterates over its frames in order.
    The shard list is shuffled at the start of each epoch (based on worker seed).

    The MP4 is decoded via torchcodec (fast, GPU-optional). Falls back to av
    if torchcodec is not installed.

    Args:
        shard_dir:    Directory containing {shard_id}.mp4 and {shard_id}.npz files,
                      plus index.json produced by build_global_shuffle_shards.py.
        image_size:   (H, W) to resize decoded frames. None keeps native resolution.
        action_key:   Key under which to return the action array (matches training schema).
        image_key:    Key under which to return the image tensor.
        seed:         Base random seed for shard shuffle order.
        copy_to_tmp:  If True, copy each shard to /tmp before reading (avoids
                      network-volume seek cost; requires ephemeral disk space).
    """

    def __init__(
        self,
        shard_dir: str,
        image_size: list[int] | None = None,
        action_key: str = "actions_cartesian",
        image_key: str = "observations.images.front_img_1",
        seed: int = 42,
        copy_to_tmp: bool = False,
    ):
        self.shard_dir = Path(shard_dir)
        self.image_size = tuple(image_size) if image_size else None
        self.action_key = action_key
        self.image_key = image_key
        self.seed = seed
        self.copy_to_tmp = copy_to_tmp

        index_path = self.shard_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"index.json not found in {shard_dir}. "
                "Run build_global_shuffle_shards.py first."
            )
        index = json.loads(index_path.read_text())
        self._shard_ids: list[str] = index.get("shard_ids", [])
        if not self._shard_ids:
            raise ValueError(f"No shards listed in {index_path}")

    def __len__(self) -> int:
        index = json.loads((self.shard_dir / "index.json").read_text())
        return index.get("n_covered_frames", len(self._shard_ids) * 2000)

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # Each worker handles a disjoint slice of shards
        shard_ids = list(self._shard_ids)
        rng = random.Random(self.seed + worker_id)
        rng.shuffle(shard_ids)
        my_shards = shard_ids[worker_id::num_workers]

        for shard_id in my_shards:
            try:
                yield from self._iter_shard(shard_id)
            except Exception as e:
                print(f"[GSShardDataset worker={worker_id}] shard {shard_id} failed: {e}")
                continue

    def _iter_shard(self, shard_id: str) -> Iterator[dict]:
        mp4_path = self.shard_dir / f"{shard_id}.mp4"
        npz_path = self.shard_dir / f"{shard_id}.npz"

        if not mp4_path.exists() or not npz_path.exists():
            return

        local_mp4 = mp4_path
        local_npz = npz_path

        if self.copy_to_tmp:
            import shutil

            # Prefer TMPDIR (set to /cache NVMe by trainModal when ephemeral disk is
            # provisioned) to avoid filling the in-memory /tmp tmpfs.
            tmp_dir = Path(os.environ.get("TMPDIR", "/tmp"))
            local_mp4 = tmp_dir / f"gs_shard_{shard_id}.mp4"
            local_npz = tmp_dir / f"gs_shard_{shard_id}.npz"
            shutil.copy2(mp4_path, local_mp4)
            shutil.copy2(npz_path, local_npz)

        try:
            npz = np.load(str(local_npz))
            actions = npz["action"]  # (T, H, D)

            # Lazy decode — one frame at a time so no full-shard float32 array is
            # ever allocated in the worker process.
            for img_tensor, action in zip(self._decode_mp4(local_mp4), actions):
                yield {
                    self.image_key: img_tensor,
                    self.action_key: torch.from_numpy(action).float(),
                }
        finally:
            if self.copy_to_tmp:
                local_mp4.unlink(missing_ok=True)
                local_npz.unlink(missing_ok=True)

    def _decode_mp4(self, path: Path):
        """Decode frames lazily from an MP4. Tries torchvision first, falls back to av."""
        try:
            import torchvision.io as tio

            video, _, _ = tio.read_video(str(path), output_format="TCHW", pts_unit="pts")
            # video: (T, C, H, W) uint8
            for t in range(video.shape[0]):
                frame = video[t].float() / 255.0  # (C, H, W)
                if self.image_size:
                    import torch.nn.functional as F
                    frame = F.interpolate(
                        frame.unsqueeze(0),
                        size=self.image_size,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                yield frame
        except Exception:
            yield from self._decode_mp4_av(path)

    def _decode_mp4_av(self, path: Path):
        """av fallback decoder."""
        import av as _av
        import torch.nn.functional as F

        container = _av.open(str(path))
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            img = frame.to_ndarray(format="rgb24")  # (H, W, 3)
            tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            if self.image_size:
                tensor = F.interpolate(
                    tensor.unsqueeze(0),
                    size=self.image_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            yield tensor
        container.close()
