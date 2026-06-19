"""GlobalShuffleShardDataset — sliding-window prefetch loader for GS shards.

Each shard pair ({id}.mp4, {id}.npz) was built by build_global_shuffle_shards.py.

At training time each DataLoader worker runs an independent background thread that
downloads shards from the volume to local ephemeral disk (/cache NVMe when available,
/tmp otherwise).  The thread maintains a bounded queue of `prefetch_shards` downloaded
shards ahead of the consumer.  When the consumer finishes a shard it deletes the local
files and the thread immediately starts fetching the next one — so disk usage stays at
`prefetch_shards * num_workers` shards at any moment.

Example: num_workers=6, prefetch_shards=2 → 12 shards on disk at all times.

Usage in Hydra config:
  _target_: egomimic.rldb.gs_shard_dataset.GlobalShuffleShardDataset
  shard_dir: /mnt/zarr-gs/global_shuffle_debug300
  image_size: [224, 224]
  prefetch_shards: 2
"""

from __future__ import annotations

import json
import os
import random
import shutil
import threading
from pathlib import Path
from queue import Queue
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset

_SENTINEL = object()


class GlobalShuffleShardDataset(IterableDataset):
    """Iterable dataset that streams frames from pre-built globally-shuffled shards.

    Each DataLoader worker owns a disjoint slice of shards (shuffled by worker seed).
    A background thread prefetches up to `prefetch_shards` shards onto local ephemeral
    disk while the worker iterates over the current shard.  When a shard is exhausted
    its local files are deleted and the thread's next download unblocks.

    Args:
        shard_dir:       Path to directory containing {id}.mp4, {id}.npz, index.json.
                         Typically /mnt/zarr-gs/<subdir> inside the Modal container.
        image_size:      (H, W) to resize decoded frames. None keeps native resolution.
        action_key:      Key for the action tensor in the returned sample dict.
        image_key:       Key for the image tensor in the returned sample dict.
        seed:            Base random seed for per-worker shard shuffle order.
        prefetch_shards: How many shards to keep pre-downloaded per worker.
                         Total disk usage = prefetch_shards * num_workers * shard_size.
                         Default 2 — with num_workers=6 this keeps 12 shards on disk.
    """

    def __init__(
        self,
        shard_dir: str,
        image_size: list[int] | None = None,
        action_key: str = "actions_cartesian",
        image_key: str = "observations.images.front_img_1",
        seed: int = 42,
        prefetch_shards: int = 2,
    ):
        self.shard_dir = Path(shard_dir)
        self.image_size = tuple(image_size) if image_size else None
        self.action_key = action_key
        self.image_key = image_key
        self.seed = seed
        self.prefetch_shards = prefetch_shards

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

    def __getitem__(self, index: int) -> dict:
        # Called by trainHydra.py for shape inference (dataset[0]).
        # Reads directly from the volume-mounted shard — no prefetch needed.
        first_shard = self._shard_ids[0]
        mp4_path = self.shard_dir / f"{first_shard}.mp4"
        npz_path = self.shard_dir / f"{first_shard}.npz"
        npz = np.load(str(npz_path))
        action = npz["action"][index]
        img_tensor = None
        for i, frame in enumerate(self._decode_mp4(mp4_path)):
            if i == index:
                img_tensor = frame
                break
        if img_tensor is None:
            raise IndexError(f"index {index} out of range for shard {first_shard}")
        return {
            self.image_key: img_tensor,
            self.action_key: torch.from_numpy(action).float(),
        }

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        shard_ids = list(self._shard_ids)
        rng = random.Random(self.seed + worker_id)
        rng.shuffle(shard_ids)
        my_shards = shard_ids[worker_id::num_workers]

        yield from self._iter_with_prefetch(my_shards, worker_id)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _local_dir(self) -> Path:
        """Return local ephemeral disk directory for shard copies."""
        # Prefer /cache (Modal NVMe ephemeral disk).  Fall back to TMPDIR or /tmp.
        for candidate in ["/cache", os.environ.get("TMPDIR"), "/tmp"]:
            if candidate and Path(candidate).is_dir():
                return Path(candidate)
        return Path("/tmp")

    def _download_shard(self, shard_id: str, worker_id: int) -> tuple[Path, Path]:
        """Copy one shard from the volume to local ephemeral disk."""
        dst = self._local_dir()
        local_mp4 = dst / f"gs_w{worker_id}_{shard_id}.mp4"
        local_npz = dst / f"gs_w{worker_id}_{shard_id}.npz"
        shutil.copy2(self.shard_dir / f"{shard_id}.mp4", local_mp4)
        shutil.copy2(self.shard_dir / f"{shard_id}.npz", local_npz)
        return local_mp4, local_npz

    def _iter_with_prefetch(
        self, shard_ids: list[str], worker_id: int
    ) -> Iterator[dict]:
        """Background-thread sliding-window prefetch → consume → delete loop."""
        # Bounded queue: blocks the download thread when prefetch_shards are already
        # sitting on disk waiting to be processed.
        q: Queue = Queue(maxsize=self.prefetch_shards)

        def _prefetch() -> None:
            for sid in shard_ids:
                try:
                    paths = self._download_shard(sid, worker_id)
                except Exception as exc:
                    print(
                        f"[GS worker={worker_id}] prefetch failed for {sid}: {exc}"
                    )
                    paths = None
                q.put(paths)  # blocks here when queue is full
            q.put(_SENTINEL)

        thread = threading.Thread(target=_prefetch, daemon=True)
        thread.start()

        while True:
            item = q.get()
            if item is _SENTINEL:
                break
            if item is None:
                continue  # download failed — skip shard
            local_mp4, local_npz = item
            try:
                yield from self._iter_local_shard(local_mp4, local_npz, worker_id)
            finally:
                # Delete immediately so the background thread can write the next shard.
                local_mp4.unlink(missing_ok=True)
                local_npz.unlink(missing_ok=True)

        thread.join()

    def _iter_local_shard(
        self, mp4_path: Path, npz_path: Path, worker_id: int
    ) -> Iterator[dict]:
        """Yield one frame at a time from a locally cached shard."""
        try:
            npz = np.load(str(npz_path))
            actions = npz["action"]  # (T, H, D)
            for img_tensor, action in zip(self._decode_mp4(mp4_path), actions):
                yield {
                    self.image_key: img_tensor,
                    self.action_key: torch.from_numpy(action).float(),
                }
        except Exception as exc:
            print(f"[GS worker={worker_id}] shard iteration failed: {exc}")

    def _decode_mp4(self, path: Path) -> Iterator[torch.Tensor]:
        """Decode MP4 frames lazily. Tries torchvision first, falls back to av."""
        try:
            import torchvision.io as tio

            video, _, _ = tio.read_video(str(path), output_format="TCHW", pts_unit="pts")
            for t in range(video.shape[0]):
                frame = video[t].float() / 255.0
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

    def _decode_mp4_av(self, path: Path) -> Iterator[torch.Tensor]:
        import av as _av
        import torch.nn.functional as F

        container = _av.open(str(path))
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            img = frame.to_ndarray(format="rgb24")
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
