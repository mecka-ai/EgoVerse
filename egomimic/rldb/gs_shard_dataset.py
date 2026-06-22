"""GlobalShuffleShardDataset — rank-level shared downloader pool.

Architecture
------------
Old: each of N DataLoader workers has its own background thread downloading shards.
     num_workers × world_size = 6 × 4 = 24 concurrent volume readers.

New: one downloader THREAD (inside worker 0's process) per rank maintains a shared
     pool of ready shards on local disk. Workers never touch the volume directly.
     Total concurrent volume readers = world_size (one per rank).

     Note: we use a thread rather than a process because PyTorch DataLoader workers
     are daemon processes, and Python forbids daemon processes from spawning children.
     A thread inside worker 0 produces into a multiprocessing.Queue; all worker
     processes (0-N) consume from it — this works fine across process boundaries.

Flow per rank per epoch:
  1. DataLoader worker 0 spawns a background thread that downloads the rank's shard
     slice: shard_ids[rank::world_size].
  2. Downloader uses n_download_threads concurrent copies to pipeline volume → disk.
     A multiprocessing.Queue(maxsize=pool_size) provides backpressure: downloader
     blocks when pool_size shards are already downloaded and waiting.
  3. All workers pull (mp4_path, npz_path) from the shared queue, iterate all frames
     of that shard, delete the files, then pull the next shard.
  4. When all shards are sent the downloader puts one None sentinel per worker.
     Workers break on None and the epoch ends cleanly.

Disk usage: at most (pool_size + n_download_threads) shards per rank at any time.
  Default pool_size=12, n_download_threads=4 → 16 shards × 34 MB ≈ 550 MB per rank.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import threading
import warnings
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset

# ---------------------------------------------------------------------------
# Module-level downloader (must be picklable — no lambda / closure)
# ---------------------------------------------------------------------------

def _run_downloader(
    shard_dir_str: str,
    shard_ids: list[str],
    path_q: "mp.Queue",
    num_workers: int,
    n_threads: int,
    local_dir_str: str,
    rank: int,
) -> None:
    """Runs in a dedicated process. Copies shards volume → local disk."""
    import concurrent.futures
    import shutil
    from pathlib import Path

    shard_dir = Path(shard_dir_str)
    local_dir = Path(local_dir_str)

    def copy_one(sid: str):
        dst_mp4 = local_dir / f"gsdl_r{rank}_{sid}.mp4"
        dst_npz = local_dir / f"gsdl_r{rank}_{sid}.npz"
        shutil.copy2(shard_dir / f"{sid}.mp4", dst_mp4)
        shutil.copy2(shard_dir / f"{sid}.npz", dst_npz)
        return dst_mp4, dst_npz

    # Sliding window of n_threads concurrent downloads.
    # path_q.put() blocks when the queue is full → natural backpressure.
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as pool:
        pending = list(shard_ids)
        window: list[tuple[str, "concurrent.futures.Future"]] = []
        idx = 0

        def _submit_next():
            nonlocal idx
            if idx < len(pending):
                sid = pending[idx]
                window.append((sid, pool.submit(copy_one, sid)))
                idx += 1

        # Prime the pipeline
        for _ in range(min(n_threads, len(pending))):
            _submit_next()

        while window:
            sid, fut = window.pop(0)
            try:
                paths = fut.result()
                _submit_next()       # overlap: start next download while we put
                path_q.put(paths)    # blocks when queue full
            except Exception as exc:
                print(f"[GS downloader rank={rank}] {sid}: {exc}", flush=True)
                _submit_next()

    # One sentinel per DataLoader worker so each breaks cleanly
    for _ in range(num_workers):
        path_q.put(None)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GlobalShuffleShardDataset(IterableDataset):
    """Iterable dataset that streams pre-built globally-shuffled shards.

    One downloader process per rank keeps a local pool of ready shards.
    All DataLoader workers in the rank consume from that shared pool.

    Args:
        shard_dir:          Path to directory with {id}.mp4, {id}.npz, index.json.
        image_size:         (H, W) to resize frames. None = native resolution.
        action_key:         Key for the action tensor in returned sample dicts.
        image_key:          Key for the image tensor in returned sample dicts.
        pool_size:          Max shards buffered on local disk per rank at once.
                            Downloader blocks when pool is full.
        n_download_threads: Concurrent copy threads in the downloader process.
    """

    def __init__(
        self,
        shard_dir: str,
        image_size: list[int] | None = None,
        action_key: str = "actions_cartesian",
        image_key: str = "observations.images.front_img_1",
        pool_size: int = 12,
        n_download_threads: int = 4,
    ):
        self.shard_dir = Path(shard_dir)
        self.image_size = tuple(image_size) if image_size else None
        self.action_key = action_key
        self.image_key = image_key
        self.pool_size = pool_size
        self.n_download_threads = n_download_threads

        index_path = self.shard_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"index.json not found in {shard_dir}. "
                "Run build_global_shuffle_shards.py first."
            )
        index = json.loads(index_path.read_text())
        all_ids: list[str] = index.get("shard_ids", [])
        self._shard_ids: list[str] = [
            sid for sid in all_ids
            if (self.shard_dir / f"{sid}.mp4").exists()
            and (self.shard_dir / f"{sid}.npz").exists()
        ]
        if not self._shard_ids:
            raise ValueError(f"No valid shard files found in {shard_dir}")
        if len(self._shard_ids) < len(all_ids):
            warnings.warn(
                f"[GlobalShuffleShardDataset] {len(all_ids) - len(self._shard_ids)} shards "
                f"listed in index.json are missing on disk and will be skipped "
                f"({len(self._shard_ids)}/{len(all_ids)} available)"
            )

        self._frames_per_shard: int = index.get("frames_per_shard", 2000)

        # Shared multiprocessing Queue — survives fork into DataLoader workers.
        # Created here (main process) so it's inherited by all worker forks.
        ctx = mp.get_context("fork")
        self._path_q: "mp.Queue" = ctx.Queue(maxsize=pool_size)

    # ------------------------------------------------------------------
    # PyTorch / trainHydra interface
    # ------------------------------------------------------------------

    def set_data_schematic(self, data_schematic, bounds_slack: float = 0.0) -> None:
        self.data_schematic = data_schematic
        self.bounds_slack = bounds_slack

    def __len__(self) -> int:
        # Return the frames this rank will actually yield in one epoch.
        # Shards are split across ranks (shard_ids[rank::world_size]), so each
        # rank owns 1/world_size of the shards. If DDP isn't initialised yet
        # (e.g. trainHydra frame-count display), fall back to total frames.
        world_size = self._get_world_size()
        rank_shard_count = len(self._shard_ids) // world_size
        return rank_shard_count * self._frames_per_shard

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

        # Worker 0 spawns the rank-level downloader process for this epoch.
        # Other workers simply block on path_q.get() until data arrives.
        if worker_id == 0:
            rank = self._get_rank()
            world_size = self._get_world_size()
            rank_shards = self._shard_ids[rank::world_size]
            local_dir = self._local_dir()
            t = threading.Thread(
                target=_run_downloader,
                args=(
                    str(self.shard_dir),
                    rank_shards,
                    self._path_q,
                    num_workers,
                    self.n_download_threads,
                    str(local_dir),
                    rank,
                ),
                daemon=True,
            )
            t.start()

        # All workers consume from the shared pool
        while True:
            item = self._path_q.get()
            if item is None:
                break
            local_mp4, local_npz = Path(item[0]), Path(item[1])
            try:
                yield from self._iter_local_shard(local_mp4, local_npz, worker_id)
            finally:
                local_mp4.unlink(missing_ok=True)
                local_npz.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _local_dir(self) -> Path:
        for candidate in ["/cache", os.environ.get("TMPDIR"), "/tmp"]:
            if candidate and Path(candidate).is_dir():
                return Path(candidate)
        return Path("/tmp")

    @staticmethod
    def _get_rank() -> int:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank()
        except Exception:
            pass
        return 0

    @staticmethod
    def _get_world_size() -> int:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_world_size()
        except Exception:
            pass
        return 1

    def _iter_local_shard(
        self, mp4_path: Path, npz_path: Path, worker_id: int
    ) -> Iterator[dict]:
        try:
            npz = np.load(str(npz_path))
            actions = npz["action"]
            for img_tensor, action in zip(self._decode_mp4(mp4_path), actions):
                yield {
                    self.image_key: img_tensor,
                    self.action_key: torch.from_numpy(action).float(),
                }
        except Exception as exc:
            print(f"[GS worker={worker_id}] shard iteration failed: {exc}", flush=True)

    def _decode_mp4(self, path: Path) -> Iterator[torch.Tensor]:
        try:
            import torchvision.io as tio
            video, _, _ = tio.read_video(str(path), output_format="TCHW", pts_unit="sec")
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
