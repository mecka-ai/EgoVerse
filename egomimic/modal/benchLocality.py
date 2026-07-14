"""Modal dataloader locality benchmark.

Runs a bounded A/B over the existing smoke dataset:
  1. baseline random batches
  2. baseline random batches with DataLoader in_order=False, if supported
  3. locality-aware batches grouped by episode shard
  4. sequential or block-shuffled batches within each episode shard

This is intentionally a dataloader-only benchmark so we can test storage access
patterns without spending GPU time on a full training job.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from modal_setup import (  # noqa: E402
    CFG,
    VOLUME_MAP,
    _prepare_repo,
    _resolve_git_state,
    app,
)


def _build_volumes() -> dict:
    # In the Modal `main` environment this volume is named with hyphens; in the
    # `robotics` environment it is named with underscores. Support both.
    vol_name = os.environ.get("MODAL_VOLUME", "mecka-data-v2")
    if vol_name in VOLUME_MAP:
        vol_obj, mount_path = VOLUME_MAP[vol_name]
    else:
        vol_obj, mount_path = modal.Volume.from_name(vol_name), CFG.volume_mount_path
    return {mount_path: vol_obj}


@app.function(
    cpu=16.0,
    memory=65536,
    timeout=3600,
    ephemeral_disk=614400,
    volumes=_build_volumes(),
)
def run_locality_benchmark(
    git_remote: str,
    git_commit: str,
    *,
    data_config: str = "mecka_all_zarr_smoke",
    dataset_name: str = "mecka_bimanual",
    max_episodes: int = 32,
    num_batches: int = 80,
    warmup_batches: int = 10,
    batch_size: int = 32,
    num_workers: int = 12,
    prefetch_factor: int = 4,
    block_size: int = 16,
    variants: str = (
        "baseline_random,baseline_random_unordered,episode_locality,"
        "episode_sequential,episode_block"
    ),
    simulated_compute_sec: float = 0.02,
    init_submodules: bool = False,
) -> dict:
    print("Preparing repo...", flush=True)
    _prepare_repo(
        git_remote=git_remote,
        git_commit=git_commit,
        init_submodules=init_submodules,
    )
    print("Repo ready; starting benchmark subprocess...", flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["HYDRA_FULL_ERROR"] = "1"
    env["MODAL_IS_REMOTE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"

    script = r'''
import gc
import json
import inspect
import os
import random
import time

import hydra
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from egomimic.pl_utils.pl_data_utils import annotation_collate
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset, ZarrDataset


class EpisodeLocalityBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, seed=42):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.groups = [
            list(indices)
            for _, indices in sorted(dataset._global_indices_by_dataset.items())
            if indices
        ]
        self.total = sum(len(g) for g in self.groups)

    def __len__(self):
        return (self.total + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = random.Random(self.seed)
        groups = [list(g) for g in self.groups]
        rng.shuffle(groups)
        for g in groups:
            rng.shuffle(g)
        # Keep a small episode-level shuffle while yielding each batch from one
        # shard where possible. This reduces cross-file seeks inside workers.
        for g in groups:
            for i in range(0, len(g), self.batch_size):
                yield g[i : i + self.batch_size]


class EpisodeSequentialBatchSampler(EpisodeLocalityBatchSampler):
    def __iter__(self):
        rng = random.Random(self.seed)
        groups = [list(g) for g in self.groups]
        rng.shuffle(groups)
        for g in groups:
            for i in range(0, len(g), self.batch_size):
                yield g[i : i + self.batch_size]


class EpisodeBlockShuffleBatchSampler(EpisodeLocalityBatchSampler):
    def __init__(self, dataset, batch_size, block_size, seed=42):
        super().__init__(dataset=dataset, batch_size=batch_size, seed=seed)
        self.block_size = max(1, int(block_size))

    def __iter__(self):
        rng = random.Random(self.seed)
        groups = [list(g) for g in self.groups]
        rng.shuffle(groups)
        for g in groups:
            blocks = [
                g[i : i + self.block_size]
                for i in range(0, len(g), self.block_size)
            ]
            rng.shuffle(blocks)
            shuffled = [idx for block in blocks for idx in block]
            for i in range(0, len(shuffled), self.batch_size):
                yield shuffled[i : i + self.batch_size]


def infer_batch_size(batch):
    if isinstance(batch, dict):
        for value in batch.values():
            if hasattr(value, "shape") and len(value.shape) > 0:
                return int(value.shape[0])
            if isinstance(value, list):
                return len(value)
    return 0


def worker_init_fn(worker_id):
    mp.set_sharing_strategy("file_system")


def build_dataset(data_config, dataset_name, max_episodes):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver(
        "multiply", lambda x, y: int(float(x)) * int(float(y)), replace=True
    )
    cfg = OmegaConf.load(
        f"/root/EgoVerse/egomimic/hydra_configs/data/{data_config}.yaml"
    )
    ds_cfg = cfg.train_datasets[dataset_name]
    key_map = hydra.utils.instantiate(ds_cfg.resolver.key_map)
    transform_list = hydra.utils.instantiate(ds_cfg.resolver.transform_list)
    folder_path = ds_cfg.resolver.folder_path
    root = os.path.abspath(str(folder_path))
    paths = []
    for name in sorted(os.listdir(root)):
        if name.startswith("."):
            continue
        path = os.path.join(root, name)
        if os.path.isdir(path):
            paths.append(path)
        if len(paths) >= max_episodes:
            break
    if not paths:
        raise RuntimeError(f"no episode directories found under {root}")
    datasets = {}
    for path in paths:
        name = os.path.basename(path)
        episode_hash = name[:-5] if name.endswith(".zarr") else name
        datasets[episode_hash] = ZarrDataset(
            path,
            key_map=key_map,
            transform_list=transform_list,
            norm_stats=None,
        )
    return MultiDataset(datasets=datasets, mode="total")


def make_loader(dataset, batch_sampler, num_workers, prefetch_factor, in_order=True):
    kwargs = {
        "dataset": dataset,
        "batch_sampler": batch_sampler,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor if num_workers else None,
        "collate_fn": annotation_collate,
        "worker_init_fn": worker_init_fn,
        "persistent_workers": False,
    }
    if "in_order" in inspect.signature(DataLoader).parameters:
        kwargs["in_order"] = in_order
    return DataLoader(**kwargs)


def run_loader(name, loader, warmup_batches, num_batches, simulated_compute_sec):
    it = iter(loader)
    for _ in range(warmup_batches):
        try:
            next(it)
        except StopIteration:
            it = iter(loader)
            next(it)

    samples = 0
    load_sec = 0.0
    t0 = time.perf_counter()
    for _ in range(num_batches):
        b0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        load_sec += time.perf_counter() - b0
        samples += infer_batch_size(batch)
        if simulated_compute_sec:
            time.sleep(simulated_compute_sec)
    total_sec = time.perf_counter() - t0
    return {
        "name": name,
        "samples": samples,
        "total_sec": total_sec,
        "load_sec": load_sec,
        "samples_per_sec": samples / total_sec if total_sec else 0.0,
        "avg_load_ms_per_batch": 1000.0 * load_sec / max(num_batches, 1),
        "avg_total_ms_per_batch": 1000.0 * total_sec / max(num_batches, 1),
    }


data_config = os.environ["BENCH_DATA_CONFIG"]
dataset_name = os.environ["BENCH_DATASET_NAME"]
max_episodes = int(os.environ["BENCH_MAX_EPISODES"])
batch_size = int(os.environ["BENCH_BATCH_SIZE"])
num_workers = int(os.environ["BENCH_NUM_WORKERS"])
prefetch_factor = int(os.environ["BENCH_PREFETCH_FACTOR"])
block_size = int(os.environ["BENCH_BLOCK_SIZE"])
num_batches = int(os.environ["BENCH_NUM_BATCHES"])
warmup_batches = int(os.environ["BENCH_WARMUP_BATCHES"])
simulated_compute_sec = float(os.environ["BENCH_SIMULATED_COMPUTE_SEC"])
variants = [
    v.strip()
    for v in os.environ["BENCH_VARIANTS"].split(",")
    if v.strip()
]

os.makedirs("/cache/torch_ipc", exist_ok=True)
os.environ["TMPDIR"] = "/cache/torch_ipc"
mp.set_sharing_strategy("file_system")
print(f"sharing_strategy={mp.get_sharing_strategy()} TMPDIR={os.environ['TMPDIR']}", flush=True)

dataset = build_dataset(data_config, dataset_name, max_episodes)
print(
    f"dataset={data_config}.{dataset_name} frames={len(dataset)} "
    f"episodes={len(dataset.datasets)} batch_size={batch_size} workers={num_workers}",
    flush=True,
)

samplers = {
    "baseline_random": BatchSampler(
        RandomSampler(dataset),
        batch_size=batch_size,
        drop_last=False,
    ),
    "baseline_random_unordered": BatchSampler(
        RandomSampler(dataset),
        batch_size=batch_size,
        drop_last=False,
    ),
    "episode_locality": EpisodeLocalityBatchSampler(dataset, batch_size=batch_size),
    "episode_sequential": EpisodeSequentialBatchSampler(dataset, batch_size=batch_size),
    "episode_block": EpisodeBlockShuffleBatchSampler(
        dataset,
        batch_size=batch_size,
        block_size=block_size,
    ),
}

results = []
for variant in variants:
    if variant not in samplers:
        raise ValueError(f"unknown variant {variant!r}; choices={sorted(samplers)}")
    print(f"starting_variant={variant}", flush=True)
    loader = make_loader(
        dataset=dataset,
        batch_sampler=samplers[variant],
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        in_order=variant != "baseline_random_unordered",
    )
    results.append(
        run_loader(
            variant,
            loader,
            warmup_batches,
            num_batches,
            simulated_compute_sec,
        )
    )
    print(f"finished_variant={variant}", flush=True)
    del loader
    gc.collect()
payload = {"results": results}
print("RESULT_JSON=" + json.dumps(payload, sort_keys=True), flush=True)
with open(os.environ["BENCH_RESULT_PATH"], "w") as f:
    json.dump(payload, f)
'''

    result_path = "/tmp/egoverse_locality_bench_result.json"
    env.update(
        {
            "BENCH_DATA_CONFIG": data_config,
            "BENCH_DATASET_NAME": dataset_name,
            "BENCH_MAX_EPISODES": str(max_episodes),
            "BENCH_NUM_BATCHES": str(num_batches),
            "BENCH_WARMUP_BATCHES": str(warmup_batches),
            "BENCH_BATCH_SIZE": str(batch_size),
            "BENCH_NUM_WORKERS": str(num_workers),
            "BENCH_PREFETCH_FACTOR": str(prefetch_factor),
            "BENCH_BLOCK_SIZE": str(block_size),
            "BENCH_VARIANTS": variants,
            "BENCH_SIMULATED_COMPUTE_SEC": str(simulated_compute_sec),
            "BENCH_RESULT_PATH": result_path,
        }
    )
    proc = subprocess.run(
        [CFG.python_bin, "-c", script],
        cwd=CFG.remote_repo_dir,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"benchmark failed with exit {proc.returncode}")
    with open(result_path) as f:
        return json.load(f)


@app.local_entrypoint()
def main(
    data_config: str = "mecka_all_zarr_smoke",
    dataset_name: str = "mecka_bimanual",
    max_episodes: int = 32,
    num_batches: int = 80,
    warmup_batches: int = 10,
    batch_size: int = 32,
    num_workers: int = 12,
    prefetch_factor: int = 4,
    block_size: int = 16,
    variants: str = (
        "baseline_random,baseline_random_unordered,episode_locality,"
        "episode_sequential,episode_block"
    ),
    simulated_compute_sec: float = 0.02,
    modal_volume: str = "mecka-data-v2",
) -> None:
    os.environ["MODAL_VOLUME"] = modal_volume
    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print("Warning: local repo has uncommitted changes; benchmark clones HEAD.")
    result = run_locality_benchmark.remote(
        git_remote,
        git_commit,
        data_config=data_config,
        dataset_name=dataset_name,
        max_episodes=max_episodes,
        num_batches=num_batches,
        warmup_batches=warmup_batches,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        block_size=block_size,
        variants=variants,
        simulated_compute_sec=simulated_compute_sec,
        init_submodules=False,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
