"""Dataloader-only benchmark for Modal storage A/B tests.

This intentionally follows the dataset/datamodule construction path in
``trainHydra.py`` but stops before model/trainer setup so timings isolate data
loading instead of Lightning startup or GPU compute.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from egomimic.rldb.zarr.utils import DataSchematic, set_global_seed
from egomimic.utils.aws.aws_data_utils import load_env
from egomimic.utils.dataloader_ipc import (
    apply_ipc_dataloader_params,
    configure_dataloader_ipc,
)


OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("multiply", lambda x, y: int(float(x)) * int(float(y)))


def _batch_size(batch: Any) -> int:
    if isinstance(batch, dict):
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return int(value.shape[0])
            if isinstance(value, list):
                return len(value)
            if isinstance(value, dict):
                nested = _batch_size(value)
                if nested:
                    return nested
    if isinstance(batch, (list, tuple)) and batch:
        return _batch_size(batch[0])
    return 0


def _propagate_data_schematic(data_schematic: DataSchematic, datasets: dict, slack: float) -> None:
    for dataset_name, dataset in datasets.items():
        if not hasattr(dataset, "set_data_schematic"):
            raise ValueError(f"{dataset_name} does not implement set_data_schematic()")
        dataset.set_data_schematic(data_schematic, bounds_slack=slack)


@hydra.main(version_base="1.3", config_path="../hydra_configs", config_name="train_zarr_cartesian.yaml")
def main(cfg: DictConfig) -> None:
    configure_dataloader_ipc()
    if cfg.get("seed"):
        set_global_seed(cfg.seed)
    load_env()

    max_batches = int(OmegaConf.select(cfg, "bench.max_batches", default=50))
    warmup_batches = int(OmegaConf.select(cfg, "bench.warmup_batches", default=1))
    dataset_name = OmegaConf.select(cfg, "bench.dataset_name", default=None)

    data_schematic: DataSchematic = hydra.utils.instantiate(cfg.data_schematic)
    train_datasets = {
        name: hydra.utils.instantiate(dataset_cfg)
        for name, dataset_cfg in cfg.data.train_datasets.items()
        if dataset_cfg is not None
    }
    valid_datasets = {
        name: hydra.utils.instantiate(dataset_cfg)
        for name, dataset_cfg in cfg.data.valid_datasets.items()
        if dataset_cfg is not None
    }
    datamodule = hydra.utils.instantiate(
        cfg.data,
        train_datasets=train_datasets,
        valid_datasets=valid_datasets,
        train_viz_datasets={},
    )

    if not datamodule.train_datasets:
        raise ValueError("No train datasets configured")
    if dataset_name is None:
        dataset_name = next(iter(datamodule.train_datasets))
    dataset = datamodule.train_datasets[dataset_name]

    t0 = time.perf_counter()
    data_schematic.infer_shapes_from_batch(dataset[0])
    print(f"DATALOADER_BENCH shape_infer_s={time.perf_counter() - t0:.6f}", flush=True)

    instantiate_copy = copy.deepcopy(cfg.data.train_datasets[dataset_name])
    keymap_cfg = instantiate_copy.resolver.key_map
    keymap = OmegaConf.to_container(keymap_cfg, resolve=False)
    keymap["norm_mode"] = True
    instantiate_copy.resolver.key_map = keymap
    norm_dataset = hydra.utils.instantiate(instantiate_copy)
    t0 = time.perf_counter()
    data_schematic.infer_norm_from_dataset(
        norm_dataset,
        dataset_name,
        sample_frac=OmegaConf.select(cfg, "norm_stats.sample_frac", default=1.0),
        num_workers=OmegaConf.select(cfg, "norm_stats.num_workers", default=4),
        precomputed_norm_path=OmegaConf.select(cfg, "norm_stats.precomputed_norm_path", default=None),
    )
    print(f"DATALOADER_BENCH norm_stats_s={time.perf_counter() - t0:.6f}", flush=True)

    if cfg.reject_outliers:
        slack = float(OmegaConf.select(cfg, "reject_outliers_slack", default=0.0))
        _propagate_data_schematic(data_schematic, datamodule.train_datasets, slack)

    if hasattr(dataset, "prepare_epoch"):
        t0 = time.perf_counter()
        dataset.prepare_epoch(0)
        print(f"DATALOADER_BENCH prepare_epoch_s={time.perf_counter() - t0:.6f}", flush=True)

    dataset_params = dict(datamodule.train_dataloader_params[dataset_name])
    is_iterable = isinstance(dataset, torch.utils.data.IterableDataset)
    shuffle = dataset_params.pop("shuffle", not is_iterable)
    if is_iterable:
        shuffle = False
    loader = DataLoader(
        dataset,
        shuffle=shuffle,
        collate_fn=datamodule.collate_fn,
        **apply_ipc_dataloader_params(dataset_params),
    )

    waits: list[float] = []
    iterator = iter(loader)
    for batch_idx in range(max_batches):
        t0 = time.perf_counter()
        batch = next(iterator)
        wait_s = time.perf_counter() - t0
        waits.append(wait_s)
        print(
            "DATALOADER_BENCH "
            f"batch={batch_idx} wait_s={wait_s:.6f} batch_size={_batch_size(batch)}",
            flush=True,
        )
        del batch

    measured = waits[warmup_batches:] if warmup_batches < len(waits) else waits
    avg_s = sum(measured) / len(measured) if measured else 0.0
    print(
        "DATALOADER_BENCH_SUMMARY "
        f"batches={len(waits)} warmup={warmup_batches} avg_wait_s={avg_s:.6f} "
        f"max_wait_s={max(measured) if measured else 0.0:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
