import os
import tempfile as _tempfile

# Must be before ANY torch import — torch seeds tempfile's internal cache on first use.
os.environ["TMPDIR"] = "/cache"
_tempfile.tempdir = "/cache"

import copy
import signal
import time
from typing import Any, Dict, List, Optional, Tuple

import torch.multiprocessing as _mp
_mp.set_sharing_strategy("file_system")

import hydra
import lightning as L
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf, open_dict
from tabulate import tabulate

from egomimic.eval.eval import Eval
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.zarr.utils import DataSchematic, set_global_seed
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.utils.aws.aws_data_utils import load_env
from egomimic.utils.dataloader_ipc import configure_dataloader_ipc
from egomimic.utils.instantiators import instantiate_callbacks, instantiate_loggers
from egomimic.utils.logging_utils import log_hyperparameters
from egomimic.utils.pylogger import RankedLogger
from egomimic.utils.utils import extras, task_wrapper

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("multiply", lambda x, y: int(float(x)) * int(float(y)))
log = RankedLogger(__name__, rank_zero_only=True)


def _build_model_config_tree(cfg: DictConfig) -> DictConfig:
    model_cfg = copy.deepcopy(cfg.model)
    if (
        "robomimic_model" in model_cfg
        and isinstance(model_cfg.robomimic_model, DictConfig)
        and "data_schematic" in model_cfg.robomimic_model
    ):
        model_cfg.robomimic_model.data_schematic = None
    return OmegaConf.create({"model": model_cfg})


def _log_dataset_frame_counts(
    train_datasets: dict,
    valid_datasets: dict,
    train_viz_datasets: dict | None = None,
) -> None:
    rows = []
    for name, ds in train_datasets.items():
        rows.append(("train", name, len(ds)))
    if train_datasets:
        rows.append(
            ("TOTAL", "(train)", sum(len(ds) for ds in train_datasets.values()))
        )
    for name, ds in valid_datasets.items():
        rows.append(("valid", name, len(ds)))
    if valid_datasets:
        rows.append(
            ("TOTAL", "(valid)", sum(len(ds) for ds in valid_datasets.values()))
        )
    if train_viz_datasets:
        for name, ds in train_viz_datasets.items():
            rows.append(("train_viz", name, len(ds)))
        rows.append(
            (
                "TOTAL",
                "(train_viz)",
                sum(len(ds) for ds in train_viz_datasets.values()),
            )
        )
    table = tabulate(
        rows,
        headers=["Split", "Dataset", "Frames"],
        tablefmt="rounded_outline",
        intfmt=",",
    )
    log.info("Dataset frame counts:\n" + table)


def _propagate_data_schematic_to_datasets(data_schematic, datasets, bounds_slack: float = 0.0):
    """
    Set the shared data schematic on all top-level datasets.
    """
    for dataset_name, dataset in datasets.items():
        if not hasattr(dataset, "set_data_schematic"):
            raise ValueError(
                f"{dataset_name} does not implement set_data_schematic(). "
                "Dataset must be a MultiDataset or PrefetchedIterableDataset."
            )
        dataset.set_data_schematic(data_schematic, bounds_slack=bounds_slack)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # DDP rank processes are spawned after this call; IPC is configured again in
    # MultiDataModuleWrapper.train_dataloader() where DataLoaders are built.
    configure_dataloader_ipc()

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

        set_global_seed(cfg.seed)
    else:
        raise ValueError("Seed must be provided in cfg for reproducibility!")

    load_env()
    # log.info(f"Instantiating data schematic <{cfg.data_schematic._target_}>")

    data_schematic: DataSchematic = hydra.utils.instantiate(cfg.data_schematic)

    # Modify dataset configs to include `data_schematic` dynamically at runtime
    train_datasets = {}
    for dataset_name in cfg.data.train_datasets:
        train_datasets[dataset_name] = hydra.utils.instantiate(
            cfg.data.train_datasets[dataset_name]
        )

    valid_datasets = {}
    for dataset_name in cfg.data.valid_datasets:
        valid_datasets[dataset_name] = hydra.utils.instantiate(
            cfg.data.valid_datasets[dataset_name]
        )

    train_viz_datasets = {}
    train_viz_cfg = OmegaConf.select(cfg, "data.train_viz_datasets", default=None)
    if train_viz_cfg:
        for dataset_name in train_viz_cfg:
            entry = train_viz_cfg[dataset_name]
            if entry is None:
                continue
            train_viz_datasets[dataset_name] = hydra.utils.instantiate(entry)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    assert (
        "MultiDataModuleWrapper" in cfg.data._target_
    ), "cfg.data._target_ must be 'MultiDataModuleWrapper'"
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data,
        train_datasets=train_datasets,
        valid_datasets=valid_datasets,
        train_viz_datasets=train_viz_datasets,
    )

    for dataset_name, dataset in datamodule.train_datasets.items():
        log.info(f"Inferring shapes for dataset <{dataset_name}>")
        t_shape = time.perf_counter()
        data_schematic.infer_shapes_from_batch(dataset[0])
        log.info(f"[Timing] Shape inference for {dataset_name}: {time.perf_counter() - t_shape:.2f}s")

        instantiate_copy = copy.deepcopy(cfg.data.train_datasets[dataset_name])
        keymap_cfg = instantiate_copy.resolver.key_map
        km = OmegaConf.to_container(keymap_cfg, resolve=False)  # plain dict

        # this remove annotation and image keys from the keymap
        km["norm_mode"] = True

        instantiate_copy.resolver.key_map = km
        norm_dataset = hydra.utils.instantiate(instantiate_copy)
        # infer_norm_from_dataset: load from precomputed JSON/dir if set, else compute (no disk write).
        t_norm = time.perf_counter()
        data_schematic.infer_norm_from_dataset(
            norm_dataset,
            dataset_name,
            sample_frac=OmegaConf.select(cfg, "norm_stats.sample_frac", default=1.0),
            num_workers=OmegaConf.select(cfg, "norm_stats.num_workers", default=4),
            precomputed_norm_path=OmegaConf.select(
                cfg, "norm_stats.precomputed_norm_path", default=None
            ),
        )
        log.info(f"[Timing] Norm stats for {dataset_name}: {time.perf_counter() - t_norm:.2f}s")
        # Cache norm stats if save_cache_dir is set
        save_cache_dir = OmegaConf.select(
            cfg, "norm_stats.save_cache_dir", default=None
        )
        if save_cache_dir:
            data_schematic.cache_stats(save_cache_dir=save_cache_dir)

    if cfg.reject_outliers:
        # Propagate the shared data schematic to top-level MultiDatasets for bounds checks.
        # Use datamodule.train_datasets (null entries already filtered by the wrapper).
        bounds_slack = float(OmegaConf.select(cfg, "reject_outliers_slack", default=0.0))
        _propagate_data_schematic_to_datasets(
            data_schematic,
            datamodule.train_datasets,
            bounds_slack=bounds_slack,
        )
    viz_func = cfg.visualization
    viz_func_dict = {}
    for embodiment_name, embodiment_viz_func in viz_func.items():
        viz_func_dict[embodiment_name] = hydra.utils.instantiate(embodiment_viz_func)

    viz_func_dict = {}
    for embodiment_name, embodiment_viz_func in cfg.visualization.items():
        viz_func_dict[embodiment_name] = hydra.utils.instantiate(embodiment_viz_func)

    # NOTE: We also pass the data_schematic_dict into the robomimic model's instatiation now that we've initialzied the shapes and norm stats.  In theory, upon loading the PL checkpoint, it will remember this, but let's see.
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = ModelWrapper(
        config_tree=_build_model_config_tree(cfg),
        data_schematic_state=data_schematic.to_state(),
        viz_func=viz_func_dict,
        scheduler_interval=cfg.model.get("scheduler_interval", "step"),
    )

    _log_dataset_frame_counts(
        datamodule.train_datasets,
        datamodule.valid_datasets,
        getattr(datamodule, "train_viz_datasets", None),
    )

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    if os.environ.get("MODAL_IS_REMOTE") == "1" and os.environ.get("MODAL_TIMEOUT_SECONDS"):
        from egomimic.modal.callbacks import ModalAutoRestartCallback
        callbacks.append(ModalAutoRestartCallback())
        log.info("[ModalAutoRestart] Callback registered")

    if any(
        hasattr(ds, "prepare_epoch")
        for ds in (*datamodule.train_datasets.values(), *datamodule.valid_datasets.values())
    ):
        from egomimic.modal.callbacks import PrefetchEpochCallback
        callbacks.append(
            PrefetchEpochCallback(datamodule.train_datasets, datamodule.valid_datasets)
        )
        log.info("[PrefetchEpoch] Callback registered (train + valid)")

    # Resolve mode: support both new `mode` key and legacy `train`/`eval` booleans
    if cfg.get("mode") is not None:
        mode = cfg.mode
    elif cfg.get("train", False):
        mode = "train"
    elif cfg.get("eval", False):
        mode = "eval"
    else:
        raise ValueError("Config must specify either `mode` or `train`/`eval` booleans")

    # In eval mode, apply trainer overrides from the eval object and disable logger
    if mode == "eval":
        eval_obj: Eval = hydra.utils.instantiate(cfg.evaluator)
        log.info(
            "Eval mode: applying trainer overrides from eval config, disabling logger"
        )
        with open_dict(cfg):
            for k, v in eval_obj.override_dict.items():
                cfg.trainer[k] = v
            cfg.trainer.devices = 1
            cfg.trainer.num_nodes = 1
            cfg.trainer.num_sanity_val_steps = 0
            cfg.logger = None

    # Configure WandB to resume an existing run when wandb_run_id is provided
    if OmegaConf.select(cfg, "wandb_run_id") and OmegaConf.select(cfg, "logger.wandb"):
        with open_dict(cfg):
            cfg.logger.wandb.id = str(cfg.wandb_run_id)
            cfg.logger.wandb["resume"] = "allow"
        log.info(f"[WandB] Resuming run id={cfg.wandb_run_id}")

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    plugins = []
    if os.environ.get("SLURM_JOB_ID"):
        plugins.append(
            SLURMEnvironment(requeue_signal=[signal.SIGUSR1, signal.SIGUSR2])
        )
        print("SLURM REQUEUE ENABLED")
    # Strip Modal-only sentinel key before passing to Lightning Trainer
    with open_dict(cfg):
        cfg.trainer.pop("_modal", None)
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if (
        os.environ.get("SLURM_JOB_ID")
        and os.environ.get("SLURM_RESTART_COUNT", "0") != "0"
    ):
        last_ckpt_path = os.path.join(
            trainer.default_root_dir, "checkpoints", "last.ckpt"
        )
        log.info("Detected SLURM requeue — resuming from 'last.ckpt'")
        cfg.ckpt_path = last_ckpt_path

    os.makedirs(os.path.join(trainer.default_root_dir, "videos"), exist_ok=True)

    if mode == "train":
        if cfg.get("evaluator") is not None:
            eval_obj: Eval = hydra.utils.instantiate(cfg.evaluator)
            eval_obj.trainer = trainer
            eval_obj.model = model.model
            model.evaluator = eval_obj
            if OmegaConf.select(cfg, "train_viz_evaluator", default=None) is not None:
                train_viz_eval: Eval = hydra.utils.instantiate(cfg.train_viz_evaluator)
                train_viz_eval.trainer = trainer
                train_viz_eval.model = model.model
                model.train_viz_evaluator = train_viz_eval
                log.info(
                    "train_viz_evaluator configured — wired to dataloader_idx=1"
                )
        log.info("Starting training!")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
            weights_only=False,
        )
    elif mode == "eval":
        eval_obj.trainer = trainer
        eval_obj.model = model.model
        model.evaluator = eval_obj
        if OmegaConf.select(cfg, "train_viz_evaluator", default=None) is not None:
            train_viz_eval: Eval = hydra.utils.instantiate(cfg.train_viz_evaluator)
            train_viz_eval.trainer = trainer
            train_viz_eval.model = model.model
            model.train_viz_evaluator = train_viz_eval
        # Load checkpoint weights manually so we can reset the epoch counter
        ckpt_path = cfg.get("ckpt_path")
        if ckpt_path:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            log.info(f"Loaded weights from {ckpt_path}")
        log.info("Starting evaluation!")
        trainer.validate(model=model, datamodule=datamodule)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    train_metrics = trainer.callback_metrics

    # if cfg.get("test"):
    #     log.info("Starting testing!")
    #     ckpt_path = trainer.checkpoint_callback.best_model_path
    #     if ckpt_path == "":
    #         log.warning("Best ckpt not found! Using current weights for testing...")
    #         ckpt_path = None
    #     trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    #     log.info(f"Best ckpt path: {ckpt_path}")

    # test_metrics = trainer.callback_metrics

    # merge train and test metrics
    test_metrics = {}  # my stub
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(
    version_base="1.3",
    config_path="./hydra_configs",
    config_name="train_zarr_cartesian.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    print(OmegaConf.to_yaml(cfg))

    # train the model
    metric_dict, _ = train(cfg)

    # # safely retrieve metric value for hydra-based hyperparameter optimization
    # metric_value = get_metric_value(
    #     metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    # )

    # # return optimized metric
    # return metric_value


if __name__ == "__main__":
    main()
