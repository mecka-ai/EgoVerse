import random
import time
from collections import OrderedDict, deque
from typing import Any, Dict

import hydra
import numpy as np
import torch
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf

import egomimic.utils.tensor_utils as TensorUtils
from egomimic.rldb.zarr.utils import DataSchematic


class ModelWrapper(LightningModule):
    """
    Wrapper class around robomimic models to ensure compatibility with Pytorch Lightning.
    """

    debug_loss_spike = False
    debug_loss_spike_factor = 1000.0
    debug_loss_spike_prob = 0.03
    grad_norm_mad_scale = 3.0
    grad_norm_mad_min_count = 100
    grad_norm_mad_window = 200

    def __init__(
        self,
        robomimic_model=None,
        optimizer=None,
        scheduler=None,
        config_tree=None,
        data_schematic_state=None,
        scheduler_interval="step",
        scheduler_frequency: int = 1,
        viz_func=None,
        evaluator=None,
        enable_grad_norm: bool = True,
    ):
        """
        Args:
            model (PolicyAlgo): robomimic model to wrap.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["robomimic_model", "viz_func"])

        if config_tree is not None:
            self.model = self._instantiate_model(
                config_tree, data_schematic_state, viz_func
            )
        elif robomimic_model is not None:  # legacy support
            self.model = robomimic_model
        else:
            raise ValueError(
                "ModelWrapper requires either an instantiated robomimic_model or "
                "a config_tree with data_schematic_state."
            )
        self.nets = (
            self.model.nets
        )  # to ensure the lightning module has access to the model's parameters
        try:
            self.params = self.model.nets["policy"].params
        except Exception:
            pass
        self.enable_grad_norm = enable_grad_norm
        self.grad_norm_history = deque(maxlen=self.grad_norm_mad_window)

        self.epoch_memory_stats = []  # Store memory stats per epoch
        self.evaluator = evaluator
        # Optional second evaluator that consumes dataloader_idx=1 (the
        # train_viz val pass). Wired by trainHydra.py when the data config
        # populates train_viz_datasets.
        self.train_viz_evaluator = None
        # Optional metric prefix (e.g. "Valid_oph/") for logging validation
        # LOSSES on dataloader_idx=1 as a full second val set. Wired by
        # trainHydra.py from the top-level `second_val_prefix` config key.
        # Default None = legacy behavior (no losses logged for idx=1).
        self.second_val_metric_prefix = None
        # Optional THIRD evaluator that consumes dataloader_idx=2 (the extra_val
        # slot), plus its own metric prefix. Wired by trainHydra.py when the data
        # config populates `extra_val_datasets` and the run sets
        # `+evaluator@extra_val_evaluator=...` / `+third_val_prefix=...`.
        # Defaults None/None = exact legacy behavior (no idx=2 loader exists).
        self.extra_val_evaluator = None
        self.extra_val_metric_prefix = None

    @staticmethod
    def _as_config(cfg):
        if cfg is None:
            return None
        if isinstance(cfg, DictConfig):
            return cfg
        return OmegaConf.create(cfg)

    def _instantiate_model(self, config_tree, data_schematic_state, viz_func=None):
        cfg = self._as_config(config_tree)
        data_schematic = DataSchematic.from_state(data_schematic_state)
        return hydra.utils.instantiate(
            cfg.model.robomimic_model,
            data_schematic=data_schematic,
            viz_func=viz_func,
        )

    # batch is now a dict, handle on model side
    def training_step(self, batch, batch_idx):
        self.train()
        loss_dicts = []

        t0 = time.time()
        batch = self.model.process_batch_for_training(batch)
        t1 = time.time()
        predictions = self.model.forward_training(batch)
        t2 = time.time()
        losses = self.model.compute_losses(predictions, batch)
        t3 = time.time()
        loss_dicts.append(losses)

        self.log(
            "Timing/Process_Batch_Sec",
            t1 - t0,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        self.log(
            "Timing/Forward_Pass_Sec",
            t2 - t1,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        self.log(
            "Timing/Compute_Losses_Sec",
            t3 - t2,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        # Average over both the hand and robot batch if applicable
        losses = OrderedDict()
        for key in loss_dicts[0].keys():
            losses[key] = torch.mean(
                torch.stack([loss_dict[key] for loss_dict in loss_dicts])
            )

        if (
            self.debug_loss_spike
            and random.random() < self.debug_loss_spike_prob
            and self.global_step > 100
        ):
            losses["action_loss"] = losses["action_loss"] * self.debug_loss_spike_factor
            if self.trainer.is_global_zero:
                print(
                    f"[LOSS_SPIKE] step={self.global_step} factor={self.debug_loss_spike_factor}",
                    flush=True,
                )

        info = {}
        info["losses"] = TensorUtils.detach(losses)
        for k, v in self.model.log_info(info).items():
            self.log("Train/" + k, v, sync_dist=True, on_step=True, on_epoch=False)

        return losses["action_loss"]

    def on_after_backward(self):
        if not self.enable_grad_norm:
            return
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), max_norm=float("inf")
        )
        grad_norm_val = float(grad_norm)
        info = {"policy_grad_norms_raw": grad_norm_val}
        grad_norm_flagged = False

        if len(self.grad_norm_history) >= self.grad_norm_mad_min_count:
            values = np.array(self.grad_norm_history, dtype=np.float32)
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            if mad > 0.0:
                threshold = median + self.grad_norm_mad_scale * mad
                info["policy_grad_norms_mad_threshold"] = threshold
                grad_norm_flagged = grad_norm_val > threshold
                info["policy_grad_norms_mad_flag"] = float(grad_norm_flagged)
                if grad_norm_flagged:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=median)
                    if self.trainer.is_global_zero:
                        print(
                            "[GRAD_NORM_SPIKE] "
                            f"step={self.global_step} "
                            f"grad_norm={grad_norm_val:.4f} "
                            f"median={median:.4f} "
                            f"mad={mad:.4f} "
                            f"threshold={threshold:.4f}",
                            flush=True,
                        )

        if not grad_norm_flagged:
            self.grad_norm_history.append(grad_norm_val)
        for k, v in info.items():
            self.log("Train/" + k, v, on_step=True, on_epoch=False, sync_dist=True)

    def on_before_optimizer_step(self, optimizer):
        if not self.enable_grad_norm:
            return
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), max_norm=float("inf")
        )
        self.log(
            "Train/policy_grad_norms_clipped",
            float(grad_norm),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

    def _evaluator_for(self, dataloader_idx: int):
        """Route val batches by dataloader_idx: 0 → eval, 1 → train_viz,
        2 → extra_val (only present when the data config defines a third leg)."""
        if dataloader_idx == 0:
            return self.evaluator
        if dataloader_idx == 1:
            return self.train_viz_evaluator
        if dataloader_idx == 2:
            return getattr(self, "extra_val_evaluator", None)
        return None

    def _rank_barrier(self, tag: str) -> None:
        """Synchronize all ranks at a named point. Cheap when everyone gets
        here at roughly the same time; useful to avoid NCCL ALLREDUCE timeouts
        when one rank has been slower (e.g. first-epoch zarr JPEG read burst,
        val-loop imbalance, sporadic filesystem stalls). No-op outside a
        distributed run."""
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return
        print(f"[BARRIER {tag}] rank={self.global_rank} entering", flush=True)
        torch.distributed.barrier()
        print(f"[BARRIER {tag}] rank={self.global_rank} released", flush=True)

    @property
    def _extra_rank_barriers(self) -> bool:
        """Extra epoch/val-boundary barriers, opt-in per algo (e.g. WAM sets
        ``_needs_rank_barriers = True``: its val loop encodes mp4s with high
        per-rank time variance). Off for every other algo so non-WAM runs keep
        their exact legacy synchronization behavior."""
        return bool(getattr(self.model, "_needs_rank_barriers", False))

    def on_validation_start(self):
        # Sync before the val loop so no rank enters validation while others
        # are still finishing the previous training epoch's last step.
        if self._extra_rank_barriers:
            self._rank_barrier("on_validation_start")
        if self.evaluator is None:
            return
        self.model.device = self.device

        self.evaluator.on_validation_start()
        if self.train_viz_evaluator is not None:
            self.train_viz_evaluator.on_validation_start()
        if getattr(self, "extra_val_evaluator", None) is not None:
            self.extra_val_evaluator.on_validation_start()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        evaluator = self._evaluator_for(dataloader_idx)
        if evaluator is None:
            return
        # When val_dataloader returns a list of CombinedLoaders (train_viz
        # eval pass), Lightning's fetcher wraps each inner CombinedLoader
        # batch as (batch_dict, batch_idx, sub_loader_idx).  Unwrap so
        # process_batch_for_training always receives the raw dict.
        if isinstance(batch, tuple) and len(batch) == 3 and isinstance(batch[0], dict):
            batch = batch[0]
        batch = self.model.process_batch_for_training(batch)

        # Log validation loss on the held-out valid set (dataloader_idx == 0).
        # dataloader_idx == 1 is the train_viz pass (qualitative rollouts only),
        # so it has no meaningful loss to report.
        if dataloader_idx == 0:
            predictions = self.model.forward_training(batch)
            losses = self.model.compute_losses(predictions, batch)
            info = {"losses": TensorUtils.detach(losses)}
            for k, v in self.model.log_info(info).items():
                self.log(
                    "Valid/" + k,
                    v,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )
        else:
            # Auxiliary val legs: idx=1 (train_viz slot) and idx=2 (extra_val
            # slot). Each logs the same loss family under its OWN prefix (e.g.
            # "Valid_oph/", "Valid_trainviz/") when that prefix is configured.
            # Only active when `second_val_prefix` / `third_val_prefix` is set in
            # the run config — legacy runs (prefix None) keep idx=1 as a
            # rollout-only pass and never have an idx=2 loader at all.
            aux_prefix = None
            if dataloader_idx == 1:
                aux_prefix = getattr(self, "second_val_metric_prefix", None)
            elif dataloader_idx == 2:
                aux_prefix = getattr(self, "extra_val_metric_prefix", None)
            if aux_prefix:
                predictions = self.model.forward_training(batch)
                losses = self.model.compute_losses(predictions, batch)
                info = {"losses": TensorUtils.detach(losses)}
                for k, v in self.model.log_info(info).items():
                    self.log(
                        aux_prefix + k,
                        v,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                        add_dataloader_idx=False,
                    )

        print(
            f"[VAL_STEP] rank={self.global_rank}, batch_idx={batch_idx}, "
            f"dataloader_idx={dataloader_idx}",
            flush=True,
        )
        evaluator.on_validation_step(batch, batch_idx, dataloader_idx)

    def on_validation_end(self):
        print(f"[ON_VALIDATION_END] rank={self.global_rank}", flush=True)
        if self.evaluator is not None:
            self.evaluator.on_validation_end()
        if self.train_viz_evaluator is not None:
            self.train_viz_evaluator.on_validation_end()
        if getattr(self, "extra_val_evaluator", None) is not None:
            self.extra_val_evaluator.on_validation_end()

        # Sync after the val loop — mp4 writers on different ranks finish at
        # different times, and without this wait the next train epoch's first
        # ALLREDUCE can hit the 30-min NCCL default. Guarded (unlike the old
        # bare barrier) so single-process runs without a process group work.
        self._rank_barrier("on_validation_end")

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        config_tree = getattr(self.hparams, "config_tree", None)
        if config_tree is not None:
            cfg = self._as_config(config_tree)
            optimizer = hydra.utils.instantiate(
                cfg.model.optimizer,
                params=self.trainer.model.parameters(),
            )
            if callable(optimizer):
                optimizer = optimizer()
            scheduler_cfg = cfg.model.get("scheduler")
            if scheduler_cfg is not None:
                scheduler = hydra.utils.instantiate(
                    scheduler_cfg,
                    optimizer=optimizer,
                )
                if callable(scheduler):
                    scheduler = scheduler()
            else:
                scheduler = None
        else:
            optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
            scheduler = (
                self.hparams.scheduler(optimizer=optimizer)
                if self.hparams.scheduler is not None
                else None
            )

        if scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.hparams.scheduler_interval,
                    "frequency": self.hparams.scheduler_frequency,
                },
            }
        return {"optimizer": optimizer}

    def on_fit_start(self):
        self.model.device = self.device
        # Guarded (unlike the old bare barrier) so single-process runs without
        # a process group work; identical behavior under DDP.
        self._rank_barrier("on_fit_start")

    def on_train_epoch_start(self):
        # Barrier at the start of each epoch (WAM-gated) — the first-epoch
        # zarr JPEG read burst is uneven across ranks, so this pulls the slow
        # rank in before any per-step ALLREDUCE.
        if self._extra_rank_barriers:
            self._rank_barrier("on_train_epoch_start")
        for i, param_group in enumerate(self.optimizers().param_groups):
            self.log(
                f"Optimizer/param_group_{i}_lr",
                param_group["lr"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        return super().on_train_epoch_start()

    def on_train_epoch_end(self):
        # Sync at epoch end (WAM-gated) so no rank races into
        # on_validation_start or the next epoch's first step while another is
        # still finishing its last backward pass.
        if self._extra_rank_barriers:
            self._rank_barrier("on_train_epoch_end")
        return super().on_train_epoch_end()
