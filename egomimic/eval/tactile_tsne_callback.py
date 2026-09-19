"""
Validation-time diagnostics for `egomimic.algo.tactile_encoder.TactileEncoder`,
run periodically during training and logged to WandB:

- `TactileEmbeddingTSNECallback`: 2D t-SNE of human vs. robot latents
  (`forward_eval`, unmasked), to watch whether the two embedding clouds
  converge once Stage 2 (OT alignment) begins.
- `TactileValidationLossCallback`: the same masked-MTAE + OT losses
  `training_step` computes, but on held-out data, logged under `Valid/*`.
  `ModelWrapper.validation_step` (pl_utils/pl_model.py) only computes/logs
  anything when a rollout `evaluator` is configured; a self-supervised
  pretraining run like this one has none, so without this callback
  `Valid/*` was never being logged at all -- Lightning's validation loop
  still ran, it just short-circuited into a no-op every batch.
- `TactileReconstructionVizCallback`: input vs. MTAE reconstruction for a
  couple of examples per platform (`TactileEncoder.reconstruct`), to look
  at reconstruction quality directly instead of only its aggregate loss.

All three sample from `trainer.datamodule.valid_datasets` directly (not the
validation dataloader) since they need independent, differently-sized
samples rather than whatever `MultiDataModuleWrapper` batches per step.
"""

from __future__ import annotations

import logging
import random

import torch
from lightning import Callback

logger = logging.getLogger(__name__)


def _sample_batch(datasets: dict, device, num_samples: int, seed: int) -> dict | None:
    batch = {}
    for name, ds in datasets.items():
        if name not in ("human", "robot") or len(ds) == 0:
            continue
        rng = random.Random(seed)
        n = min(num_samples, len(ds))
        idxs = rng.sample(range(len(ds)), n)
        tactile = torch.stack([ds[i]["tactile"] for i in idxs]).to(device)
        batch[name] = {"tactile": tactile}
    if "human" not in batch or "robot" not in batch:
        return None
    return batch


def _valid_datasets(trainer) -> dict | None:
    datamodule = getattr(trainer, "datamodule", None)
    valid_datasets = getattr(datamodule, "valid_datasets", None) if datamodule else None
    return valid_datasets or None


def _log_figure(trainer, key: str, fig, step: int, tag: str) -> None:
    logged = False
    for lgr in trainer.loggers:
        if type(lgr).__name__ == "WandbLogger" and hasattr(lgr, "experiment"):
            try:
                import wandb

                lgr.experiment.log({key: wandb.Image(fig)}, step=step)
                logged = True
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[{tag}] failed to log to wandb: {e}")
    if not logged:
        out_path = f"{trainer.default_root_dir}/{key}_step{step}.png"
        fig.savefig(out_path)
        logger.info(f"[{tag}] no WandB logger found; saved to {out_path}")


class TactileEmbeddingTSNECallback(Callback):
    def __init__(self, every_n_epochs: int = 5, num_samples: int = 128, seed: int = 0):
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.seed = seed

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if self.every_n_epochs <= 0 or (epoch + 1) % self.every_n_epochs != 0:
            return
        valid_datasets = _valid_datasets(trainer)
        if not valid_datasets:
            return

        batch = _sample_batch(valid_datasets, pl_module.device, self.num_samples, self.seed)
        if batch is None:
            logger.info("[TactileTSNE] need both 'human' and 'robot' valid datasets; skipping")
            return

        was_training = pl_module.training
        pl_module.eval()
        try:
            with torch.no_grad():
                processed = pl_module.model.process_batch_for_training(batch)
                preds = pl_module.model.forward_eval(processed)
        finally:
            if was_training:
                pl_module.train()

        if "human_z" not in preds or "robot_z" not in preds:
            logger.info("[TactileTSNE] forward_eval did not return human_z/robot_z; skipping")
            return

        human_z = preds["human_z"].detach().float().cpu().numpy()
        robot_z = preds["robot_z"].detach().float().cpu().numpy()
        self._plot_and_log(trainer, human_z, robot_z, step=trainer.global_step, epoch=epoch)

    def _plot_and_log(self, trainer, human_z, robot_z, step: int, epoch: int) -> None:
        import numpy as np

        try:
            from sklearn.manifold import TSNE
        except ImportError:
            logger.warning("[TactileTSNE] scikit-learn not installed; skipping t-SNE plot")
            return

        combined = np.concatenate([human_z, robot_z], axis=0)
        n = combined.shape[0]
        # TSNE requires perplexity < n_samples; keep it sane for small smoke-test batches.
        perplexity = max(2, min(30, n // 4))
        if n < 4:
            logger.info(f"[TactileTSNE] only {n} points total; skipping (need >= 4)")
            return

        embedded = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=0).fit_transform(
            combined
        )
        n_h = human_z.shape[0]

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(embedded[:n_h, 0], embedded[:n_h, 1], c="tab:blue", label="human", alpha=0.6, s=18)
        ax.scatter(embedded[n_h:, 0], embedded[n_h:, 1], c="tab:orange", label="robot", alpha=0.6, s=18)
        ax.set_title(f"Tactile latent t-SNE (epoch {epoch}, step {step})")
        ax.legend()
        fig.tight_layout()
        _log_figure(trainer, "tactile_tsne", fig, step, "TactileTSNE")
        plt.close(fig)


class TactileValidationLossCallback(Callback):
    """
    Computes the same masked-MTAE + margin-gated-OT losses `training_step`
    does, on a held-out sample, and logs them as `Valid/*`. Uses
    `forward_training(..., increment_step=False)` so this doesn't advance
    the model's own step counter (which drives the warmup/alignment
    schedule) ahead of the real training step count.
    """

    def __init__(self, num_samples: int = 64, seed: int = 1):
        self.num_samples = num_samples
        self.seed = seed

    def on_validation_epoch_end(self, trainer, pl_module):
        valid_datasets = _valid_datasets(trainer)
        if not valid_datasets:
            return
        batch = _sample_batch(valid_datasets, pl_module.device, self.num_samples, self.seed)
        if batch is None:
            logger.info("[TactileValLoss] need both 'human' and 'robot' valid datasets; skipping")
            return

        was_training = pl_module.training
        pl_module.eval()
        try:
            with torch.no_grad():
                processed = pl_module.model.process_batch_for_training(batch)
                preds = pl_module.model.forward_training(processed, increment_step=False)
                losses = pl_module.model.compute_losses(preds, processed)
        finally:
            if was_training:
                pl_module.train()

        for k, v in pl_module.model.log_info({"losses": losses}).items():
            pl_module.log(f"Valid/{k}", v, on_step=False, on_epoch=True, sync_dist=True)


class TactileReconstructionVizCallback(Callback):
    """
    Input vs. MTAE reconstruction for a couple of held-out examples per
    platform. Human tactile has no real spatial layout (see
    tactile_dataset.py), so it's plotted as taxel-value line traces rather
    than an image; robot fingertip images are plotted as grayscale tiles.
    """

    def __init__(self, every_n_epochs: int = 5, num_examples: int = 2, seed: int = 2):
        self.every_n_epochs = every_n_epochs
        self.num_examples = num_examples
        self.seed = seed

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if self.every_n_epochs <= 0 or (epoch + 1) % self.every_n_epochs != 0:
            return
        valid_datasets = _valid_datasets(trainer)
        if not valid_datasets:
            return
        batch = _sample_batch(valid_datasets, pl_module.device, self.num_examples, self.seed)
        if batch is None:
            logger.info("[TactileReconViz] need both 'human' and 'robot' valid datasets; skipping")
            return

        was_training = pl_module.training
        pl_module.eval()
        try:
            with torch.no_grad():
                processed = pl_module.model.process_batch_for_training(batch)
                recon = pl_module.model.reconstruct(processed)
        finally:
            if was_training:
                pl_module.train()

        step = trainer.global_step
        if "human" in recon:
            fig = self._plot_human(recon["human"])
            _log_figure(trainer, "tactile_recon_human", fig, step, "TactileReconViz")
            import matplotlib.pyplot as plt

            plt.close(fig)
        if "robot" in recon:
            fig = self._plot_robot(recon["robot"])
            _log_figure(trainer, "tactile_recon_robot", fig, step, "TactileReconViz")
            import matplotlib.pyplot as plt

            plt.close(fig)

    def _plot_human(self, d: dict):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = d["input"].detach().float().cpu().numpy()  # [B, T, 2, 1, 460]
        recon = d["recon"].detach().float().cpu().numpy()
        b = x.shape[0]
        t_mid = x.shape[1] // 2
        hand_names = ["left", "right"]

        fig, axes = plt.subplots(b, 2, figsize=(10, 3 * b), squeeze=False)
        for i in range(b):
            for c, hand in enumerate(hand_names):
                ax = axes[i][c]
                ax.plot(x[i, t_mid, c, 0], label="input", alpha=0.8, linewidth=1)
                ax.plot(recon[i, t_mid, c, 0], label="recon", alpha=0.8, linewidth=1)
                ax.set_title(f"human ex{i} {hand} hand, t={t_mid} (mask_ratio={d['mask_ratio']:.2f})")
                ax.set_xlabel("taxel index")
                if i == 0 and c == 0:
                    ax.legend()
        fig.tight_layout()
        return fig

    def _plot_robot(self, d: dict):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = d["input"].detach().float().cpu().numpy()  # [B, T, 10, H, W]
        recon = d["recon"].detach().float().cpu().numpy()
        b = x.shape[0]
        t_mid = x.shape[1] // 2
        n_fingers_shown = min(3, x.shape[2])

        fig, axes = plt.subplots(b * 2, n_fingers_shown, figsize=(3 * n_fingers_shown, 6 * b), squeeze=False)
        for i in range(b):
            for f in range(n_fingers_shown):
                axes[2 * i][f].imshow(x[i, t_mid, f], cmap="gray", vmin=0, vmax=1)
                axes[2 * i][f].set_title(f"ex{i} finger{f} input")
                axes[2 * i][f].axis("off")
                axes[2 * i + 1][f].imshow(recon[i, t_mid, f], cmap="gray", vmin=0, vmax=1)
                axes[2 * i + 1][f].set_title(f"ex{i} finger{f} recon")
                axes[2 * i + 1][f].axis("off")
        fig.suptitle(f"robot reconstruction, t={t_mid} (mask_ratio={d['mask_ratio']:.2f})")
        fig.tight_layout()
        return fig
