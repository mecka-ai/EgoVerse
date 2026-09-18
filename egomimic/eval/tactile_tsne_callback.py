"""
Lightning callback that periodically embeds a sample of human + robot
tactile windows through `TactileEncoder.forward_eval` and logs a 2D t-SNE
scatter (colored by platform) to WandB, so alignment between the two
embedding clouds can be watched over the course of Stage 1 (MTAE-only) ->
Stage 2 (+ OT) training, instead of only inferring it from the loss curves.
"""

from __future__ import annotations

import logging

import torch
from lightning import Callback

logger = logging.getLogger(__name__)


class TactileEmbeddingTSNECallback(Callback):
    def __init__(self, every_n_epochs: int = 5, num_samples: int = 128, seed: int = 0):
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.seed = seed

    def _sample_batch(self, datasets: dict, device) -> dict | None:
        import random

        batch = {}
        for name, ds in datasets.items():
            if name not in ("human", "robot") or len(ds) == 0:
                continue
            rng = random.Random(self.seed)
            n = min(self.num_samples, len(ds))
            idxs = rng.sample(range(len(ds)), n)
            tactile = torch.stack([ds[i]["tactile"] for i in idxs]).to(device)
            batch[name] = {"tactile": tactile}
        if "human" not in batch or "robot" not in batch:
            return None
        return batch

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if self.every_n_epochs <= 0 or (epoch + 1) % self.every_n_epochs != 0:
            return
        datamodule = getattr(trainer, "datamodule", None)
        valid_datasets = getattr(datamodule, "valid_datasets", None) if datamodule else None
        if not valid_datasets:
            return

        batch = self._sample_batch(valid_datasets, pl_module.device)
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

        logged = False
        for lgr in trainer.loggers:
            if type(lgr).__name__ == "WandbLogger" and hasattr(lgr, "experiment"):
                try:
                    import wandb

                    lgr.experiment.log({"tactile_tsne": wandb.Image(fig)}, step=step)
                    logged = True
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"[TactileTSNE] failed to log to wandb: {e}")
        if not logged:
            out_path = f"{trainer.default_root_dir}/tactile_tsne_epoch{epoch}.png"
            fig.savefig(out_path)
            logger.info(f"[TactileTSNE] no WandB logger found; saved to {out_path}")
        plt.close(fig)
