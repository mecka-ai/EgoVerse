"""Validation evaluator for the span action autoencoder (TemporalCNNAutoencoder).

Two jobs:
  1. Its mere presence makes ``ModelWrapper.validation_step`` log the held-out
     ``Valid/`` reconstruction loss — that logging is gated on a non-None evaluator,
     so with ``~evaluator`` removed the run produces no Validation charts at all.
  2. Each validation pass it renders GT-vs-reconstruction plots of a few val spans
     to W&B. Reconstructions live in ActionNorms (shape-only, time-warped) space, so
     we plot per-channel curves rather than projecting onto the ego image like the
     OAT/QueST tokenizer evaluator (that projection is impossible here — ActionNorms
     discards absolute position/scale/duration).

The plotting is fully wrapped in try/except so it can never block training; the
scalar Valid loss is logged by the ModelWrapper regardless.
"""

import logging

import numpy as np

from egomimic.eval.eval import Eval
from egomimic.rldb.embodiment.embodiment import get_embodiment

log = logging.getLogger(__name__)


class SpanReconstructionEval(Eval):
    """Logs Valid loss (via ModelWrapper) + GT-vs-recon span plots to W&B."""

    def __init__(self, num_spans: int = 4):
        super().__init__()
        self.trainer = None
        self.model = None
        self.num_spans = int(num_spans)
        # Consumed by the standalone eval entrypoint (mode=eval); unused in train.
        self.override_dict = {
            "limit_train_batches": 0,
            "limit_val_batches": 50,
            "check_val_every_n_epoch": 1,
            "max_epochs": 1,
            "min_epochs": 1,
            "num_sanity_val_steps": 0,
        }
        self._logged_this_pass = False

    def _wandb_run(self):
        """Return the wandb.Run handle on rank 0, or None."""
        if self.trainer is None or not self.trainer.is_global_zero:
            return None
        for lgr in self.trainer.loggers or []:
            exp = getattr(lgr, "experiment", None)
            if exp is not None and hasattr(exp, "log") and hasattr(exp, "id"):
                return exp
        return None

    def on_validation_start(self):
        self._logged_this_pass = False

    def on_validation_end(self):
        self._logged_this_pass = False

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Render once per val pass, rank-0 only, on the first val batch.
        if dataloader_idx != 0 or self._logged_this_pass:
            return
        run = self._wandb_run()
        if run is None:
            return
        try:
            self._render(batch, run)
            self._logged_this_pass = True
        except Exception as e:  # never block training on viz
            log.warning("[SpanReconstructionEval] viz failed: %s", e)

    def _render(self, batch, run):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import wandb

        model = self.model
        recons = model.forward_eval(batch)  # {f"{emb}_{ac_key}": (B, L, D)}
        epoch = self.trainer.current_epoch
        images = []

        for embodiment_id, _batch in batch.items():
            emb = get_embodiment(embodiment_id).lower()
            ac_key = model.ac_keys_by_id[embodiment_id]
            gt = _batch[ac_key].detach().float().cpu().numpy()  # (B, L, D)
            rc = recons[f"{emb}_{ac_key}"].detach().float().cpu().numpy()
            B, L, D = gt.shape
            n = min(self.num_spans, B)

            for i in range(n):
                ncols = 4
                nrows = int(np.ceil(D / ncols))
                fig, axes = plt.subplots(
                    nrows, ncols, figsize=(3 * ncols, 2 * nrows), squeeze=False
                )
                mse = float(((gt[i] - rc[i]) ** 2).mean())
                for d in range(D):
                    ax = axes[d // ncols][d % ncols]
                    ax.plot(gt[i, :, d], color="tab:blue", lw=1.2,
                            label="gt" if d == 0 else None)
                    ax.plot(rc[i, :, d], color="tab:red", lw=1.0, ls="--",
                            label="recon" if d == 0 else None)
                    ax.set_title(f"ch{d}", fontsize=7)
                    ax.tick_params(labelsize=6)
                for d in range(D, nrows * ncols):
                    axes[d // ncols][d % ncols].axis("off")
                fig.suptitle(
                    f"{emb} span {i}  epoch={epoch}  mse={mse:.4f}", fontsize=10
                )
                fig.legend(loc="upper right", fontsize=8)
                fig.tight_layout()
                images.append(
                    wandb.Image(fig, caption=f"{emb} span{i} mse={mse:.4f}")
                )
                plt.close(fig)

        if images:
            run.log({"Valid/reconstruction": images, "epoch": epoch})
            log.info(
                "[SpanReconstructionEval] logged %d reconstruction plots at epoch %d",
                len(images), epoch,
            )
