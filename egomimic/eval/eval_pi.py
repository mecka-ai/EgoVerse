import logging

import torch
from torchmetrics import MeanSquaredError

from egomimic.eval.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import get_embodiment
from egomimic.utils.egomimicUtils import (
    frechet_gaussian_over_time,
    reverse_kl_from_samples,
)

logger = logging.getLogger(__name__)


class PIEvalVideo(EvalVideo):
    """
    Eval class for PI models. Computes paired/final MSE, Frechet over time, and
    reverse KL from samples per embodiment (the same Valid/ metric family as
    HPTEvalVideo, so PI and HPT runs are comparable), and delegates
    per-embodiment image visualization to the algo's viz_func.
    """

    def __init__(self, limit_val_batches: int = 400, rkl_samples: int = 8):
        super().__init__(limit_val_batches=limit_val_batches)
        self.rkl_samples = rkl_samples

    def compute_metrics_and_viz(self, batch):
        algo = self.model
        preds = algo.forward_eval(batch)

        metrics = {}
        images_dict = {}
        mse = MeanSquaredError()
        for embodiment_id, _batch in batch.items():
            _batch = algo.data_schematic.unnormalize_data(_batch, embodiment_id)
            embodiment_name = get_embodiment(embodiment_id).lower()
            ac_key = algo.ac_keys[embodiment_id]
            pred_key = f"{embodiment_name}_{ac_key}"
            if pred_key in preds:
                metrics[f"Valid/{pred_key}_paired_mse_avg"] = mse(
                    preds[pred_key].cpu(), _batch[ac_key].cpu()
                )
                metrics[f"Valid/{pred_key}_final_mse_avg"] = mse(
                    preds[pred_key][:, -1].cpu(), _batch[ac_key][:, -1].cpu()
                )
                # Distribution metrics matching HPTEvalVideo's key family.
                # Guarded like viz below: a failure here must not abort a
                # multi-hour run — Valid/*_mse and Valid/Loss remain the
                # primary training signal.
                try:
                    fd = frechet_gaussian_over_time(
                        preds[pred_key], _batch[ac_key]
                    )
                    metrics[f"Valid/{pred_key}_frechet_gauss_avg"] = (
                        fd.mean().item()
                    )
                    metrics[f"Valid/{pred_key}_frechet_gauss_min"] = (
                        fd.min().item()
                    )
                    metrics[f"Valid/{pred_key}_frechet_gauss_max"] = (
                        fd.max().item()
                    )
                    if self.rkl_samples and self.rkl_samples > 1:
                        M = int(self.rkl_samples)
                        samples = self._collect_policy_samples(
                            batch, embodiment_id, pred_key,
                            ref=_batch[ac_key], M=M,
                        )
                        rkl = reverse_kl_from_samples(
                            samples, _batch[ac_key].to(samples.device)
                        )
                        metrics[f"Valid/{pred_key}_reverse_kl_M{M}"] = rkl.item()
                except Exception as exc:
                    logger.warning(
                        "PIEvalVideo: frechet/reverse-KL failed for %s (%r); "
                        "skipping distribution metrics",
                        embodiment_name,
                        exc,
                    )

            # Visualization is non-essential: the Valid/*_mse metrics above are
            # the training signal. A viz error (missing per-embodiment viz_func,
            # unexpected image key, etc.) must never abort a (multi-hour) run, so
            # log and skip images for this embodiment instead of raising.
            try:
                ims = self._visualize_preds(preds, _batch)
            except Exception as exc:
                logger.warning(
                    "PIEvalVideo: visualization failed for %s (%r); skipping images",
                    embodiment_name,
                    exc,
                )
                ims = None
            if ims is not None:
                images_dict[embodiment_id] = ims
        return metrics, images_dict

    @torch.no_grad()
    def _collect_policy_samples(self, batch, embodiment_id, pred_key, ref, M):
        """M stochastic decodes for reverse KL.

        forward_eval passes noise=None to sample_actions, which draws fresh
        flow-matching noise on every call, so repeated calls yield distinct
        action samples. Outputs are unnormalized (forward_eval unnormalizes
        internally), matching the unnormalized reference actions.
        """
        B, T, D = ref.shape
        samples = []
        for _ in range(M):
            out = self.model.forward_eval({embodiment_id: batch[embodiment_id]})
            samples.append(out[pred_key][:, :T, :D].unsqueeze(0))
        return torch.cat(samples, dim=0)

    def _visualize_preds(self, predictions, batch):
        algo = self.model
        if algo.viz_func is None:
            raise ValueError("viz_func is not set")
        embodiment_id = batch["embodiment"][0].item()
        embodiment_name = get_embodiment(embodiment_id).lower()
        if embodiment_name not in algo.viz_func:
            logger.warning(
                "PIEvalVideo: no viz_func configured for embodiment '%s'; skipping viz",
                embodiment_name,
            )
            return None
        return algo.viz_func[embodiment_name](predictions, batch)
