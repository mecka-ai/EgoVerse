import logging

from torchmetrics import MeanSquaredError

from egomimic.eval.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import get_embodiment

logger = logging.getLogger(__name__)


class PIEvalVideo(EvalVideo):
    """
    Eval class for PI models. Computes paired/final MSE per embodiment and
    delegates per-embodiment image visualization to the algo's viz_func.
    """

    def compute_metrics_and_viz(self, batch, do_viz=True):
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

            # Visualization is non-essential: the Valid/*_mse metrics above are
            # the training signal. A viz error (missing per-embodiment viz_func,
            # unexpected image key, etc.) must never abort a (multi-hour) run, so
            # log and skip images for this embodiment instead of raising.
            if do_viz:
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
