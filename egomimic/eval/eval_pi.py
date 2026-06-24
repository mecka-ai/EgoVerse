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

    @staticmethod
    def _xyzypr_split_idx(action_dim):
        """Index lists for the translation (xyz) and rotation (ypr) channels of
        a cartesian action laid out as a sequence of 6D [x,y,z,yaw,pitch,roll]
        poses (e.g. mecka bimanual = left[0:6] + right[6:12]). Returns
        (xyz_idx, ypr_idx) or None if the action isn't a clean stack of 6D poses.
        """
        if action_dim <= 0 or action_dim % 6 != 0:
            return None
        n_pose = action_dim // 6
        xyz_idx = [6 * p + i for p in range(n_pose) for i in range(3)]
        ypr_idx = [6 * p + 3 + i for p in range(n_pose) for i in range(3)]
        return xyz_idx, ypr_idx

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
                # torchmetrics MeanSquaredError.update calls preds.view(-1),
                # which requires contiguous inputs; slices like [:, -1] and
                # advanced-index gathers are not guaranteed contiguous, so
                # .contiguous() every tensor handed to mse().
                pred = preds[pred_key].cpu()
                gt = _batch[ac_key].cpu()
                metrics[f"Valid/{pred_key}_paired_mse_avg"] = mse(
                    pred.contiguous(), gt.contiguous()
                )
                metrics[f"Valid/{pred_key}_final_mse_avg"] = mse(
                    pred[:, -1].contiguous(), gt[:, -1].contiguous()
                )
                # Break the combined MSE into translation (xyz, meters) vs
                # rotation (ypr, radians) so we can see which channel dominates
                # the error. Only valid when the action is a stack of 6D poses.
                split = self._xyzypr_split_idx(pred.shape[-1])
                if split is not None:
                    xyz_idx, ypr_idx = split
                    metrics[f"Valid/{pred_key}_paired_xyz_mse_avg"] = mse(
                        pred[..., xyz_idx].contiguous(), gt[..., xyz_idx].contiguous()
                    )
                    metrics[f"Valid/{pred_key}_paired_ypr_mse_avg"] = mse(
                        pred[..., ypr_idx].contiguous(), gt[..., ypr_idx].contiguous()
                    )
                    metrics[f"Valid/{pred_key}_final_xyz_mse_avg"] = mse(
                        pred[:, -1][..., xyz_idx].contiguous(),
                        gt[:, -1][..., xyz_idx].contiguous(),
                    )
                    metrics[f"Valid/{pred_key}_final_ypr_mse_avg"] = mse(
                        pred[:, -1][..., ypr_idx].contiguous(),
                        gt[:, -1][..., ypr_idx].contiguous(),
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
