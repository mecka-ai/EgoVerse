import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.io as tvio
from torchmetrics import MeanSquaredError

from egomimic.eval.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import get_embodiment


def _is_rank0() -> bool:
    return (
        (not torch.distributed.is_available())
        or (not torch.distributed.is_initialized())
        or (torch.distributed.get_rank() == 0)
    )


class WAMEvalVideo(EvalVideo):
    """Evaluator for WAM (World-Action Model). Per embodiment it writes TWO
    separate validation videos per val sample:

      1. ``validation_video_*.mp4`` — animated predicted vs GT **action**
         overlay drawn on the GT observation clip (via ``viz_func``); BC loss
         + action MSE metrics.
      2. ``predicted_video_*.mp4``  — the world model's **predicted future
         frames** (rolled out chunk-by-chunk by ``WAM.val_rollout``,
         VAE-decoded), with the SAME action overlay drawn on the imagined
         frames. Written from the frames forward_eval stashes on the algo as
         ``_eval_frames[eid]``.

    Overlay-animation layout (generalized from the upstream aria geometry):
      the val clip has T_pix pixel frames; the dataset provides n_gt actions
      for the (T_pix - 1) predicted frames, so each displayed pixel frame is
      held ``FR = n_gt / (T_pix - 1)`` output frames (mecka wan22_5b:
      n_gt=16, T_pix=17 -> FR=1; upstream aria: n_gt=192, T_pix=33 -> FR=6).
      Within each DiT action-chunk window (``num_action_per_block`` actions)
      the arrow trail shrinks by one action per output frame and resets to
      full length at every chunk boundary.
    """

    # Playback rate for both mp4s. Our mecka WAM clips are short (16 predicted
    # frames per sample at FR=1), so play at 10 fps (~1.6 s/sample; real-time
    # source rate would be 30). Both videos have the same frame count per
    # sample so they stay in sync side-by-side.
    VAL_FPS = 10
    PREDICTED_FPS = 10

    def __init__(
        self,
        viz_func=None,
        limit_val_batches: int = 400,
        teacher_force_rolling: bool = False,
    ):
        # This fork's EvalVideo takes only limit_val_batches; the per-embodiment
        # action-overlay viz callables come in via viz_func (evaluator/viz config).
        # ``teacher_force_rolling`` is metadata read by the offline eval driver
        # (``eval_dreamzero._patch_algo_use_sample_rolling``) to pick fully-AR
        # vs dreamzero Fig-14a TF rolling. Ignored inside this class.
        super().__init__(limit_val_batches=limit_val_batches)
        self.viz_func = viz_func
        self.teacher_force_rolling = bool(teacher_force_rolling)

    def compute_metrics_and_viz(self, batch):
        algo = self.model
        preds = algo.forward_eval(batch)  # rolls out actions + future frames

        metrics = {}
        images_dict = {}
        mse = MeanSquaredError()
        total_loss = None
        n_loss = 0

        for embodiment_id, _batch in batch.items():
            _batch = algo.data_schematic.unnormalize_data(_batch, embodiment_id)
            name = get_embodiment(embodiment_id).lower()
            ac_key = algo.ac_keys[embodiment_id]
            pred_key = f"{name}_{ac_key}"

            # Capture the full clip BEFORE any collapse — we need it to draw
            # the per-frame action overlay on every downsampled GT frame so
            # validation_video plays 1:1 in sync with predicted_video.
            clip_by_cam = {}
            for ck in algo.camera_keys.get(embodiment_id, []):
                if (
                    ck in _batch
                    and torch.is_tensor(_batch[ck])
                    and _batch[ck].dim() == 5
                ):
                    clip_by_cam[ck] = _batch[ck]  # (B, T, C, H, W)

            loss_key = f"{name}_loss"
            if loss_key in preds:
                loss_val = preds[loss_key]
                metrics[f"Valid/{loss_key}"] = loss_val
                total_loss = loss_val if total_loss is None else total_loss + loss_val
                n_loss += 1

            if pred_key in preds:
                # MSE aligns pred to gt's shorter length — pred can be longer
                # (full rolling output when val cam_horizon > train's) while gt
                # stays at the dataset action_horizon. The paired MSE is a
                # per-timestep metric so we compare only where both exist.
                gt_actions = _batch[ac_key]
                T_gt = gt_actions.shape[1]
                metrics[f"Valid/{pred_key}_paired_mse_avg"] = mse(
                    preds[pred_key][:, :T_gt].cpu(), gt_actions.cpu()
                )

            # Video 1: per-output-frame action-overlay animation on the GT clip.
            if clip_by_cam:
                images_dict[embodiment_id] = self._animate_overlays(
                    algo, embodiment_id, name, ac_key, preds, _batch, clip_by_cam
                )
            else:
                # No clip in the batch (single-frame cameras): fall back to the
                # static one-frame overlay.
                for ck in algo.camera_keys.get(embodiment_id, []):
                    if (
                        ck in _batch
                        and torch.is_tensor(_batch[ck])
                        and _batch[ck].dim() == 5
                    ):
                        _batch[ck] = _batch[ck][:, 0]
                images_dict[embodiment_id] = self._visualize_preds(preds, _batch)

        if total_loss is not None and n_loss > 0:
            metrics["Valid/action_loss"] = total_loss / n_loss

        return metrics, images_dict

    # ------------------------------------------------------------------
    # Overlay animation (val_video) + pred-frame overlay (predicted_video)
    # ------------------------------------------------------------------
    def _animate_overlays(
        self, algo, embodiment_id, name, ac_key, preds, _batch, clip_by_cam
    ):
        pred_key = f"{name}_{ac_key}"
        some_ck = next(iter(clip_by_cam))
        T_pix = clip_by_cam[some_ck].shape[1]  # e.g. 17
        gt_actions_full = _batch[ac_key]
        pred_actions_full = preds.get(pred_key)

        # Derive the chunk size from the DiT config so this stays in sync.
        dit = algo.nets["policy"].dit
        K = getattr(dit, "num_frame_per_block", 1)
        num_action_per_block = getattr(dit, "num_action_per_block", None)
        if num_action_per_block is None:
            total_actions = gt_actions_full.shape[1]
            M = max(1, (algo.nets["policy"].num_video_frames - 1) // K)
            num_action_per_block = max(1, total_actions // M)
        H_actions = int(num_action_per_block)

        # GT actions come from the dataset (ROPE-locked action_horizon). Pred
        # actions come from the rolling sampler and can be LONGER (val clips
        # past the training horizon). Track n_gt and n_pred separately so the
        # pred trail extends past GT's window and GT arrows simply drop out
        # once exhausted.
        n_gt = gt_actions_full.shape[1]
        n_pred = pred_actions_full.shape[1] if pred_actions_full is not None else 0

        # Output frames per displayed pixel frame (see class docstring).
        FR = max(1, int(round(n_gt / max(1, T_pix - 1))))
        total_out_frames = (T_pix - 1) * FR

        # Per-raw-frame head-pose chunk (see Mecka.get_wam_keymap ->
        # ``obs_head_pose_chunk``). Shape (B, cam_horizon, 7) xyzwxyz; row 0 ==
        # the pose the data-pipeline transform used as target (H_0), row k ==
        # the pose current at pixel_idx=k. If absent (older checkpoints /
        # embodiments without head pose in zarr), we skip the per-frame
        # reprojection and fall back to H_0-frame actions unchanged.
        head_pose_chunk = _batch.get("obs_head_pose_chunk")

        def _slices_for(start, end):
            """(gt_slice, pred_slice) for one animation window; None past end."""
            if start < n_gt:
                gt_slice = gt_actions_full[:, start : min(end, n_gt)]
            else:
                gt_slice = None
            if pred_actions_full is not None and start < n_pred:
                pred_slice = pred_actions_full[:, start : min(end, n_pred)]
            else:
                pred_slice = None
            return gt_slice, pred_slice

        def _reproject(gt_slice, pred_slice, pixel_idx):
            if head_pose_chunk is not None and pixel_idx < head_pose_chunk.shape[1]:
                if gt_slice is not None and gt_slice.shape[1] > 0:
                    gt_slice = self._reproject_actions_to_head_t(
                        gt_slice, head_pose_chunk, pixel_idx
                    )
                if pred_slice is not None and pred_slice.shape[1] > 0:
                    pred_slice = self._reproject_actions_to_head_t(
                        pred_slice, head_pose_chunk, pixel_idx
                    )
            return gt_slice, pred_slice

        def _frame_dicts(gt_slice, pred_slice, base_batch):
            """Build (preds_f, batch_f) with the sliced chunks. A None slice
            becomes an empty (B, 0, D) chunk so the renderer draws zero arrows
            for that family while still drawing the other."""
            batch_f = dict(base_batch)
            batch_f[ac_key] = (
                gt_slice if gt_slice is not None else gt_actions_full[:, :0]
            )
            preds_f = dict(preds)
            preds_f[pred_key] = (
                pred_slice
                if pred_slice is not None
                else (
                    pred_actions_full[:, :0]
                    if pred_actions_full is not None
                    else gt_actions_full[:, :0]
                )
            )
            return preds_f, batch_f

        # ---- Video 1: overlay animation on the GT clip ---------------------
        #
        # ``start``: shifted by +FR so that at each moment the pixel updates
        # (f % FR == 0), the arrow's origin action index matches the current
        # pixel's source time. Advances by 1 per output frame so the trail
        # shrinks smoothly.
        # ``end``: fixed at the end of the current H_actions-long DiT chunk
        # (also +FR-shifted). At each chunk boundary the trail resets to full
        # length, then shrinks H_actions -> 1 as ``start`` advances.
        per_frame_ims = []
        for f in range(total_out_frames):
            pixel_idx = 1 + f // FR  # skip frame 0 = anchor
            start = f + FR
            chunk_bucket = f // H_actions
            end = (chunk_bucket + 1) * H_actions + FR

            _batch_f = dict(_batch)
            for ck, clip in clip_by_cam.items():
                # pixel_idx can overshoot the read clip (rare rounding);
                # freeze on the last pixel frame so playback continues.
                _batch_f[ck] = clip[:, min(pixel_idx, clip.shape[1] - 1)]

            gt_slice, pred_slice = _slices_for(start, end)
            gt_slice, pred_slice = _reproject(gt_slice, pred_slice, pixel_idx)
            preds_f, _batch_f = _frame_dicts(gt_slice, pred_slice, _batch_f)
            per_frame_ims.append(self._visualize_preds(preds_f, _batch_f))

        stacked = np.stack(per_frame_ims, axis=0)  # (F, B, H, W, 3)
        stacked = stacked.transpose(1, 0, 2, 3, 4)  # (B, F, H, W, 3)
        B_, F_, H_, W_, C_ = stacked.shape
        out_images = stacked.reshape(B_ * F_, H_, W_, C_)

        # ---- Video 2 overlay: arrows on the model's imagined frames --------
        # Render the SAME action overlay on top of the VAE-decoded predicted
        # frames (upscaled to GT-clip resolution so the intrinsics-based
        # projection lands correctly), so ``predicted_video_*.mp4`` shows what
        # the DiT rolled out AND the action trajectory it produced.
        #
        # RANK-GUARD: this loop calls ``viz_func`` T_pred times per sample
        # (matplotlib/PIL under the hood). On multi-GPU training val loops the
        # per-rank time variance was tripping the 30-min NCCL watchdog on the
        # next sync collective. Only rank 0 does the overlay compute + writes —
        # other ranks fall back to the raw (non-overlay) VAE decode in
        # ``on_validation_step`` and clear their buffers in
        # ``on_validation_end``.
        pred_video_t = getattr(self.model, "_eval_frames", {}).get(embodiment_id)
        if _is_rank0() and pred_video_t is not None:
            image_key = some_ck
            ref_H = clip_by_cam[image_key].shape[-2]
            ref_W = clip_by_cam[image_key].shape[-1]
            Bp, Cp, T_pred, _, _ = pred_video_t.shape
            pv = F.interpolate(
                pred_video_t.permute(0, 2, 1, 3, 4).reshape(
                    Bp * T_pred,
                    Cp,
                    pred_video_t.shape[-2],
                    pred_video_t.shape[-1],
                ),
                size=(ref_H, ref_W),
                mode="bilinear",
                align_corners=False,
            ).reshape(Bp, T_pred, Cp, ref_H, ref_W)
            # Normalize to [0, 1] (viz_func expects the same range as
            # ``clip_by_cam``, which the data pipeline provides in [0, 1]).
            pv = (pv.clamp(-1, 1) + 1.0) * 0.5
            pred_overlay_frames = []
            for p in range(T_pred):
                # ``_eval_frames`` already dropped the anchor frame
                # (WAM.val_rollout: viz_video = pred_frames[:, :, 1:]), so p=0
                # corresponds to pixel_idx=1 (first predicted frame).
                pixel_idx = p + 1
                f_val = p * FR  # equivalent frame in val_video's timeline
                start = f_val + FR
                chunk_bucket = f_val // H_actions
                end = (chunk_bucket + 1) * H_actions + FR

                gt_slice, pred_slice = _slices_for(start, end)
                gt_slice, pred_slice = _reproject(gt_slice, pred_slice, pixel_idx)
                _batch_p = dict(_batch)
                _batch_p[image_key] = pv[:, p]  # (B, C, H, W)
                preds_p, _batch_p = _frame_dicts(gt_slice, pred_slice, _batch_p)
                pred_overlay_frames.append(self._visualize_preds(preds_p, _batch_p))

            p_stacked = np.stack(pred_overlay_frames, axis=0)  # (T, B, H, W, 3)
            p_stacked = p_stacked.transpose(1, 0, 2, 3, 4)  # (B, T, H, W, 3)
            Bp_, Tp_, Hp_, Wp_, Cp_ = p_stacked.shape
            if not hasattr(self, "_pred_overlay_cache"):
                self._pred_overlay_cache = {}
            # Cache as (B*T_pred, H, W, 3) so on_validation_step can extend
            # ``_pred_buffer`` with the same layout the no-overlay path yields.
            self._pred_overlay_cache[embodiment_id] = p_stacked.reshape(
                Bp_ * Tp_, Hp_, Wp_, Cp_
            )

        return out_images

    def _visualize_preds(self, predictions, batch):
        if self.viz_func is None:
            raise ValueError("viz_func is not set")
        name = get_embodiment(batch["embodiment"][0].item()).lower()
        return self.viz_func[name](predictions, batch)

    # ------------------------------------------------------------------
    # Buffering + mp4 writes: one mp4 per val sample, sizes inferred from
    # per-embodiment step counts (robust to per-embodiment cam_horizon
    # overrides — a long-clip embodiment still gets ONE mp4 per episode).
    # ------------------------------------------------------------------
    def _ensure_buffers(self):
        if not hasattr(self, "_pred_buffer"):
            self._pred_buffer = {}
            self._pred_step_count = {}
            self._val_step_count = {}

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Reimplements the base's metric-log + frame-buffer WITHOUT its
        # 1000-frame mid-loop flush: the per-sample mp4 split in
        # on_validation_end needs the full buffer (a mid-flush would both
        # break the split arithmetic and collide with the per-sample
        # ``validation_video_{s}.mp4`` names).
        self._ensure_buffers()
        metrics, images_dict = self.compute_metrics_and_viz(batch)

        device = self.trainer.lightning_module.device
        metrics = {
            k: (v.to(device) if torch.is_tensor(v) else torch.tensor(v, device=device))
            for k, v in metrics.items()
        }

        for key, images in images_dict.items():
            if key not in self.val_image_buffer or self.val_image_buffer[key] is None:
                self.val_image_buffer[key] = []
                self.val_counter[key] = 0
            self.val_image_buffer[key].extend(torch.from_numpy(images))
            self._val_step_count[key] = self._val_step_count.get(key, 0) + 1

        for eid, video in getattr(self.model, "_eval_frames", {}).items():
            if video is None:
                continue
            # Prefer the action-overlay-annotated pred frames built inside
            # compute_metrics_and_viz. Fall back to raw upscaled pred frames
            # if the overlay cache is missing (non-rank0, or no clip batch).
            overlay = getattr(self, "_pred_overlay_cache", {}).pop(eid, None)
            if overlay is not None:
                frames = overlay
            else:
                frames = self._predicted_frames_all(video)  # (B*T, Hd, Wd, 3)
            self._pred_buffer.setdefault(eid, []).extend(torch.from_numpy(frames))
            self._pred_step_count[eid] = self._pred_step_count.get(eid, 0) + 1

        # add_dataloader_idx=False: keep canonical Valid/ chart names (see the
        # base EvalVideo for rationale).
        self.trainer.lightning_module.log_dict(
            metrics, sync_dist=True, add_dataloader_idx=False
        )

    def on_validation_end(self):
        self._ensure_buffers()
        # RANK-GUARD: h264 encode + shared-FS writes on every DDP rank caused
        # per-rank divergence (NCCL watchdog). Each rank's buffer only holds
        # frames from its OWN shard of val samples (DistributedSampler), so
        # rank-0-only writes lose the other shards' videos — fine for
        # training-time eyeballing; offline eval runs single-rank anyway.
        if not _is_rank0():
            self._pred_buffer = {}
            self._pred_step_count = {}
            for key in list(self.val_image_buffer.keys()):
                self.val_image_buffer[key] = []
                self.val_counter[key] = 0
            self._val_step_count = {}
            return

        # PREDICTED: one mp4 per val step (per-eid sizes inferred from the
        # buffer length / step count).
        for eid, buf in self._pred_buffer.items():
            if not buf:
                continue
            n_samples = self._pred_step_count.get(eid, 0)
            if n_samples <= 0:
                continue
            frames_per_sample = len(buf) // n_samples
            if frames_per_sample <= 0:
                continue
            out_dir = os.path.join(
                self.video_dir(),
                f"epoch_{self.trainer.current_epoch}",
                str(get_embodiment(eid)),
            )
            os.makedirs(out_dir, exist_ok=True)
            for s in range(n_samples):
                frames = torch.stack(
                    buf[s * frames_per_sample : (s + 1) * frames_per_sample]
                )
                tvio.write_video(
                    os.path.join(out_dir, f"predicted_video_{s}.mp4"),
                    frames,
                    fps=self.PREDICTED_FPS,
                    video_codec="h264",
                )
        self._pred_buffer = {}
        self._pred_step_count = {}

        # VAL OVERLAY: one mp4 per val step, similarly inferred.
        for key, buffer in self.val_image_buffer.items():
            if not buffer:
                continue
            n_samples = self._val_step_count.get(key, 0)
            if n_samples <= 0:
                continue
            frames_per_sample = len(buffer) // n_samples
            if frames_per_sample <= 0:
                continue
            out_dir = os.path.join(
                self.video_dir(),
                f"epoch_{self.trainer.current_epoch}",
                str(get_embodiment(key)),
            )
            os.makedirs(out_dir, exist_ok=True)
            for s in range(n_samples):
                frames = torch.stack(
                    buffer[s * frames_per_sample : (s + 1) * frames_per_sample]
                )
                tvio.write_video(
                    os.path.join(out_dir, f"validation_video_{s}.mp4"),
                    frames,
                    fps=self.VAL_FPS,
                    video_codec="h264",
                )
            self.val_counter[key] = 0
            self.val_image_buffer[key] = []
        self._val_step_count = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _reproject_actions_to_head_t(
        actions_slice: torch.Tensor,
        head_pose_chunk: torch.Tensor,
        pixel_idx: int,
    ) -> torch.Tensor:
        """Rotate an action chunk from the H_0 head frame to the H_t head frame
        so the arrow lands on the hand as the head moves during the sample
        window.

        Actions in ``actions_slice`` are 12D ``[L xyz L ypr R xyz R ypr]`` in
        the frame-0 head frame (the data pipeline's
        ``ActionChunkCoordinateFrameTransform`` target). This helper composes
        ``T = H_t^-1 @ H_0`` and applies it to the left/right XYZ blocks (dims
        0:3 and 6:9) — YPR blocks are left untouched because the WAM viz uses
        ``mode="traj"`` which only reads XYZ.

        Args:
            actions_slice: (B, T_chunk, 12) — action slice in H_0 frame.
            head_pose_chunk: (B, cam_horizon, 7) xyzwxyz — per-pixel head pose
                chunk. ``[:, 0]`` is H_0, ``[:, pixel_idx]`` is H_t.
            pixel_idx: index into ``head_pose_chunk`` for the current pixel.

        Returns: (B, T_chunk, 12) with L_xyz / R_xyz reprojected; YPR unchanged.
        """
        from egomimic.utils.pose_utils import _xyzwxyz_to_matrix

        was_tensor = isinstance(actions_slice, torch.Tensor)
        a = (
            actions_slice.detach().cpu().numpy()
            if was_tensor
            else np.asarray(actions_slice)
        )
        hpc = (
            head_pose_chunk.detach().cpu().numpy()
            if isinstance(head_pose_chunk, torch.Tensor)
            else np.asarray(head_pose_chunk)
        )
        B, T_chunk, D = a.shape
        result = a.copy()
        for b in range(B):
            H_0_mat = _xyzwxyz_to_matrix(hpc[b, 0][None])[0]  # (4, 4)
            H_t_mat = _xyzwxyz_to_matrix(hpc[b, pixel_idx][None])[0]  # (4, 4)
            try:
                T_ht_h0 = np.linalg.inv(H_t_mat) @ H_0_mat  # 4x4 h_0 -> h_t
            except np.linalg.LinAlgError:
                continue  # degenerate pose — leave this sample as-is
            for arm_start in (0, 6):  # L_xyz at 0:3, R_xyz at 6:9
                xyz = a[b, :, arm_start : arm_start + 3]  # (T_chunk, 3)
                xyz_h = np.concatenate(
                    [xyz, np.ones((xyz.shape[0], 1), dtype=xyz.dtype)], axis=1
                )
                xyz_t = (T_ht_h0 @ xyz_h.T).T[:, :3].astype(xyz.dtype)
                result[b, :, arm_start : arm_start + 3] = xyz_t
        if was_tensor:
            return (
                torch.from_numpy(result)
                .to(actions_slice.dtype)
                .to(actions_slice.device)
            )
        return result

    @staticmethod
    def _predicted_frames_all(video: torch.Tensor, hw=(360, 640)) -> np.ndarray:
        """(B, C, T, H, W) in [-1, 1] -> (B*T, Hd, Wd, 3) uint8, resized to the
        GT display resolution so the aspect ratio matches a real frame (the
        world model runs at 160x320)."""
        B, C, T, H, W = video.shape
        v = video.clamp(-1, 1).permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        v = F.interpolate(v, size=tuple(hw), mode="bilinear", align_corners=False)
        v = ((v + 1.0) * 127.5).to(torch.uint8).cpu()
        return v.permute(0, 2, 3, 1).numpy()  # (B*T, Hd, Wd, 3)
