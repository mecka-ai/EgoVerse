import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.io as tvio
from torchmetrics import MeanSquaredError

from egomimic.eval.eval_video import EvalVideo
from egomimic.eval.wam_rollout import (
    DEFAULT_SOURCE_FPS,
    WAM_VIDEO_FPS,
    video_frame_stride,
)
from egomimic.rldb.embodiment.embodiment import get_embodiment


def _is_rank0() -> bool:
    return (
        (not torch.distributed.is_available())
        or (not torch.distributed.is_initialized())
        or (torch.distributed.get_rank() == 0)
    )


class WAMEvalVideo(EvalVideo):
    """Evaluator for WAM (World-Action Model). Per embodiment it writes TWO
    separate validation videos per val sample — and when the val split is built
    with one full-episode window per episode (``egomimic.eval.wam_episode``,
    used by both the training-time evaluator and the offline driver) "per val
    sample" means **per episode**, so each mp4 spans a whole episode:

      1. ``validation_video_*.mp4`` — animated predicted vs GT **action**
         overlay drawn on the GT observation clip (via ``viz_func``); BC loss
         + action MSE metrics.
      2. ``predicted_video_*.mp4``  — the world model's **predicted future
         frames** (rolled out chunk-by-chunk by ``WAM.val_rollout``,
         VAE-decoded), with the SAME action overlay drawn on the imagined
         frames. Written from the frames forward_eval stashes on the algo as
         ``_eval_frames[eid]``.

    Rate contract: the data pipeline strides the 30 fps source x6, so the clip
    the model sees -- and the mp4 -- are 5 fps and real time. Actions are NOT
    strided; they stay at raw 30 Hz. One mp4 frame per displayed frame.

    Receding action overlay (both videos, GT green + predicted red): a chunk is
    ``K * 4`` displayed frames (K=2 -> 8) carrying ``num_action_per_block`` (48)
    actions at 30 Hz, so each displayed frame consumes 48 / 8 = 6 actions. At
    displayed frame ``j``, with ``(c, d) = divmod(j, 8)``, we draw actions
    ``[c*48 + 6d : c*48 + 48]``: the whole chunk at d=0, six fewer each frame,
    the final six at d=7, and zero exactly at the boundary where the next
    chunk's 48 appear. All 8 frames and all 48 actions are visible across the
    chunk's 1.6 s.

    Both mp4s of a given episode are written with the SAME frame count and the
    SAME fps (``wam_rollout.WAM_VIDEO_FPS``), so an episode of N predicted
    frames yields two N-frame clips of N / fps seconds that play in lockstep.
    """

    # Playback rate for BOTH mp4s — see ``wam_rollout.WAM_VIDEO_FPS`` (the
    # single source of truth). Both videos are written with the same frame
    # count per episode so they stay frame-synchronised side by side, and both
    # use the same rate so an N-frame clip always lasts N / fps seconds. Do not
    # reintroduce separate VAL_FPS / PREDICTED_FPS constants: two rates for two
    # videos of the same episode is what made the pair play at different speeds.
    VAL_FPS = WAM_VIDEO_FPS
    PREDICTED_FPS = WAM_VIDEO_FPS

    def __init__(
        self,
        viz_func=None,
        limit_val_batches: int = 400,
        rolling_mode: str = "tf",
        emit_video: bool = True,
        emit_metrics: bool = True,
        source_fps: float = DEFAULT_SOURCE_FPS,
        data_frame_stride: int = 1,
    ):
        """``rolling_mode`` is "tf" (recondition the sliding history on GT after
        every chunk) or "ar" (condition chunk 0 on the first GT frame, then on
        the model's own predictions forever). Read by the offline driver's
        ``_select_rolling_mode``; inert inside this class.

        ``emit_video`` / ``emit_metrics`` are the ONLY difference between the
        three eval configs (eval_wam_video / _metrics / _video_metrics). With
        emit_video False we skip the whole per-frame overlay render, which is the
        expensive part -- a metrics-only pass does no matplotlib work at all.
        """
        super().__init__(limit_val_batches=limit_val_batches)
        self.viz_func = viz_func
        mode = str(rolling_mode).lower()
        if mode not in ("tf", "ar"):
            raise ValueError(f"rolling_mode must be 'tf' or 'ar', got {rolling_mode!r}")
        self.rolling_mode = mode
        self.emit_video = bool(emit_video)
        self.emit_metrics = bool(emit_metrics)
        if not (self.emit_video or self.emit_metrics):
            raise ValueError(
                "emit_video and emit_metrics are both False -- this evaluator "
                "would do a full rollout and throw everything away."
            )
        # data_frame_stride MUST reflect the stride the data pipeline applied,
        # or video_frame_stride() decimates an already-5 fps clip a second time.
        # The offline driver calls set_source_fps() with the plan's value; the
        # training path passes it here (trainHydra sets it from the same plan).
        self.set_source_fps(source_fps, data_frame_stride)

    @property
    def teacher_force_rolling(self) -> bool:
        """Back-compat view of ``rolling_mode`` for the offline driver."""
        return self.rolling_mode == "tf"

    def set_source_fps(self, source_fps: float, data_frame_stride: int = 1) -> None:
        """Set the NATIVE prediction frame rate and derive the video subsample.

        The offline driver calls this with the rate read from the episode's zarr
        metadata (``wam_episode._source_fps``) so playback speed follows the data
        rather than a constant.
        """
        self.source_fps = float(source_fps)
        # MUST pass the data stride. The rollout predicts at
        # source_fps / data_frame_stride, so on the 5 fps pipeline the correct
        # playback stride is 1. Calling this with source_fps alone returned 6 and
        # decimated an already-5 fps clip a second time: a 320-frame full-episode
        # rollout landed in the mp4 as ceil(320/6) = 54 frames (10.8 s instead of
        # 64 s), which looked like the episode tiling had failed.
        self.data_frame_stride = max(1, int(data_frame_stride))
        self.frame_stride = video_frame_stride(self.source_fps, self.data_frame_stride)

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
            # Skipped entirely when the config only wants metrics -- the overlay
            # is one viz_func (matplotlib/PIL) call per displayed frame per
            # episode, which dwarfs the rollout itself.
            if not self.emit_video:
                pass
            elif clip_by_cam:
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

        if not self.emit_metrics:
            metrics = {}
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

        # ONE output frame per DISPLAYED frame. The data pipeline already
        # strided the clip to 5 fps, so writing one mp4 frame per displayed frame
        # makes the video real time and exactly as long as the span it covers.
        # (Upstream held each pixel frame FR = n_gt/(T_pix-1) output frames to
        # animate a 30 Hz recession on a 5 fps clip; that multiplies the video
        # length by FR, which is the "slowed down instead of subsampled" bug.)
        #
        # ACTION RECESSION, per the val-loop contract:
        #   a chunk is K * 4 displayed frames (K=2 -> 8) = H_actions actions
        #   (48) at raw 30 Hz, so each displayed frame consumes
        #   H_actions / (K*4) = 6 actions. At displayed frame j we draw
        #   actions [c*H + d*6 : c*H + H] where (c, d) = divmod(j, 8): the whole
        #   48-action chunk at d=0, six fewer each frame, the final 6 at d=7,
        #   and zero exactly at the boundary where the next chunk's 48 appear.
        #   All 8 frames and all 48 actions are visible across the chunk's 1.6 s.
        chunk_disp = max(1, int(K) * 4)  # displayed frames / chunk
        actions_per_disp = max(1, H_actions // chunk_disp)  # 48 // 8 = 6 @ 30 Hz
        n_disp = max(0, T_pix - 1)  # frame 0 is the anchor

        # Keep validation_video and predicted_video the SAME length so the pair
        # is frame-synchronised. The rolling rollout emits K*num_chunks latents ->
        # 4*K*num_chunks displayed frames, which can be < T_pix - 1 when the
        # episode length is not a multiple of 4*K.
        pred_video_probe = getattr(self.model, "_eval_frames", {}).get(embodiment_id)
        if pred_video_probe is not None:
            n_disp = min(n_disp, int(pred_video_probe.shape[2]))

        # Legacy 30 fps data (no data stride) still needs the real-time
        # subsample, and its PHASE must be global across the episode's windows:
        # restarting at 0 each window leaves kept indices off the uniform grid
        # and the playback jitters. frame_stride is 1 on the 5 fps pipeline, so
        # this is a no-op there.
        stride = max(1, int(getattr(self, "frame_stride", 1)))
        if not hasattr(self, "_native_offset"):
            self._native_offset = {}
        okey = (embodiment_id, self._episode_key(_batch))
        offset = self._native_offset.get(okey, 0)
        first = (-offset) % stride
        keep_j = list(range(first, n_disp, stride))
        self._native_offset[okey] = offset + n_disp

        def _action_window(j):
            """(start, end) into the FULL action tensor for displayed frame j.

            Both ends are LEAD-shifted by one displayed frame
            (``actions_per_disp`` = 6 actions @ 30 Hz). Displayed frame j shows
            raw frame 6*(j+1) -- clip index 0 is the anchor, which is never
            drawn -- so the first arrow must be action 6*(j+1), not 6*j.
            Without the shift every overlay trailed the video by exactly one
            displayed frame (0.2 s), which is visible as arrows lagging the
            palm. The end shifts with it so the count still runs
            48,42,...,6 and resets to 48 at the chunk boundary.
            """
            c = int(j) // chunk_disp
            lead = actions_per_disp
            return (int(j) + 1) * lead, (c + 1) * H_actions + lead

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

        # ---- Video 1: receding action overlay on the GT (5 fps) clip -------
        per_frame_ims = []
        for j in keep_j:
            pixel_idx = j + 1  # frame 0 of the clip is the anchor
            start, end = _action_window(j)

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
        pred_video_t = pred_video_probe
        if _is_rank0() and pred_video_t is not None:
            image_key = some_ck
            ref_H = clip_by_cam[image_key].shape[-2]
            ref_W = clip_by_cam[image_key].shape[-1]
            # Select exactly the instants the GT-overlay timeline kept (same
            # index grid, see keep_j above), so the two mp4s have
            # identical frame counts and index i is the same instant in both.
            # Selecting before the upscale also skips 5/6 of the interpolate.
            pred_video_t = pred_video_t.index_select(
                2, torch.tensor(keep_j, device=pred_video_t.device)
            )
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
            # ``i`` indexes the SELECTED frames in pv; ``p`` is the frame's index
            # on the native prediction timeline, which is what the action-overlay
            # arithmetic below must use so the arrows keep native granularity.
            for i, j in enumerate(keep_j):
                # ``_eval_frames`` already dropped the anchor frame
                # (WAM.val_rollout: viz_video = pred_frames[:, :, 1:]), so j=0
                # corresponds to pixel_idx=1 (first predicted frame). Both videos
                # use the SAME j grid and the SAME action window, so index i is
                # the same instant with the same arrows in both files.
                pixel_idx = j + 1
                start, end = _action_window(j)

                gt_slice, pred_slice = _slices_for(start, end)
                gt_slice, pred_slice = _reproject(gt_slice, pred_slice, pixel_idx)
                _batch_p = dict(_batch)
                _batch_p[image_key] = pv[:, i]  # (B, C, H, W)
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
        if not hasattr(self, "_ep_open"):
            # ONE open clip per (video kind, embodiment): [episode_hash, frames].
            # The val split walks each episode start-to-finish (see
            # egomimic.eval.wam_episode), so frames arrive grouped by episode and
            # a clip can be written the moment the episode_hash changes. Writing
            # eagerly matters: a 1936-frame episode of 360x640 RGB is ~1.3 GB per
            # video kind, so holding all 5 val episodes until on_validation_end
            # would need ~13 GB of frame buffers.
            #
            # This replaces the old "split the flat buffer into n_samples equal
            # slices" scheme, which assumed one mp4 per val STEP and cannot
            # express episodes with differing window counts.
            self._ep_open: dict[tuple[str, int], list] = {}
            self._ep_ordinal: dict[tuple[str, int], int] = {}
            # {(eid, episode): native frames emitted so far} -> keeps the
            # real-time subsample on one uniform grid across the episode's
            # windows (see _animate_overlays).
            self._native_offset: dict[tuple[int, str], int] = {}

    @staticmethod
    def _episode_key(_batch) -> str:
        """Stable per-episode identity for grouping frames into one mp4."""
        ep = _batch.get("episode_hash") if isinstance(_batch, dict) else None
        if isinstance(ep, (list, tuple)):
            return str(ep[0]) if len(ep) else "episode_0"
        return "episode_0" if ep is None else str(ep)

    def _append_episode_frames(self, stem: str, eid, episode: str, frames) -> None:
        """Buffer ``frames`` into the open clip, flushing on episode change."""
        key = (stem, eid)
        cur = self._ep_open.get(key)
        if cur is not None and cur[0] != episode:
            self._flush_episode(stem, eid)
            cur = None
        if cur is None:
            cur = [episode, []]
            self._ep_open[key] = cur
        cur[1].extend(torch.from_numpy(frames))

    def _flush_episode(self, stem: str, eid) -> None:
        """Write the open ``<stem>`` clip for ``eid`` and free its frames."""
        key = (stem, eid)
        cur = self._ep_open.pop(key, None)
        if cur is None or not cur[1]:
            return
        episode, frames = cur
        # Ordinal is per (kind, embodiment) and both kinds are appended in the
        # same step order, so predicted_video_<s> and validation_video_<s> are
        # always the same episode.
        s = self._ep_ordinal.get(key, 0)
        self._ep_ordinal[key] = s + 1
        out_dir = os.path.join(
            self.video_dir(),
            f"epoch_{self.trainer.current_epoch}",
            str(get_embodiment(eid)),
        )
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{stem}_{s}.mp4")
        stacked = torch.stack(frames)
        frames.clear()
        tvio.write_video(path, stacked, fps=WAM_VIDEO_FPS, video_codec="h264")
        print(
            f"[WAMEvalVideo] {stem}_{s}.mp4: episode {episode} "
            f"{stacked.shape[0]} frames @ {WAM_VIDEO_FPS} fps = "
            f"{stacked.shape[0] / WAM_VIDEO_FPS:.2f}s -> {path}",
            flush=True,
        )

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

        # RANK-GUARD: only rank 0 buffers/writes mp4s (h264 encode + shared-FS
        # writes on every rank caused per-rank divergence and NCCL watchdog
        # timeouts). Other ranks still log metrics below.
        rank0 = _is_rank0()

        for key, images in images_dict.items():
            if rank0:
                self._append_episode_frames(
                    "validation_video", key, self._episode_key(batch.get(key)), images
                )
            self._val_step_count[key] = self._val_step_count.get(key, 0) + 1

        for eid, video in getattr(self.model, "_eval_frames", {}).items():
            if video is None or not self.emit_video:
                continue
            # Prefer the action-overlay-annotated pred frames built inside
            # compute_metrics_and_viz. Fall back to raw upscaled pred frames
            # if the overlay cache is missing (non-rank0, or no clip batch).
            overlay = getattr(self, "_pred_overlay_cache", {}).pop(eid, None)
            if rank0:
                if overlay is not None:
                    frames = overlay
                else:
                    # Same real-time subsample as the overlay path.
                    frames = self._predicted_frames_all(
                        video, stride=max(1, int(getattr(self, "frame_stride", 1)))
                    )  # (B*T, Hd, Wd, 3)
                self._append_episode_frames(
                    "predicted_video", eid, self._episode_key(batch.get(eid)), frames
                )
            self._pred_step_count[eid] = self._pred_step_count.get(eid, 0) + 1

        # add_dataloader_idx=False: keep canonical Valid/ chart names (see the
        # base EvalVideo for rationale). Guarded because log_dict with
        # sync_dist=True on an empty dict is a pointless collective on a
        # video-only config, and collectives that some ranks skip deadlock.
        if metrics:
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
        # Flush whatever episode was still open when the val loop ended (the
        # per-episode writes themselves already happened in on_validation_step
        # as each episode finished).
        if _is_rank0():
            for stem, eid in list(self._ep_open.keys()):
                self._flush_episode(stem, eid)
        self._reset_buffers()

    def _reset_buffers(self) -> None:
        self._ep_open = {}
        self._ep_ordinal = {}
        self._native_offset = {}
        self._pred_buffer = {}
        self._pred_step_count = {}
        self._val_step_count = {}
        for key in list(self.val_image_buffer.keys()):
            self.val_image_buffer[key] = []
            self.val_counter[key] = 0

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
    def _predicted_frames_all(
        video: torch.Tensor, hw=(360, 640), stride: int = 1
    ) -> np.ndarray:
        """(B, C, T, H, W) in [-1, 1] -> (B*T, Hd, Wd, 3) uint8, resized to the
        GT display resolution so the aspect ratio matches a real frame (the
        world model runs at 160x320). ``stride`` applies the same real-time
        subsample the overlay path uses (see ``video_frame_stride``)."""
        if stride > 1:
            video = video[:, :, ::stride]
        B, C, T, H, W = video.shape
        v = video.clamp(-1, 1).permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        v = F.interpolate(v, size=tuple(hw), mode="bilinear", align_corners=False)
        v = ((v + 1.0) * 127.5).to(torch.uint8).cpu()
        return v.permute(0, 2, 3, 1).numpy()  # (B*T, Hd, Wd, 3)
