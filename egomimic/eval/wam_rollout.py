"""Shared WAM (World-Action Model) rolling rollout + video-geometry helpers.

This module is the SINGLE implementation of the dreamzero-style rolling
prediction used by BOTH WAM validation paths:

  * training-time  — ``WAMModel.sample_rolling`` (called by ``WAM.val_rollout``,
    rendered by ``egomimic.eval.eval_wam.WAMEvalVideo``), and
  * offline         — ``egomimic.eval.eval_dreamzero`` (the Modal offline driver
    ``egomimic/modal/offline_val_wam.py`` and the val sweep
    ``egomimic/modal/wam_val_sweep.py``).

Before this module the two paths had independent copies of the rolling loop
(``WAMModel.sample_rolling`` vs ``eval_dreamzero._sample_rolling_{tf,ar}``),
which is how they drifted apart. Everything time-related — the teacher-forcing
context window, the number of rolling chunks, the pixel<->latent stride and the
mp4 frame rate — now lives here.

--------------------------------------------------------------------------
Latent / pixel timeline
--------------------------------------------------------------------------
The Wan VAE is a causal 3D VAE with 4x temporal compression that treats the
first frame specially (``VideoVAE38_.encode``: ``iter_ = 1 + (t - 1) // 4``):

    latent 0      <- pixel frame 0                     ("anchor")
    latent i >= 1 <- pixel frames 4i-3 .. 4i           (LATENT_PIXEL_STRIDE=4)

so a clip of ``T`` pixel frames encodes to ``F = 1 + (T - 1) // 4`` latents and
decoding ``F`` latents yields ``1 + (F - 1) * 4`` pixel frames. Because the
encoder streams the temporal axis with a causal feature cache, ``T`` may be as
long as a whole episode — which is what makes the full-episode rolling below
possible with bounded memory.

--------------------------------------------------------------------------
Rolling layout (per DiT forward)
--------------------------------------------------------------------------
``K = dit.num_frame_per_block`` latents are predicted per DiT forward and
``F_total = model.num_video_frames`` is the DiT window length, so the window is

    video   = [ctx_0 .. ctx_{n_hist-1} | noisy_0 .. noisy_{K-1}]   (F_total)
    clean_x = [ctx_0 .. ctx_{n_hist-1} | video_last_K]
    ts_v    = [  0   ..      0         |   t    ..     t     ]

with ``n_hist = F_total - K``. In ``CausalWanModel`` the window's frame 0 is the
"first image" conditioning token and frames 1.. form ``num_image_blocks =
(F_total - 1) // K`` blocks of K frames; image block b attends to action block b,
so the K noisy frames always occupy the LAST image block and the actions that
belong to them are the LAST ``num_action_per_block`` action tokens.

--------------------------------------------------------------------------
Teacher-forcing boundary (the invariant this module enforces)
--------------------------------------------------------------------------
Predicted chunk ``k`` (0-based) covers latents ``[k*K + 1, k*K + K]`` — latent 0
is the GT anchor and is never predicted. Its conditioning context is therefore

    chunk_start = k * K + 1
    ctx_end     = chunk_start            (EXCLUSIVE)
    ctx_start   = ctx_end - n_hist

i.e. the context is exactly the ``n_hist`` latents immediately PRECEDING the
chunk, ending on the last latent of chunk ``k-1``: never earlier, never
overlapping forward into chunk ``k``. ``assert_tf_alignment`` checks
``ctx_end == chunk_start`` for every chunk of a full episode.

The old code expressed this as an incremental slide
(``history = cat([history[:, :, K:], gt[step*K+1 : step*K+K+1]])``). That
happens to be algebraically equivalent *within one window*, but it is stateful:
it re-derives the boundary from the previous iteration instead of from the chunk
index, so any restart of the loop (e.g. per-window in a multi-window episode
walk) silently rewinds the context to the window's own frame 0 while the chunk
index keeps advancing — the "teacher forcing conditions on an earlier window"
bug. Gathering the context by absolute latent index, as below, makes that class
of error impossible to express.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# --------------------------------------------------------------------------
# Time constants — the single source of truth for every WAM mp4.
# --------------------------------------------------------------------------
# Playback rate for BOTH `predicted_video_*.mp4` and `validation_video_*.mp4`.
# The two videos are written with the same frame count per episode so they stay
# frame-synchronised side by side, and both are written at this rate. Frames are
# SUBSAMPLED from the native prediction rate to reach it (see
# ``video_frame_stride``) so playback is real time, never slow motion.
WAM_VIDEO_FPS = 5

# Pixel frames represented by one non-anchor latent frame (Wan VAE 4x temporal).
LATENT_PIXEL_STRIDE = 4

# Frame rate of the SOURCE frames the rollout predicts at. This is NOT a display
# choice: ``VideoVAE38_.encode`` consumes ``x[:, :, 1+4*(i-1) : 1+4*i]`` per
# latent and ``decode`` emits 4 DISTINCT pixel frames per latent, at whatever
# temporal spacing the frames were fed in with. This fork's WAM keymap reads
# ``cam_horizon`` CONSECUTIVE frames from a 30 fps mecka zarr (upstream's
# ``video_stride=6``, which pre-subsampled 30 -> 5 fps before the VAE, is not
# ported), so the model's native prediction granularity is one frame per 30 fps
# source frame. Overridden per-episode from the zarr metadata where available.
DEFAULT_SOURCE_FPS = 30.0


def video_frame_stride(source_fps: float = DEFAULT_SOURCE_FPS) -> int:
    """Keep every Nth predicted frame so the mp4 plays back in REAL TIME.

    The model predicts at ``source_fps`` granularity (see ``DEFAULT_SOURCE_FPS``)
    but the mp4s are written at ``WAM_VIDEO_FPS``. Writing every predicted frame
    into a 5 fps container turns 64.8 s of episode into 387 s of video — a 6x
    slowdown. Subsampling the ASSEMBLED frames by this stride gives real-time
    playback instead.

    This is deliberately a video-assembly concern only: the rollout, the
    teacher-forcing conditioning, the chunk/latent geometry and the per-frame
    action predictions all stay at native ``source_fps`` granularity, and the
    metrics are computed over ALL predicted frames. Only which frames get
    encoded into the container changes.
    """
    return max(1, int(round(float(source_fps) / float(WAM_VIDEO_FPS))))


def latents_to_pixels(num_latents: int) -> int:
    """Pixel-frame count produced by VAE-decoding ``num_latents`` latents."""
    if num_latents <= 0:
        return 0
    return 1 + (num_latents - 1) * LATENT_PIXEL_STRIDE


def pixels_to_latents(num_pixels: int) -> int:
    """Latent count produced by VAE-encoding ``num_pixels`` pixel frames."""
    if num_pixels <= 0:
        return 0
    return 1 + (num_pixels - 1) // LATENT_PIXEL_STRIDE


# --------------------------------------------------------------------------
# Index arithmetic
# --------------------------------------------------------------------------
def chunk_bounds(chunk_idx: int, K: int) -> tuple[int, int]:
    """Latent range ``[start, end)`` of predicted chunk ``chunk_idx``.

    Latent 0 is the GT anchor (never predicted), so chunk k covers latents
    ``k*K + 1 .. k*K + K``.
    """
    start = chunk_idx * K + 1
    return start, start + K


def tf_context_indices(chunk_idx: int, K: int, n_hist: int) -> list[int]:
    """Absolute latent indices of the conditioning context for chunk ``k``.

    Returns exactly ``n_hist`` indices ending at ``chunk_start - 1``. Negative
    indices — only reachable for the first ``ceil(n_hist / K)`` chunks of an
    episode, where no earlier GT exists — are clamped to 0, edge-repeating the
    episode anchor. This padding is confined to the true start of the episode:
    once the rollout has advanced past it every context slot is a real,
    strictly-increasing latent index.
    """
    chunk_start, _ = chunk_bounds(chunk_idx, K)
    ctx_end = chunk_start
    ctx_start = ctx_end - n_hist
    return [max(0, i) for i in range(ctx_start, ctx_end)]


def assert_tf_alignment(num_chunks: int, K: int, n_hist: int) -> str:
    """Unit-style check of the TF boundary across EVERY chunk of an episode.

    Verifies, for every chunk k in ``range(num_chunks)``:
      1. the context is ``n_hist`` long,
      2. ``ctx_end == chunk_start`` — conditioning stops exactly where the
         predicted chunk starts (the last context latent is the last latent of
         chunk k-1): never earlier, never overlapping forward,
      3. the context never contains an index inside chunk k or later,
      4. contexts advance monotonically by K from one chunk to the next (no
         rewind — the bug this guards against).

    Returns a one-line summary suitable for logging. Raises AssertionError on
    any violation.
    """
    prev_end = None
    for k in range(num_chunks):
        chunk_start, chunk_end = chunk_bounds(k, K)
        idx = tf_context_indices(k, K, n_hist)
        assert (
            len(idx) == n_hist
        ), f"chunk {k}: context length {len(idx)} != n_hist {n_hist}"
        ctx_end = idx[-1] + 1
        assert ctx_end == chunk_start, (
            f"chunk {k}: context ends at latent {ctx_end} but the chunk starts "
            f"at {chunk_start} — teacher forcing is misaligned by "
            f"{chunk_start - ctx_end} latents"
        )
        assert max(idx) < chunk_start, (
            f"chunk {k}: context {idx} overlaps forward into the predicted "
            f"chunk [{chunk_start}, {chunk_end})"
        )
        assert idx == sorted(idx), f"chunk {k}: context indices not monotonic: {idx}"
        if prev_end is not None:
            assert ctx_end == prev_end + K, (
                f"chunk {k}: context end {ctx_end} did not advance by K={K} "
                f"from the previous chunk's {prev_end} — the conditioning "
                f"window moved backwards or skipped ahead"
            )
        prev_end = ctx_end
    last_start, last_end = chunk_bounds(max(0, num_chunks - 1), K)
    return (
        f"TF alignment OK: {num_chunks} chunks, K={K}, n_hist={n_hist}; "
        f"ctx_end == chunk_start for every chunk; last chunk covers latents "
        f"[{last_start}, {last_end}) with context "
        f"[{last_start - n_hist}, {last_start})"
    )


@dataclass(frozen=True)
class RollingGeometry:
    """Resolved geometry of one rolling rollout."""

    K: int  # latents predicted per DiT forward (dit.num_frame_per_block)
    F_total: int  # DiT window length in latents (model.num_video_frames)
    n_hist: int  # context latents per window = F_total - K
    F_gt: int  # GT latents available for this episode/clip
    num_chunks: int  # rolling steps actually run
    num_action_per_block: int  # action tokens per predicted chunk

    @property
    def pred_latents(self) -> int:
        return self.num_chunks * self.K

    @property
    def pred_pixel_frames(self) -> int:
        """Pixel frames in the emitted video (anchor recon dropped)."""
        return latents_to_pixels(1 + self.pred_latents) - 1

    def video_frames(self, source_fps: float = DEFAULT_SOURCE_FPS) -> int:
        """Frames actually written to the mp4 after real-time subsampling."""
        stride = video_frame_stride(source_fps)
        return (self.pred_pixel_frames + stride - 1) // stride

    def duration_s(self, source_fps: float = DEFAULT_SOURCE_FPS) -> float:
        """Playback length of the mp4. Equals the real elapsed time of the
        predicted span (``pred_pixel_frames / source_fps``) up to one frame."""
        return self.video_frames(source_fps) / WAM_VIDEO_FPS

    def summary(self, source_fps: float = DEFAULT_SOURCE_FPS) -> str:
        return (
            f"K={self.K} F_total={self.F_total} n_hist={self.n_hist} "
            f"F_gt={self.F_gt} chunks={self.num_chunks} "
            f"pred_latents={self.pred_latents} "
            f"pred_frames={self.pred_pixel_frames} @ {source_fps:g} fps native "
            f"-> stride {video_frame_stride(source_fps)} -> "
            f"{self.video_frames(source_fps)} frames @ {WAM_VIDEO_FPS} fps = "
            f"{self.duration_s(source_fps):.2f}s "
            f"(real time {self.pred_pixel_frames / source_fps:.2f}s)"
        )


def resolve_geometry(model, F_gt: int, num_chunks: int | None) -> RollingGeometry:
    """Derive the rolling geometry from the model config + available GT."""
    K = int(getattr(model.dit, "num_frame_per_block", 1))
    F_total = int(model.num_video_frames)
    n_hist = F_total - K
    if n_hist < 1:
        raise ValueError(
            f"num_video_frames ({F_total}) must exceed num_frame_per_block ({K}) "
            "so at least one conditioning latent remains."
        )
    num_action_per_block = int(
        getattr(
            model.dit,
            "num_action_per_block",
            model.action_horizon // max(1, (F_total - 1) // K),
        )
    )
    # Roll for as many whole K-latent chunks as the GT timeline supports. The
    # anchor (latent 0) is not predicted, hence (F_gt - 1).
    max_chunks = max(1, (F_gt - 1) // K)
    resolved = max_chunks if num_chunks is None else min(int(num_chunks), max_chunks)
    return RollingGeometry(
        K=K,
        F_total=F_total,
        n_hist=n_hist,
        F_gt=F_gt,
        num_chunks=max(1, resolved),
        num_action_per_block=num_action_per_block,
    )


# --------------------------------------------------------------------------
# Rollout
# --------------------------------------------------------------------------
def _denoise_chunk(
    model, sched, video, action, history, n_hist, F_total, seq_len, context, state, emb
):
    """Run the flow-match scheduler over one chunk.

    Updates the K noisy positions ``video[:, :, n_hist:]`` while pinning the
    context positions ``video[:, :, :n_hist]`` to ``history`` after every step.
    ``clean_x = cat(history, current_noisy_estimate)`` reproduces the causal
    teacher-forcing prefix the DiT is trained with.
    """
    for t in sched.timesteps:
        ts_v = torch.zeros(video.shape[0], F_total, device=video.device)
        ts_v[:, n_hist:] = t
        ts_a = t.to(video.device).expand(video.shape[0], action.shape[1])
        clean_x = torch.cat([history, video[:, :, n_hist:]], dim=2)
        v_vel, a_vel = model.dit(
            video,
            ts_v,
            ts_a,
            context,
            seq_len,
            action=action,
            state=state,
            embodiment_id=emb,
            clean_x=clean_x,
        )
        video = sched.step(v_vel, t, video)
        video[:, :, :n_hist] = history  # context stays clean
        action = sched.step(a_vel, t, action)
    return video[:, :, n_hist:], action


@torch.no_grad()
def rollout_episode(
    model,
    data: dict,
    teacher_force: bool = True,
    num_chunks: int | None = None,
    log=None,
    return_geometry: bool = False,
):
    """Roll the world-action model across a WHOLE clip, chunk by chunk.

    ``data["video"]`` may be any length — one training window or an entire
    episode. It is VAE-encoded ONCE into a single continuous latent timeline,
    and every chunk's conditioning context is gathered from that timeline by
    absolute index (see the module docstring), so the context marches strictly
    forward for the full length of the episode.

    Args:
        model: a ``WAMModel``.
        data: ``{"video", "state", "action", "embodiment_id"}`` — the same dict
            ``WAM._to_wam_data`` / ``WAM.val_rollout`` build.
        teacher_force: True  -> dreamzero Fig-14a GT teacher forcing: chunk k
            conditions on GT latents. False -> fully autoregressive: chunk k
            conditions on the model's own earlier predictions (only latent 0 is
            GT). Both share the identical index arithmetic; the flag only
            chooses which timeline the context is read from.
        num_chunks: cap on rolling steps (``None`` = as many as the clip
            supports). Also honoured as ``WAM.val_rollout_chunks``.
        log: optional logger for the geometry / TF-alignment lines.
        return_geometry: also return the ``RollingGeometry``.

    Returns:
        ``(pred_actions, frames)`` or ``(pred_actions, frames, geometry)``:
        ``pred_actions`` is ``(B, num_chunks * num_action_per_block, action_dim)``
        and ``frames`` is ``(B, C, 1 + num_chunks*K*4, H, W)`` in [-1, 1] — the
        VAE decode of ``[GT anchor] + [all predicted latents]``.
    """
    from egomimic.models.wam_nets import FlowMatchScheduler

    sched = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
    sched.set_timesteps(model.num_inference_steps, training=False)

    gt_latents = model._encode(data["video"])  # (B, Cz, F_gt, h, w)
    B, Cz, F_gt, h, w = gt_latents.shape
    device, dtype = gt_latents.device, gt_latents.dtype

    geo = resolve_geometry(model, F_gt, num_chunks)
    K, F_total, n_hist = geo.K, geo.F_total, geo.n_hist

    # Fail loudly if the boundary arithmetic is wrong for ANY chunk of this
    # episode, before spending inference time on it.
    alignment = assert_tf_alignment(geo.num_chunks, K, n_hist)
    if log is not None:
        log.info(f"[wam_rollout] {geo.summary()} teacher_force={teacher_force}")
        log.info(f"[wam_rollout] {alignment}")

    seq_len = F_total * (h // 2) * (w // 2)
    state, _ = model._prep_state_action(data)
    context = model._zero_context(B, device, dtype)
    emb = model._emb_ids(data, B, device)

    # The timeline the context is gathered from. Teacher forcing reads GT;
    # autoregressive rolling overwrites each chunk's slots with the model's own
    # predictions as it goes, so later contexts see predictions. Latent 0 (the
    # anchor) is GT in both modes.
    timeline = gt_latents if teacher_force else gt_latents.clone()

    pred_latents = []
    pred_actions_chunks = []
    for k in range(geo.num_chunks):
        ctx_idx = torch.tensor(
            tf_context_indices(k, K, n_hist), dtype=torch.long, device=device
        )
        history = timeline.index_select(2, ctx_idx)  # (B, Cz, n_hist, h, w)

        video = torch.randn(B, Cz, F_total, h, w, device=device, dtype=dtype)
        video[:, :, :n_hist] = history
        action = torch.randn(
            B, model.action_horizon, model.action_dim, device=device, dtype=dtype
        )

        new_latents, new_action = _denoise_chunk(
            model,
            sched,
            video,
            action,
            history,
            n_hist,
            F_total,
            seq_len,
            context,
            state,
            emb,
        )
        pred_latents.append(new_latents)
        # The K noisy frames sit in the LAST image block, so their actions are
        # the last num_action_per_block action tokens (see module docstring).
        pred_actions_chunks.append(new_action[:, -geo.num_action_per_block :])

        if not teacher_force:
            chunk_start, chunk_end = chunk_bounds(k, K)
            hi = min(chunk_end, F_gt)
            if hi > chunk_start:
                timeline[:, :, chunk_start:hi] = new_latents[:, :, : hi - chunk_start]

    pred_actions = torch.cat(pred_actions_chunks, dim=1)
    all_latents = torch.cat([gt_latents[:, :, :1]] + pred_latents, dim=2)
    frames = model.vae.decode(all_latents)
    if return_geometry:
        return pred_actions, frames, geo
    return pred_actions, frames


# --------------------------------------------------------------------------
# Full-episode rolling across consecutive dataloader windows
# --------------------------------------------------------------------------
class EpisodeRoller:
    """Continuous rolling across the consecutive windows of ONE episode.

    ``rollout_episode`` above rolls whatever clip it is handed. For a real mecka
    episode that clip cannot be handed over in one piece: the val episodes are
    1945-2245 frames at 30 fps and the dataloader decodes JPEG frames to float
    (3, 360, 640), so one full-episode sample would be ~12 GB in a worker. The
    val split therefore tiles each episode into ``cam_horizon``-length windows
    (stride ``cam_horizon - 1``, so window j's frame 0 is window j-1's last
    frame) and this class drives them.

    What makes the result a FULL-EPISODE rollout rather than a sequence of
    independent ~3 s rollouts is that the conditioning context lives on the
    EPISODE's latent timeline and is carried across window boundaries:

        window j, local chunk 0  ->  conditions on the GT latents that END
                                     window j-1, NOT on window j's own frame 0
                                     repeated n_hist times.

    Re-bootstrapping the context at every window (which is what calling
    ``rollout_episode`` per window would do) is exactly the "teacher forcing
    conditions on an earlier window / the context goes backwards" bug: the
    predicted chunk index keeps marching forward through the episode while the
    context snaps back to the current window's first frame, so the model is
    re-anchored to a frozen frame ~3 s in the past every 16 frames.

    Latent bookkeeping (all indices are GLOBAL, i.e. on the episode timeline):
      * ``_tail``      — the last ``n_hist`` REALIZED latents, the context for
                         the next chunk.
      * ``_tail_end``  — exclusive global latent index one past ``_tail``'s last
                         entry. The invariant asserted for every chunk of every
                         window is ``_tail_end == chunk_start``.

    Window j contributes GT latents ``1 + j*(F_w - 1) .. (j+1)*(F_w - 1)`` to the
    global timeline (its latent 0 re-encodes the previous window's last frame),
    so ``_tail_end`` advances by exactly K per chunk with no gaps or overlaps.
    """

    def __init__(self, model, teacher_force: bool = True, log=None):
        self.model = model
        self.teacher_force = bool(teacher_force)
        self.log = log
        self.episode_key: str | None = None
        self._tail: torch.Tensor | None = None
        self._tail_end: int = 0
        self._windows: int = 0
        self._chunks: int = 0
        self._pixel_frames: int = 0
        self._geo: RollingGeometry | None = None

    # -- episode lifecycle --------------------------------------------------
    def begin_episode(self, episode_key: str) -> None:
        """Start a new episode: drop the carried context and the counters."""
        self.episode_key = episode_key
        self._tail = None
        self._tail_end = 0
        self._windows = 0
        self._chunks = 0
        self._pixel_frames = 0
        self._geo = None

    @property
    def is_active(self) -> bool:
        return self._tail is not None

    def stats(self) -> dict:
        return {
            "episode": self.episode_key,
            "windows": self._windows,
            "chunks": self._chunks,
            "pixel_frames": self._pixel_frames,
            "latents_realized": self._tail_end,
            "duration_s": self._pixel_frames / WAM_VIDEO_FPS,
        }

    # -- one window ---------------------------------------------------------
    @torch.no_grad()
    def roll_window(self, data: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll every chunk of ONE window, continuing the episode's context.

        Returns ``(pred_actions, frames)`` for THIS window only:
        ``pred_actions`` is ``(B, n_local_chunks * num_action_per_block, D)`` and
        ``frames`` is ``(B, C, n_local_chunks*K*4, H, W)`` — the newly predicted
        pixel frames with the VAE recon of the window's anchor dropped, so
        concatenating the per-window results across an episode yields exactly
        one full-episode clip with no duplicated frames at the seams.
        """
        from egomimic.models.wam_nets import FlowMatchScheduler

        model = self.model
        sched = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        sched.set_timesteps(model.num_inference_steps, training=False)

        win_lat = model._encode(data["video"])  # (B, Cz, F_w, h, w)
        B, Cz, F_w, h, w = win_lat.shape
        device, dtype = win_lat.device, win_lat.dtype

        geo = resolve_geometry(model, F_w, None)
        K, F_total, n_hist = geo.K, geo.F_total, geo.n_hist
        self._geo = geo

        if self._tail is None:
            # Episode start: no earlier GT exists, so edge-repeat the anchor —
            # the same clamping ``tf_context_indices`` applies, and the ONLY
            # place in the episode where the context is padded.
            self._tail = win_lat[:, :, :1].repeat(1, 1, n_hist, 1, 1)
            self._tail_end = 1  # global latent 0 is realized (the GT anchor)
            if self.log is not None:
                self.log.info(
                    f"[wam_rollout] episode {self.episode_key}: {geo.summary()} "
                    f"teacher_force={self.teacher_force}; "
                    f"{assert_tf_alignment(geo.num_chunks, K, n_hist)}"
                )
        elif self._tail.shape[2] != n_hist:
            raise RuntimeError(
                f"carried context has {self._tail.shape[2]} latents but this "
                f"window needs n_hist={n_hist} — the model geometry changed "
                "mid-episode"
            )

        seq_len = F_total * (h // 2) * (w // 2)
        state, _ = model._prep_state_action(data)
        context = model._zero_context(B, device, dtype)
        emb = model._emb_ids(data, B, device)

        local_preds = []
        local_actions = []
        for c in range(geo.num_chunks):
            chunk_start = self._tail_end
            # THE invariant: conditioning ends exactly where the chunk begins.
            # Holds across window boundaries because ``_tail`` is never rebuilt
            # from the current window's frame 0 after the first window.
            assert self._tail_end == chunk_start, (
                f"episode {self.episode_key} window {self._windows} chunk {c}: "
                f"context ends at global latent {self._tail_end} but the chunk "
                f"starts at {chunk_start}"
            )
            assert (chunk_start - 1) % K == 0, (
                f"episode {self.episode_key} window {self._windows} chunk {c}: "
                f"chunk start {chunk_start} is not on a K={K} boundary — the "
                "global chunk grid drifted"
            )
            history = self._tail

            video = torch.randn(B, Cz, F_total, h, w, device=device, dtype=dtype)
            video[:, :, :n_hist] = history
            action = torch.randn(
                B, model.action_horizon, model.action_dim, device=device, dtype=dtype
            )
            new_latents, new_action = _denoise_chunk(
                model,
                sched,
                video,
                action,
                history,
                n_hist,
                F_total,
                seq_len,
                context,
                state,
                emb,
            )
            local_preds.append(new_latents)
            local_actions.append(new_action[:, -geo.num_action_per_block :])

            # Advance the episode timeline. Teacher forcing realizes the GT for
            # these positions (this window's latents 1+c*K .. 1+(c+1)*K, which
            # ARE global latents chunk_start .. chunk_start+K); autoregressive
            # rolling realizes the model's own predictions.
            if self.teacher_force:
                realized = win_lat[:, :, 1 + c * K : 1 + (c + 1) * K]
            else:
                realized = new_latents
            self._tail = torch.cat([self._tail[:, :, K:], realized], dim=2)
            self._tail_end += K
            self._chunks += 1

        pred_actions = torch.cat(local_actions, dim=1)
        all_latents = torch.cat([win_lat[:, :, :1]] + local_preds, dim=2)
        frames = model.vae.decode(all_latents)[:, :, 1:]  # drop anchor recon
        self._windows += 1
        self._pixel_frames += frames.shape[2]
        return pred_actions, frames
