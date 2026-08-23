"""Full-episode validation-split planning for WAM.

The WAM evaluator writes one ``predicted_video_*.mp4`` + ``validation_video_*.mp4``
pair per validation EPISODE, each spanning the whole episode. That needs the val
dataloader to walk every episode end to end, which is what this module sets up.

Why not simply ask the dataset for one episode-length window? Because a mecka
dishwashing episode is 1945-2245 frames at 30 fps and ``ZarrWamDataset`` decodes
JPEG frames to float ``(3, 360, 640)`` — a single full-episode sample is ~12 GB
in a dataloader worker. So the episode is TILED into ``cam_horizon``-length
windows with stride ``cam_horizon - 1`` (window j's frame 0 is window j-1's last
frame, so the windows cover the episode exactly once with no gaps) and
``egomimic.eval.wam_rollout.EpisodeRoller`` carries the teacher-forcing context
across the window seams. Memory stays at one 17-frame clip; the rollout is
continuous over the whole episode.

Two properties of the tiling matter:

  * **stride == cam_horizon - 1.** Window j covers raw frames
    ``[j*S, j*S + cam_horizon)`` with ``S = cam_horizon - 1``, so its VAE latents
    ``1 .. F_w-1`` are exactly the next ``F_w - 1`` latents of the episode
    timeline and its latent 0 re-encodes the previous window's last frame. Any
    other stride would make the global latent grid drift and the roller's
    ``_tail_end == chunk_start`` assertion fire.
  * **no tail padding.** ``ZarrDataset`` pads windows that overrun the episode by
    repeating the final frame (a frozen clip with zero-velocity actions), so the
    last partial window is dropped rather than fed to the model. Coverage is
    therefore ``(M-1)*S + cam_horizon`` frames, i.e. within ``cam_horizon`` of
    the true episode length.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from egomimic.eval.wam_rollout import (
    DEFAULT_SOURCE_FPS,
    WAM_VIDEO_FPS,
    pixels_to_latents,
    video_frame_stride,
)


@dataclass
class EpisodePlan:
    """The window walk for one validation episode."""

    episode: str
    total_frames: int
    cam_horizon: int
    stride: int
    source_fps: float = DEFAULT_SOURCE_FPS
    # Stride the data pipeline already applied (6 => the clip is 5 fps).
    data_frame_stride: int = 1
    anchors: list[int] = field(default_factory=list)

    @property
    def num_windows(self) -> int:
        return len(self.anchors)

    @property
    def frame_stride(self) -> int:
        """Video-assembly subsample stride for real-time playback at 5 fps."""
        return video_frame_stride(self.source_fps, self.data_frame_stride)

    @property
    def covered_frames(self) -> int:
        if not self.anchors:
            return 0
        return self.anchors[-1] + self.cam_horizon

    @property
    def latents_per_window(self) -> int:
        return pixels_to_latents(self.cam_horizon)

    @property
    def pred_pixel_frames(self) -> int:
        """Frames the rollout PREDICTS (native ``source_fps`` granularity):
        4 pixel frames per predicted latent, ``(F_w - 1)`` predicted latents per
        window (the anchor is not predicted)."""
        return self.num_windows * (self.latents_per_window - 1) * 4

    @property
    def video_frames(self) -> int:
        """Frames actually WRITTEN to the mp4, after real-time subsampling."""
        s = self.frame_stride
        return (self.pred_pixel_frames + s - 1) // s

    @property
    def duration_s(self) -> float:
        """Playback length of the mp4."""
        return self.video_frames / WAM_VIDEO_FPS

    @property
    def predicted_span_s(self) -> float:
        """REAL elapsed time of the span the rollout covered."""
        return self.pred_pixel_frames / self.source_fps

    @property
    def episode_duration_s(self) -> float:
        """REAL elapsed time of the whole episode."""
        return self.total_frames / self.source_fps

    def describe(self) -> str:
        return (
            f"{self.episode}: {self.total_frames} frames @ {self.source_fps:g} fps "
            f"({self.episode_duration_s:.1f}s real) -> {self.num_windows} windows "
            f"x {self.cam_horizon} (stride {self.stride}), covering "
            f"{self.covered_frames}/{self.total_frames} frames "
            f"({100.0 * self.covered_frames / max(1, self.total_frames):.1f}%); "
            f"predicts {self.pred_pixel_frames} frames @ {self.source_fps:g} fps, "
            f"subsampled x{self.frame_stride} -> video {self.video_frames} frames "
            f"@ {WAM_VIDEO_FPS} fps = {self.duration_s:.1f}s "
            f"(predicted span {self.predicted_span_s:.1f}s real)"
        )

    def assert_realtime(self) -> None:
        """Fail loudly if the mp4 would not play back in real time.

        This is the check that catches "5 fps by slowing down instead of
        subsampling": if every predicted frame were written into a 5 fps
        container, ``duration_s`` would be ``source_fps / WAM_VIDEO_FPS`` times
        ``predicted_span_s`` (6x for 30 fps mecka data) and this assertion fires.
        Tolerance is one output frame, since the subsample keeps
        ``ceil(n / stride)`` frames.
        """
        tol = 1.0 / WAM_VIDEO_FPS
        drift = abs(self.duration_s - self.predicted_span_s)
        assert drift <= tol, (
            f"{self.episode}: video would be {self.duration_s:.2f}s for a "
            f"{self.predicted_span_s:.2f}s predicted span "
            f"({self.duration_s / max(1e-9, self.predicted_span_s):.2f}x real time; "
            f"drift {drift:.2f}s > {tol:.2f}s). The mp4 must be a real-time "
            f"subsample of the native {self.source_fps:g} fps prediction, not a "
            f"slowed-down copy of it."
        )


def _camera_horizon(dataset) -> int:
    """The camera-clip horizon this per-episode dataset reads."""
    for spec in (dataset.key_map or {}).values():
        if spec.get("key_type") == "camera_keys" and spec.get("horizon"):
            return int(spec["horizon"])
    raise ValueError(
        "no camera key with a horizon in the dataset key_map — WAM needs a "
        "clip-valued camera key (see Mecka.get_wam_keymap)."
    )


def _data_frame_stride(dataset) -> int:
    """Stride the DATA pipeline applies to the camera clip (1 if none).

    ``Mecka.get_wam_keymap(frame_stride=6)`` stamps ``stride`` onto every
    horizoned key, so the clip is a 5 fps resample of a 30 fps source. The
    window tiling must then advance in RAW frames per SAMPLED frame, and the
    playback subsample must not decimate a second time.
    """
    for spec in (dataset.key_map or {}).values():
        if spec.get("key_type") == "camera_keys" and spec.get("horizon"):
            return max(1, int(spec.get("stride") or 1))
    return 1


def _source_fps(dataset, log=None) -> float:
    """Frame rate of the episode's raw camera stream, from its zarr metadata.

    Mecka episodes persist ``fps`` in the zarr attributes (30 for the dishwashing
    set). Falling back to a constant would silently reintroduce a wrong-speed
    video on any dataset recorded at another rate, so we read it and only fall
    back with a warning.
    """
    meta = getattr(dataset, "metadata", None) or {}
    for key in ("fps", "frame_rate", "framerate", "hz"):
        val = meta.get(key)
        if val:
            try:
                fps = float(val)
            except (TypeError, ValueError):
                continue
            if fps > 0:
                return fps
    if log is not None:
        log.warning(
            f"[wam_episode] no fps in zarr metadata for "
            f"{getattr(dataset, 'episode_path', '?')}; assuming "
            f"{DEFAULT_SOURCE_FPS:g} fps. If that is wrong the mp4 playback "
            "speed will be wrong."
        )
    return DEFAULT_SOURCE_FPS


def plan_full_episode_walk(
    mds,
    num_episodes: int,
    max_windows_per_episode: int | None = None,
    log=None,
) -> list[EpisodePlan]:
    """Rewrite ``mds`` in place so its samples tile whole episodes, in order.

    Keeps the first ``num_episodes`` episodes (sorted, for determinism) and
    replaces the flat sample index with the tiled window anchors of each, so
    iterating the val dataloader sequentially walks episode 0 start-to-finish,
    then episode 1, and so on. Returns one :class:`EpisodePlan` per kept
    episode.

    ``max_windows_per_episode`` truncates each episode's walk — for smoke tests
    only; leave it ``None`` for the real full-episode pass.
    """
    all_names = sorted(mds.datasets.keys())
    if len(all_names) < num_episodes and log is not None:
        log.warning(
            f"[wam_episode] valid split has {len(all_names)} episodes < "
            f"requested {num_episodes}; using all of them."
        )
    kept = all_names[:num_episodes]
    if not kept:
        raise ValueError("valid split is empty — nothing to evaluate.")

    mds.datasets = {name: mds.datasets[name] for name in kept}
    mds.index_map = []
    mds._global_indices_by_dataset = {name: [] for name in kept}

    plans: list[EpisodePlan] = []
    for name in kept:
        ds = mds.datasets[name]
        # total_frames is populated lazily from the zarr metadata / SQL row.
        if not getattr(ds, "total_frames", 0):
            ds._ensure_episode_reader()
        total = int(ds.total_frames)
        cam_h = _camera_horizon(ds)
        # With a data stride the clip is a resample, so one window SPANS
        # (cam_h - 1) * dstride + 1 raw frames while still yielding cam_h
        # samples. Anchors must therefore advance in RAW frames by
        # (cam_h - 1) * dstride to keep window j's frame 0 on window j-1's last
        # SAMPLED frame — stepping by cam_h - 1 would overlap windows ~6x and
        # re-emit the same content over and over.
        dstride = _data_frame_stride(ds)
        stride = (cam_h - 1) * dstride
        if stride < 1:
            raise ValueError(f"cam_horizon must be > 1 (got {cam_h})")
        # Only whole windows: a window that overruns the episode would be
        # tail-padded with copies of the final frame.
        last_anchor = total - ((cam_h - 1) * dstride + 1)
        if last_anchor < 0:
            if log is not None:
                log.warning(
                    f"[wam_episode] episode {name} has {total} frames < "
                    f"cam_horizon {cam_h}; skipping."
                )
            continue
        anchors = list(range(0, last_anchor + 1, stride))
        if max_windows_per_episode is not None:
            anchors = anchors[: int(max_windows_per_episode)]
        plan = EpisodePlan(
            episode=name,
            total_frames=total,
            cam_horizon=cam_h,
            stride=stride,
            data_frame_stride=dstride,
            source_fps=_source_fps(ds, log=log),
            anchors=anchors,
        )
        # Catch a non-real-time video BEFORE spending inference time on it.
        plan.assert_realtime()
        plans.append(plan)
        for anchor in anchors:
            global_idx = len(mds.index_map)
            mds.index_map.append((name, anchor))
            mds._global_indices_by_dataset[name].append(global_idx)

    if log is not None:
        for plan in plans:
            log.info(f"[wam_episode] {plan.describe()}")
        log.info(
            f"[wam_episode] {len(plans)} episodes / {len(mds.index_map)} val "
            "windows total (set trainer.limit_val_batches to this at batch_size=1)"
        )
    return plans


def total_windows(plans: list[EpisodePlan]) -> int:
    return sum(p.num_windows for p in plans)
