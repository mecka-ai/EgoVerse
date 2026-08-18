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

from egomimic.eval.wam_rollout import WAM_VIDEO_FPS, pixels_to_latents


@dataclass
class EpisodePlan:
    """The window walk for one validation episode."""

    episode: str
    total_frames: int
    cam_horizon: int
    stride: int
    anchors: list[int] = field(default_factory=list)

    @property
    def num_windows(self) -> int:
        return len(self.anchors)

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
        """Frames in this episode's mp4s: 4 pixel frames per predicted latent,
        ``(F_w - 1)`` predicted latents per window (the anchor is not predicted)."""
        return self.num_windows * (self.latents_per_window - 1) * 4

    @property
    def duration_s(self) -> float:
        return self.pred_pixel_frames / WAM_VIDEO_FPS

    def describe(self) -> str:
        return (
            f"{self.episode}: {self.total_frames} frames -> {self.num_windows} "
            f"windows x {self.cam_horizon} (stride {self.stride}), "
            f"covering {self.covered_frames}/{self.total_frames} frames "
            f"({100.0 * self.covered_frames / max(1, self.total_frames):.1f}%); "
            f"video {self.pred_pixel_frames} frames @ {WAM_VIDEO_FPS} fps = "
            f"{self.duration_s:.1f}s"
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
        stride = cam_h - 1
        if stride < 1:
            raise ValueError(f"cam_horizon must be > 1 (got {cam_h})")
        # Only whole windows: a window that overruns the episode would be
        # tail-padded with copies of the final frame.
        last_anchor = total - cam_h
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
            anchors=anchors,
        )
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
