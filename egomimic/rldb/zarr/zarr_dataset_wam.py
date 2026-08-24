"""
WAM dataset: loads camera keys as a short frame *clip* for the world model.

The base ``ZarrDataset`` (used by VLA/Pi training) reads a single frame per
camera key. WAM needs a contiguous clip ``[t, t+1, ..., t+horizon-1]`` (the Wan
VAE compresses 4x temporally, so ~5 frames -> 2 latent frames: a conditioning
frame + a future frame). This subclass **extends** the base — it only overrides
``__getitem__`` to decode a *windowed* camera read (a camera key carrying a
``horizon``) into a clip ``(T, C, H, W)``; proprio/action-chunk windowing,
transforms, bounds and normalization are all inherited unchanged. The base
loader is left untouched.

Wire it via ``S3WamEpisodeResolver`` (``_dataset_class = ZarrWamDataset``) and a
keymap whose camera key has a ``horizon`` (see ``Human.get_keymap(cam_horizon=...)``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import simplejpeg
import torch

from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.zarr.zarr_dataset_multi import (
    LocalEpisodeResolver,
    ModalEpisodeResolver,
    MultiDataset,
    S3EpisodeResolver,
    ZarrDataset,
    ZarrEpisode,
    get_fallback_idx,
    logger,
)
from egomimic.utils.pose_utils import _matrix_to_xyzwxyz

__all__ = [
    "ZarrWamDataset",
    "WamMultiDataset",
    "S3WamEpisodeResolver",
    "LocalWamEpisodeResolver",
    "ModalWamEpisodeResolver",
    "ZarrEpisode",
]


class WamMultiDataset(MultiDataset):
    """MultiDataset for the WAM pipeline. Identical to the base except it skips
    quantile bounds rejection (WAM action/state are raw frame-aligned ee-poses
    with no bound calibration) — a one-method override, base left untouched."""

    def _check_bounds(self, *args, **kwargs):
        return None


class ZarrWamDataset(ZarrDataset):
    """ZarrDataset that decodes windowed camera reads into a clip (T, C, H, W)."""

    _QUAT_ZERO_EPS = 1e-6
    # Decode-explosion guards. A corrupt zarr chunk / JPEG header can declare
    # an absurd decompressed size; the resulting allocation is a HOST-LEVEL
    # OOM-kill (no Python exception, rank dies silently — observed as the
    # deterministic wam22_dw48 crash at a fixed shuffle position). Bound what
    # we are willing to materialize and turn violations into ordinary
    # exceptions the resample machinery can absorb.
    _MAX_JPEG_BYTES = 64 * 1024 * 1024  # compressed frame cap (64 MB)
    # Per-side decoded-dimension cap. Real egoverse camera frames are <=1080p;
    # 2560 leaves headroom while bounding the worst-case 17-frame float64 clip
    # at ~2.5 GB (vs ~26 GB at an 8k cap) so a lying SOF header can't OOM the
    # worker pool even if every frame in a window is corrupt the same way.
    _MAX_JPEG_DIM = 2560
    _MIN_JPEG_DIM = 8
    _MAX_RAW_READ_BYTES = 1024 * 1024 * 1024  # one key's window read cap (1 GB)

    @classmethod
    def _decode_one(cls, buf) -> np.ndarray:
        # Header pre-check BEFORE the pixel allocation: a corrupt SOF segment
        # claiming e.g. 60000x60000 would otherwise malloc ~10 GB per frame
        # (x8 more after the float64 /255.0), OOM-killing the worker before
        # any except-clause can run. Raising here routes into the caller's
        # "JPEG decode failed" resample path instead.
        n_bytes = len(buf) if hasattr(buf, "__len__") else 0
        if n_bytes < 16 or n_bytes > cls._MAX_JPEG_BYTES:
            raise ValueError(f"implausible JPEG buffer size {n_bytes} bytes")
        h, w, _, _ = simplejpeg.decode_jpeg_header(buf)
        if not (
            cls._MIN_JPEG_DIM <= h <= cls._MAX_JPEG_DIM
            and cls._MIN_JPEG_DIM <= w <= cls._MAX_JPEG_DIM
        ):
            raise ValueError(f"implausible JPEG dims {h}x{w}")
        d = simplejpeg.decode_jpeg(buf, colorspace="RGB")
        return np.transpose(d, (2, 0, 1)) / 255.0  # (C, H, W) in [0,1]

    @classmethod
    def _oversized_read(cls, arr) -> str | None:
        """Return a reason string when a just-read window is implausibly large.

        Bounds the bytes we will carry forward into padding / float conversion
        / stacking (each of which COPIES, multiplying a corrupt decompression
        by up to 8x via the float64 conversions downstream). Object arrays of
        JPEG buffers are summed by buffer length.
        """
        if arr is None:
            return None
        total = 0
        if isinstance(arr, np.ndarray):
            if arr.dtype == object:
                try:
                    total = sum(len(b) for b in arr.ravel() if hasattr(b, "__len__"))
                except Exception:
                    return None
            else:
                total = int(arr.nbytes)
        elif hasattr(arr, "__len__") and isinstance(arr, (bytes, bytearray)):
            total = len(arr)
        if total > cls._MAX_RAW_READ_BYTES:
            return f"{total} bytes > cap {cls._MAX_RAW_READ_BYTES}"
        return None

    @classmethod
    def _validate_arrays(cls, data: dict) -> str | None:
        """Scan loaded arrays for NaN/Inf and zero-quaternion rows.

        Why: ``WamMultiDataset._check_bounds`` is a no-op for WAM, so the
        MultiDataset-level NaN/quantile check is skipped. A single bad row —
        NaN in a state chunk, or an all-zero quaternion in ``obs_head_pose`` /
        ``{left,right}_extrinsics_pose`` — crashes
        ``ActionChunkCoordinateFrameTransform`` because
        ``SE3.from_matrix(_xyzwxyz_to_matrix(...))`` on a degenerate quat
        yields a NaN matrix whose ``.inverse()`` propagates NaN into every
        subsequent chunk. In a DDP dataloader worker that failure can wedge
        the rank silently (worker stalls or spins in a retry loop), dropping
        GPU util to 0% on that rank while the other ranks time out on the
        next NCCL collective. Catching these here lets us log the defect and
        resample a fresh idx cleanly.

        Returns a one-line reason string on the first defect, or None if clean.
        """
        for k, v in data.items():
            if not isinstance(v, np.ndarray):
                continue
            if v.dtype.kind not in "fc":
                continue
            if not np.all(np.isfinite(v)):
                return f"NaN/Inf in key={k}"
            # xyzwxyz layout: last dim 7 means [x, y, z, qw, qx, qy, qz].
            # All last-dim-7 arrays in the WAM keymap are poses.
            if v.shape and v.shape[-1] == 7:
                quat = v[..., 3:7].reshape(-1, 4)
                norms = np.linalg.norm(quat, axis=-1)
                if np.any(norms < cls._QUAT_ZERO_EPS):
                    return f"Zero-quat in key={k} (min |q|={float(norms.min()):.2e})"
        return None

    def __getitem__(self, idx, _fallback_origin=None, _attempts=None):
        # This fork's base ZarrDataset opens the reader lazily and derives
        # total_frames / _image_keys / _json_keys in _init_from_metadata; the
        # base __getitem__ calls this first. Our windowed override must too, or
        # self.episode_reader is None and total_frames is 0.
        self._ensure_episode_reader()
        origin = _fallback_origin if _fallback_origin is not None else idx
        attempts = _attempts

        def _next(reason: str, key: str = "") -> int:
            nonlocal attempts
            next_idx, attempts = get_fallback_idx(
                idx=idx,
                candidates=range(self.total_frames),
                _attempts=attempts,
                max_attempts=self.total_frames,
                exhausted_error=(
                    f"Entire episode bad (no valid indices): "
                    f"ep={Path(self.episode_path).name}"
                ),
            )
            logger.warning(
                f"{reason} ep={Path(self.episode_path).name} frame={idx}"
                + (f" key={key}" if key else "")
                + f" | attempt {attempts}, trying random idx {next_idx}"
            )
            return next_idx

        while True:
            data = {}
            retry = False
            for k in self.key_map:
                zarr_key = self.key_map[k]["zarr_key"]
                key_type = self.key_map[k].get("key_type", None)
                horizon = self.key_map[k].get("horizon", None)

                if key_type == "annotation_keys":
                    data[k] = self._annotation_text_for_frame(idx)
                    continue

                # ``stride`` samples this key's horizon every Nth RAW frame, so
                # a 30 fps source is read as a 5 fps clip (stride 6). WAM-only,
                # and it lives HERE because this class implements its own
                # __getitem__ and so inherits nothing from ZarrDataset's read.
                #
                # The bug this fixes: the stride reached get_wam_keymap and the
                # episode tiler (anchors stepping (cam_horizon-1)*stride = 96)
                # but NOT the frame read, so clips stayed 17 CONSECUTIVE frames
                # (0.567 s) while windows sat 96 raw frames apart -- 16 output
                # frames advancing 1 raw frame each, then a 6x jump at the seam.
                # Exactly "30 fps content emitted as 5 fps inside the chunk".
                stride = int(self.key_map[k].get("stride") or 1)
                strided = None
                if horizon is not None:
                    if stride > 1:
                        # The reader takes {key: (start, end)} and UNPACKS the
                        # value as a 2-tuple, so it cannot express a step and an
                        # index array would raise. Reading the contiguous span
                        # and slicing would pull (horizon-1)*stride+1 frames to
                        # keep horizon -- 97 instead of 17 for a 5 fps clip,
                        # ~6x the bytes on 360x640 frames and a /dev/shm hazard
                        # at 10 workers. One single-frame read per wanted index
                        # keeps only what we use.
                        stop = min(idx + (horizon - 1) * stride + 1, self.total_frames)
                        strided = list(range(idx, stop, stride))
                        read_interval = None
                    else:
                        end_idx = self._chunk_end_idx(idx, horizon, key_type)
                        read_interval = (idx, end_idx)
                else:
                    read_interval = (idx, None)
                # Guarded zarr read: a corrupt chunk can raise deep inside
                # numcodecs (or decompress into something enormous). Turn both
                # into a logged resample instead of a dead dataloader worker.
                try:
                    if strided is not None:
                        parts = [
                            self.episode_reader.read({zarr_key: (i, i + 1)})[zarr_key]
                            for i in strided
                        ]
                        raw_data = {zarr_key: np.concatenate(parts, axis=0)}
                    else:
                        raw_data = self.episode_reader.read({zarr_key: read_interval})
                except Exception as e:
                    idx = _next(f"Zarr read failed ({type(e).__name__}: {e})", key=k)
                    retry = True
                    break
                over = self._oversized_read(raw_data.get(zarr_key))
                if over is not None:
                    idx = _next(f"Oversized zarr read ({over})", key=k)
                    retry = True
                    break
                self._pad_sequences(raw_data, horizon)
                data[k] = raw_data[zarr_key]

                if zarr_key in self._image_keys:
                    try:
                        if horizon is not None:
                            # windowed camera read -> clip (T, C, H, W)
                            data[k] = np.stack([self._decode_one(b) for b in data[k]])
                        else:
                            data[k] = self._decode_one(data[k])  # (C, H, W)
                    except Exception:
                        idx = _next("JPEG decode failed", key=k)
                        retry = True
                        break
                elif zarr_key in self._json_keys:
                    if isinstance(data[k], np.ndarray):
                        data[k] = [self._decode_json_entry(v) for v in data[k]]
                    else:
                        data[k] = self._decode_json_entry(data[k])
            if retry:
                continue

            # --- Per-episode camera calibration into the batch (BEFORE
            # transforms) so downstream steps that operate in camera frame —
            # e.g. ``Eva.get_wam_transform_list`` — can pull per-episode
            # ``{left,right}_extrinsics_pose`` (via the transform's
            # ``extra_batch_key`` setdefault fallback) instead of hardcoding
            # class-level constants.
            extr = self.episode_reader.extrinsics
            if isinstance(extr, dict):
                for arm_key, se3 in extr.items():
                    arr = np.asarray(se3, dtype=np.float32)
                    if arr.shape == (4, 4):
                        xyzq = _matrix_to_xyzwxyz(arr[None, :])[0].astype(np.float32)
                        data[f"{arm_key}_extrinsics_pose"] = xyzq

            # Pre-transform validation: catch NaN/Inf and zero-quats before
            # they blow up the SE3 coord-frame transform (see _validate_arrays).
            bad = self._validate_arrays(data)
            if bad is not None:
                idx = _next(bad)
                continue

            if self.transform:
                try:
                    for transform in self.transform or []:
                        data = transform.transform(data)
                except Exception as e:
                    idx = _next(f"Transform failed ({type(e).__name__}: {e})")
                    continue

            # Post-transform validation: transforms (interpolation, SE3
            # inverse, unwrap) can introduce NaN on near-degenerate inputs
            # that pass the pre-check.
            bad = self._validate_arrays(data)
            if bad is not None:
                idx = _next(f"post-transform: {bad}")
                continue

            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    data[k] = torch.from_numpy(v).to(torch.float32)

            data["embodiment"] = get_embodiment_id(self.embodiment)
            ep_name = Path(self.episode_path).name
            data["episode_hash"] = (
                ep_name[:-5] if ep_name.endswith(".zarr") else ep_name
            )

            # Per-episode intrinsics (3x4 K) from zarr metadata so viz /
            # projection can use per-episode calibration; NaN sentinel when
            # the episode doesn't persist a K (viz falls back to constants).
            K = self.episode_reader.intrinsics
            if isinstance(K, dict):
                K = next(
                    (v for k, v in K.items() if "front" in str(k).lower()),
                    next(iter(K.values()), None) if K else None,
                )
            if K is not None:
                K = np.asarray(K, dtype=np.float32)
                if K.shape == (3, 3):
                    K = np.concatenate([K, np.zeros((3, 1), dtype=np.float32)], axis=1)
                if K.shape != (3, 4):
                    K = np.full((3, 4), np.nan, dtype=np.float32)
            else:
                K = np.full((3, 4), np.nan, dtype=np.float32)
            data["intrinsics"] = torch.from_numpy(np.ascontiguousarray(K))
            _ = origin
            return data


class S3WamEpisodeResolver(S3EpisodeResolver):
    """S3 resolver that builds ZarrWamDataset leaves (clip-loading camera keys)."""

    _dataset_class = ZarrWamDataset


class LocalWamEpisodeResolver(LocalEpisodeResolver):
    _dataset_class = ZarrWamDataset


class ModalWamEpisodeResolver(ModalEpisodeResolver):
    """Modal-volume resolver (SQL filter + /mnt/zarr-data) building ZarrWamDataset
    clip leaves. This is the WAM data path for the Modal training pipeline."""

    _dataset_class = ZarrWamDataset
