"""Convert XDOF/ABC-130k MCAP episodes to Zarr format.

ABC-130k (https://huggingface.co/datasets/XDOF/ABC-130k) ships each episode as
an ``episode.mcap`` (protobuf-encoded robot/video streams) plus an optional
``annotation.mcap`` (timestamped subtask labels). The platform is a bimanual
pair of 6-DoF YAM arms with parallel-jaw grippers and three cameras
(top + two wrists), so episodes map cleanly onto the ``eva_bimanual``
embodiment already understood by the rest of the pipeline.

Mirrors the ``main(args)`` interface of ``eva_to_zarr.py`` so a batch driver can
swap converters. Unlike the EVA HDF5 path, ABC stores full 4x4 end-effector
pose matrices (native robot frame, no table->eef transform) and H.264 video,
so this converter emits the final per-arm zarr keys directly and decodes video
with PyAV.

Two input sources are supported (``--source``):
    local : read episode.mcap[/annotation.mcap] from a directory on disk.
    hf    : stream the MCAP bytes straight from the Hugging Face Hub and convert
            on the fly, without staging the raw ``.mcap`` on disk. Each stream
            is read sequentially exactly once, so download and decode overlap.

MCAP topics consumed (each 1:1 per frame):
    /instruction                         -> task_description (single message)
    /top-camera                          -> images.front_1      (H.264)
    /left-wrist-camera                   -> images.left_wrist   (H.264)
    /right-wrist-camera                  -> images.right_wrist  (H.264)
    /{side}-arm-state   (RobotState)     -> {side}.obs_ee_pose, {side}.obs_joints
    /{side}-arm-action  (RobotState)     -> {side}.cmd_ee_pose, {side}.cmd_joints
    /{side}-ee-state    (GripperState)   -> {side}.obs_gripper
    /{side}-ee-action   (GripperState)   -> {side}.cmd_gripper

Requires: mcap, mcap-protobuf-support, av (PyAV). The ``hf`` source additionally
requires huggingface_hub and requests.
"""

import argparse
import contextlib
import io
import logging
import os
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from egomimic.rldb.zarr.zarr_writer import ZarrWriter
from egomimic.utils.pose_utils import xyzw_to_wxyz

logger = logging.getLogger(__name__)


def str2bool(v) -> bool:
    """Parse a CLI boolean. Defined locally so this converter depends only on
    light deps (numpy/scipy/zarr/mcap/av) and not the torch-heavy egomimicUtils.
    """
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "y", "t")


# Per-arm topic groups. "front" camera is shared (top-down) and always included.
_ARM_TOPICS = {
    "left": {
        "arm_state": "/left-arm-state",
        "arm_action": "/left-arm-action",
        "ee_state": "/left-ee-state",
        "ee_action": "/left-ee-action",
        "wrist_cam": "/left-wrist-camera",
        "wrist_zarr": "images.left_wrist",
    },
    "right": {
        "arm_state": "/right-arm-state",
        "arm_action": "/right-arm-action",
        "ee_state": "/right-ee-state",
        "ee_action": "/right-ee-action",
        "wrist_cam": "/right-wrist-camera",
        "wrist_zarr": "images.right_wrist",
    },
}
_TOP_CAMERA_TOPIC = "/top-camera"
_INSTRUCTION_TOPIC = "/instruction"


def _arm_to_embodiment(arm: str) -> str:
    """Map arm string to embodiment identifier (parity with eva_to_zarr)."""
    return {
        "left": "eva_left_arm",
        "right": "eva_right_arm",
        "both": "eva_bimanual",
    }.get(arm, "eva_bimanual")


def _ts_ns(ts) -> int:
    """Protobuf Timestamp (seconds, nanos) -> int nanoseconds."""
    return int(ts.seconds) * 1_000_000_000 + int(ts.nanos)


def _pose16_to_xyz_quat(pose_flat) -> np.ndarray:
    """Row-major flattened 4x4 homogeneous transform -> [x, y, z, qw, qx, qy, qz].

    ABC poses are native robot-frame end-effector transforms; orientation is
    converted to a wxyz quaternion to match the EVA zarr convention.
    """
    M = np.asarray(pose_flat, dtype=np.float64).reshape(4, 4)
    translation = M[:3, 3]
    quat_xyzw = R.from_matrix(M[:3, :3]).as_quat()
    quat_wxyz = xyzw_to_wxyz(quat_xyzw)
    return np.concatenate([translation, quat_wxyz], axis=-1)


# JPEG quality matches ZarrWriter.JPEG_QUALITY so pre-encoded bytes are identical
# to what ZarrWriter would produce from raw frames.
_JPEG_QUALITY = 85


def _decode_h264_jpeg(packets: list[bytes]) -> tuple[np.ndarray, list[int]]:
    """Decode H.264 access units, JPEG-encoding each frame as it is produced.

    Returns ``(encoded, [H, W, 3])`` where ``encoded`` is an object array of JPEG
    byte strings, shape ``(T,)``. Decoding is incremental: each RGB frame is
    encoded and released immediately, so peak memory is the compressed JPEG set
    (~tens of KB/frame) rather than the full raw-RGB tensor (~MB/frame).
    """
    import av  # imported lazily so the module loads without PyAV present
    import simplejpeg

    codec = av.CodecContext.create("h264", "r")
    encoded: list[bytes] = []
    shape: list[int] | None = None

    def _emit(frame) -> None:
        nonlocal shape
        img = np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
        if shape is None:
            shape = list(img.shape)  # [H, W, 3]
        encoded.append(
            simplejpeg.encode_jpeg(img, quality=_JPEG_QUALITY, colorspace="RGB")
        )

    for data in packets:
        for packet in codec.parse(data):
            for frame in codec.decode(packet):
                _emit(frame)
    for frame in codec.decode(None):  # flush the reorder buffer
        _emit(frame)

    arr = np.empty((len(encoded),), dtype=object)
    for i, b in enumerate(encoded):
        arr[i] = b
    return arr, (shape if shape is not None else [0, 0, 3])


class _HTTPStream(io.RawIOBase):
    """Read-only, non-seekable binary file-like over a streaming HTTP response.

    Lets the MCAP ``NonSeekingReader`` consume an episode directly from the
    network so conversion overlaps the download.
    """

    def __init__(self, byte_iter):
        self._it = byte_iter
        self._buf = b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def readinto(self, b) -> int:
        while not self._buf:
            try:
                self._buf = next(self._it)
            except StopIteration:
                return 0
        n = min(len(b), len(self._buf))
        b[:n] = self._buf[:n]
        self._buf = self._buf[n:]
        return n


class ABCMcapExtractor:
    """Reads one ABC-130k episode and returns aligned feature arrays."""

    @classmethod
    def process_episode(cls, episode_dir: Path, arm: str) -> dict:
        """Extract features from a local episode directory.

        See :meth:`process_streams` for the returned structure.
        """
        episode_dir = Path(episode_dir)
        episode_path = episode_dir / "episode.mcap"
        if not episode_path.exists():
            raise FileNotFoundError(f"No episode.mcap in {episode_dir}")
        ann_path = episode_dir / "annotation.mcap"

        with open(episode_path, "rb") as ep_stream:
            if ann_path.exists():
                with open(ann_path, "rb") as ann_stream:
                    return cls.process_streams(ep_stream, ann_stream, arm)
            return cls.process_streams(ep_stream, None, arm)

    @classmethod
    def process_streams(cls, episode_stream, annotation_stream, arm: str) -> dict:
        """Extract numeric + image features from open MCAP byte streams.

        ``episode_stream`` / ``annotation_stream`` are any readable binary
        file-like objects (local files or HTTP streams); ``annotation_stream``
        may be ``None``. Each stream is read sequentially exactly once, so an
        episode can be converted straight from a download without staging the
        raw ``.mcap`` on disk.

        Returns a dict containing:
            task_description : str
            annotations      : list[(text, start_idx, end_idx)]
            numeric <key>    : np.ndarray, shape (T, *)
            image  <key>     : (encoded_jpeg_obj_array (T,), [H, W, 3]) tuple
        All frame-indexed arrays/images are truncated to a common length T.
        """
        from mcap.reader import make_reader
        from mcap_protobuf.decoder import DecoderFactory

        sides = ["left", "right"] if arm == "both" else [arm]
        cam_topics = [_TOP_CAMERA_TOPIC] + [_ARM_TOPICS[s]["wrist_cam"] for s in sides]
        numeric_topics = []
        for s in sides:
            t = _ARM_TOPICS[s]
            numeric_topics += [
                t["arm_state"],
                t["arm_action"],
                t["ee_state"],
                t["ee_action"],
            ]
        wanted = set(cam_topics + numeric_topics + [_INSTRUCTION_TOPIC])

        # Accumulators, collected in message order (streams are 1:1 per frame).
        cam_packets: dict[str, list[bytes]] = {t: [] for t in cam_topics}
        raw: dict[str, list] = {t: [] for t in numeric_topics}
        clock_ts: list[int] = []  # canonical frame clock = first arm-state topic
        clock_topic = _ARM_TOPICS[sides[0]]["arm_state"]
        task_description = ""

        reader = make_reader(episode_stream, decoder_factories=[DecoderFactory()])
        for _schema, channel, _msg, dec in reader.iter_decoded_messages(
            topics=list(wanted)
        ):
            topic = channel.topic
            if topic == _INSTRUCTION_TOPIC:
                task_description = str(dec.data)
            elif topic in cam_packets:
                cam_packets[topic].append(bytes(dec.data))
            elif topic in raw:
                raw[topic].append(dec)
                if topic == clock_topic:
                    clock_ts.append(_ts_ns(dec.timestamp))

        # Some tasks record joints only (empty RobotState.pose) -> no EE pose,
        # which is incompatible with the eva_bimanual cartesian format. Detect it
        # here (before the expensive video decode) and fail with a clear reason.
        for s in sides:
            st = raw[_ARM_TOPICS[s]["arm_state"]]
            if not st or len(list(st[0].pose)) != 16:
                raise ValueError(
                    f"{s}-arm has no 4x4 EE pose (joints-only episode); "
                    "incompatible with eva_bimanual cartesian format"
                )

        # Build per-arm numeric arrays.
        feats: dict[str, np.ndarray] = {}
        for s in sides:
            t = _ARM_TOPICS[s]
            arm_state = raw[t["arm_state"]]
            arm_action = raw[t["arm_action"]]
            ee_state = raw[t["ee_state"]]
            ee_action = raw[t["ee_action"]]

            feats[f"{s}.obs_ee_pose"] = np.stack(
                [_pose16_to_xyz_quat(m.pose) for m in arm_state]
            )
            feats[f"{s}.cmd_ee_pose"] = np.stack(
                [_pose16_to_xyz_quat(m.pose) for m in arm_action]
            )
            feats[f"{s}.obs_joints"] = np.asarray(
                [list(m.position) for m in arm_state], dtype=np.float32
            )
            feats[f"{s}.cmd_joints"] = np.asarray(
                [list(m.position) for m in arm_action], dtype=np.float32
            )
            feats[f"{s}.obs_gripper"] = np.asarray(
                [list(m.position) for m in ee_state], dtype=np.float32
            ).reshape(-1, 1)
            feats[f"{s}.cmd_gripper"] = np.asarray(
                [list(m.position) for m in ee_action], dtype=np.float32
            ).reshape(-1, 1)

        # Decode video to per-frame JPEG bytes. Packets for each camera are freed
        # right after decode so peak memory stays at the compressed set, not the
        # raw-RGB tensor. Image values are (encoded_obj_array, [H, W, 3]) tuples.
        images: dict[str, tuple[np.ndarray, list[int]]] = {}
        images["images.front_1"] = _decode_h264_jpeg(cam_packets.pop(_TOP_CAMERA_TOPIC))
        for s in sides:
            t = _ARM_TOPICS[s]
            images[t["wrist_zarr"]] = _decode_h264_jpeg(cam_packets.pop(t["wrist_cam"]))

        # Align everything to the common frame count (video can lag state by ~1).
        lengths = [len(v) for v in feats.values()]
        lengths += [len(arr) for arr, _ in images.values()]
        total = min(lengths) if lengths else 0
        if total <= 0:
            raise ValueError("Empty episode after decode")
        feats = {k: v[:total] for k, v in feats.items()}
        for k, (arr, shp) in images.items():
            feats[k] = (arr[:total], shp)
        clock_arr = np.asarray(clock_ts[:total])

        annotations = (
            cls._annotations_from_stream(annotation_stream, clock_arr, total)
            if annotation_stream is not None
            else []
        )

        feats["task_description"] = task_description
        feats["annotations"] = annotations
        return feats

    @staticmethod
    def _annotations_from_stream(stream, clock_ts: np.ndarray, total: int) -> list:
        """Map annotation.mcap subtask timestamps to (text, start_idx, end_idx)."""
        if stream is None or len(clock_ts) == 0:
            return []

        from mcap.reader import make_reader
        from mcap_protobuf.decoder import DecoderFactory

        marks: list[tuple[int, str]] = []
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _schema, _channel, _msg, dec in reader.iter_decoded_messages():
            idx = int(np.argmin(np.abs(clock_ts - _ts_ns(dec.timestamp))))
            marks.append((idx, str(dec.data)))

        marks.sort(key=lambda m: m[0])
        segments = []
        for i, (start, text) in enumerate(marks):
            end = marks[i + 1][0] if i + 1 < len(marks) else total
            if end > start:
                segments.append((text, start, end))
        return segments


def _write_zarr(
    feats: dict,
    output_dir: Path,
    dataset_name: str,
    arm: str,
    fps: int,
    task_name: str,
    save_mp4: bool,
    chunk_timesteps: int,
) -> tuple[Path, Path | None]:
    """Write an extracted feature dict to a ``<dataset_name>.zarr`` store."""
    task_description = feats.pop("task_description", "")
    annotations = feats.pop("annotations", [])

    # Image values are (encoded_jpeg_obj_array, [H, W, 3]) tuples -> pre-encoded path.
    pre_encoded_image_data = {k: v for k, v in feats.items() if k.startswith("images.")}
    numeric_data = {k: v for k, v in feats.items() if not k.startswith("images.")}

    zarr_path = ZarrWriter.create_and_write(
        episode_path=output_dir / f"{dataset_name}.zarr",
        numeric_data=numeric_data or None,
        pre_encoded_image_data=pre_encoded_image_data or None,
        embodiment=_arm_to_embodiment(arm),
        fps=fps,
        task_name=task_name or task_description,
        task_description=task_description,
        annotations=annotations,
        chunk_timesteps=chunk_timesteps,
    )
    logger.info("Wrote zarr episode: %s", zarr_path)

    mp4_path = None
    front = pre_encoded_image_data.get("images.front_1")
    if save_mp4 and front is not None and len(front[0]):
        import simplejpeg

        # Lazy import: pulls cv2, only needed for the optional preview.
        from egomimic.utils.video_utils import save_preview_mp4

        enc, _shape = front
        # Decode only the front camera, only when a preview is explicitly requested.
        frames = np.stack(
            [simplejpeg.decode_jpeg(bytes(b), colorspace="RGB") for b in enc]
        )
        mp4_path = output_dir / f"{dataset_name}.mp4"
        try:
            # save_preview_mp4 expects (T, C, H, W); frames are (T, H, W, C).
            save_preview_mp4(
                frames.transpose(0, 3, 1, 2), mp4_path, fps, half_res=False
            )
            logger.info("Saved preview MP4: %s", mp4_path)
        except Exception:
            logger.warning(
                "Failed to save preview MP4 at %s:\n%s",
                mp4_path,
                traceback.format_exc(),
            )
    return zarr_path, mp4_path


def convert_episode(
    episode_dir: Path,
    output_dir: Path,
    dataset_name: str,
    arm: str,
    fps: int,
    task_name: str = "",
    save_mp4: bool = False,
    chunk_timesteps: int = 100,
) -> tuple[Path, Path | None]:
    """Convert one local ABC-130k episode directory to a ``<dataset_name>.zarr``."""
    feats = ABCMcapExtractor.process_episode(episode_dir, arm)
    return _write_zarr(
        feats, output_dir, dataset_name, arm, fps, task_name, save_mp4, chunk_timesteps
    )


@contextlib.contextmanager
def _open_hf_episode_streams(repo_id: str, hf_episode_dir: str, token: str | None):
    """Yield (episode_stream, annotation_stream_or_None) for an episode on the Hub.

    ``episode.mcap`` is streamed lazily; the tiny ``annotation.mcap`` (if present)
    is buffered in memory. Nothing is written to disk.
    """
    import requests
    from huggingface_hub import hf_hub_url

    hf_episode_dir = hf_episode_dir.strip("/")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    ep_url = hf_hub_url(
        repo_id=repo_id, filename=f"{hf_episode_dir}/episode.mcap", repo_type="dataset"
    )
    ann_url = hf_hub_url(
        repo_id=repo_id,
        filename=f"{hf_episode_dir}/annotation.mcap",
        repo_type="dataset",
    )

    ep_resp = requests.get(ep_url, headers=headers, stream=True, timeout=120)
    ep_resp.raise_for_status()
    try:
        ann_resp = requests.get(ann_url, headers=headers, timeout=120)
        ann_stream = (
            io.BytesIO(ann_resp.content) if ann_resp.status_code == 200 else None
        )
        ep_stream = _HTTPStream(ep_resp.iter_content(chunk_size=1 << 20))
        yield ep_stream, ann_stream
    finally:
        ep_resp.close()


def convert_hf_episode(
    repo_id: str,
    hf_episode_dir: str,
    output_dir: Path,
    dataset_name: str,
    arm: str,
    fps: int,
    token: str | None = None,
    task_name: str = "",
    save_mp4: bool = False,
    chunk_timesteps: int = 100,
) -> tuple[Path, Path | None]:
    """Stream one ABC-130k episode from the Hub and convert it without staging."""
    with _open_hf_episode_streams(repo_id, hf_episode_dir, token) as (ep, ann):
        feats = ABCMcapExtractor.process_streams(ep, ann, arm)
    return _write_zarr(
        feats, output_dir, dataset_name, arm, fps, task_name, save_mp4, chunk_timesteps
    )


def _dataset_name_from(path_or_dir: str) -> str:
    """Episode UUID from a directory name like ``episode_<uuid>``."""
    name = Path(path_or_dir).name
    return name.replace("episode_", "") or name


def main(args) -> tuple[Path | None, Path | None]:
    """Convert one ABC-130k episode to Zarr (parity with eva_to_zarr)."""
    try:
        if args.source == "hf":
            return convert_hf_episode(
                repo_id=args.repo_id,
                hf_episode_dir=str(args.raw_path),
                output_dir=Path(args.output_dir),
                dataset_name=_dataset_name_from(str(args.raw_path)),
                arm=args.arm,
                fps=args.fps,
                token=args.hf_token or os.environ.get("HF_TOKEN"),
                task_name=args.task_name,
                save_mp4=args.save_mp4,
                chunk_timesteps=args.chunk_timesteps,
            )
        return convert_episode(
            episode_dir=Path(args.raw_path),
            output_dir=Path(args.output_dir),
            dataset_name=_dataset_name_from(str(args.raw_path)),
            arm=args.arm,
            fps=args.fps,
            task_name=args.task_name,
            save_mp4=args.save_mp4,
            chunk_timesteps=args.chunk_timesteps,
        )
    except Exception:
        logger.error("Error converting %s:\n%s", args.raw_path, traceback.format_exc())
        return None, None


def argument_parse():
    parser = argparse.ArgumentParser(
        description="Convert XDOF/ABC-130k MCAP episodes to Zarr episodes."
    )
    parser.add_argument(
        "--source",
        choices=["local", "hf"],
        default="local",
        help="Read episode from a local directory or stream it from the Hub.",
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        required=True,
        help=(
            "local: episode directory containing episode.mcap[, annotation.mcap]. "
            "hf: in-repo episode dir, e.g. data/train/<task>/episode_<uuid>."
        ),
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="XDOF/ABC-130k",
        help="Hugging Face dataset repo id (used when --source hf).",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default="",
        help="HF token; falls back to the HF_TOKEN environment variable.",
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second.")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Root output directory."
    )
    parser.add_argument(
        "--arm", type=str, choices=["left", "right", "both"], default="both"
    )
    parser.add_argument("--save-mp4", type=str2bool, default=False)
    parser.add_argument(
        "--chunk-timesteps",
        type=int,
        default=100,
        help="Timesteps per zarr chunk for numeric arrays.",
    )
    parser.add_argument("--task-name", type=str, default="")

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(argument_parse())
