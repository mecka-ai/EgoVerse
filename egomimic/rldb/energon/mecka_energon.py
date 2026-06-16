"""Feed the model from pre-baked per-sample tar shards via **Megatron-Energon**.

A drop-in alternative to the shufflebuffer/zarr paths: Energon ingests the SAME WebDataset-style
records (``<key>.jpg`` + ``<key>.npy``, one self-contained sample each) that the materializer
(`egomimic/modal/build_sb_shards.py`) produces, after a one-time ``energon prepare`` writes a
``.nv-meta`` sidecar beside the shards. A ``CrudeWebdataset`` cook rebuilds the raw key_map inputs
and runs the SAME ``transform_list``, so the per-sample dict is identical to the other loaders
(``observations.images.front_img_1`` (3,360,640) f32, ``actions_cartesian`` (100,12) f32,
``observations.state.ee_pose`` (12,) f32, ``embodiment`` / ``metadata.robot_name`` ints) — the
model, training loop, and norm-stats wiring are unchanged.

Why Energon over the shufflebuffer engine: it **streams byte-ranges** from the tars (never staging a
whole shard in RAM), ships a reservoir shuffle + cross-shard interleave + rank×worker sharding, and
adds **exact deterministic mid-epoch resume** (restore-by-key) — NVIDIA-maintained. Selectable as
``data=mecka_all_energon``.

Energon IS the loader, so this is wired in via :class:`EnergonMultiDataModuleWrapper` (in
``pl_utils/pl_data_utils.py``), which builds Energon's own ``SavableDataLoader`` per embodiment
instead of wrapping a torch ``DataLoader``. The objects in the data config's ``train_datasets`` are
:class:`EnergonShardDataset` config-holders: they answer ``dataset[0]`` for shape inference and carry
the per-embodiment Energon settings the datamodule needs to build the real loader.
"""
from __future__ import annotations

import io
import tarfile
from dataclasses import dataclass

import numpy as np
import torch
from megatron.energon import (
    Cooker,
    DefaultTaskEncoder,
    Sample,
    basic_sample_keys,
    stateless,
)

from egomimic.pl_utils.pl_data_utils import annotation_collate
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.human import Mecka

VIZ_IMAGE_KEY = "observations.images.front_img_1"
EMBODIMENT_NAME = "mecka_bimanual"

# Built once at import; re-imported (not pickled) in every Energon worker, so the cook stays a
# plain module-level @stateless function — picklable by reference.
_EID = get_embodiment_id(EMBODIMENT_NAME)
_TRANSFORM_LIST = Mecka.get_transform_list(mode="cartesian")


def decode_record(jpg_bytes: bytes, npy_bytes: bytes) -> dict:
    """Rebuild the raw key_map inputs from one record and run the SAME transform_list as every other
    loader. Identical output dict to ``SBShardIterableDataset._decode`` (sans norm_mode)."""
    import simplejpeg

    meta = np.load(io.BytesIO(npy_bytes), allow_pickle=True).item()
    data = {
        "left.action_ee_pose": torch.as_tensor(meta["action_l"], dtype=torch.float32),
        "right.action_ee_pose": torch.as_tensor(meta["action_r"], dtype=torch.float32),
        "left.obs_ee_pose": torch.as_tensor(meta["proprio_l"], dtype=torch.float32),
        "right.obs_ee_pose": torch.as_tensor(meta["proprio_r"], dtype=torch.float32),
        "obs_head_pose": torch.as_tensor(meta["proprio_head"], dtype=torch.float32),
        "embodiment": _EID,
        "metadata.robot_name": _EID,
    }
    img = simplejpeg.decode_jpeg(jpg_bytes)  # HWC uint8
    data[VIZ_IMAGE_KEY] = (
        torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
    )  # (3,360,640) f32
    for t in _TRANSFORM_LIST:  # → actions_cartesian etc.
        data = t.transform(data)
    # Emit float32 (pose transforms run in float64); leave the int embodiment/metadata scalars.
    for k, v in list(data.items()):
        if isinstance(v, np.ndarray):
            data[k] = torch.from_numpy(np.ascontiguousarray(v)).to(torch.float32)
        elif isinstance(v, torch.Tensor) and v.dtype == torch.float64:
            data[k] = v.to(torch.float32)
    return data


@dataclass
class RobotSample(Sample):
    """Energon sample carrying the decoded model-ready dict."""

    data: dict


@stateless
def cook(sample) -> "RobotSample":
    """CrudeWebdataset cook: decode one ``<key>.jpg`` + ``<key>.npy`` record into the model dict."""
    data = decode_record(sample["jpg"], sample["npy"])
    return RobotSample(**basic_sample_keys(sample), data=data)


class MeckaEnergonEncoder(DefaultTaskEncoder):
    """Decodes via the cook (raw bytes → model dict) and collates per-embodiment.

    ``batch`` returns the collated dict (``annotation_collate`` == ``default_collate`` + list-key
    passthrough, same collate the Lightning datamodule uses). The embodiment-name key is added by
    the ``CombinedLoader`` in the datamodule, yielding the ``{embodiment_name: batch}`` the model's
    ``process_batch_for_training`` consumes.
    """

    decoder = None  # raw bytes — we decode in cook
    cookers = [Cooker(cook)]

    def batch(self, samples):
        return annotation_collate([s.data for s in samples])


def _first_record(shard_dir: str) -> tuple[bytes, bytes]:
    """Read the first complete ``<key>.jpg`` + ``<key>.npy`` pair from the first tar — for the
    one-shot ``dataset[0]`` shape-inference call (no Energon machinery needed)."""
    import os

    tars = sorted(
        f for f in os.listdir(shard_dir) if f.endswith(".tar")
    )
    if not tars:
        raise RuntimeError(f"No *.tar shards found in {shard_dir}")
    parts: dict[str, dict] = {}
    with tarfile.open(os.path.join(shard_dir, tars[0]), "r") as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = m.name
            dot = name.find(".")
            key, ext = name[:dot], name[dot + 1 :]
            if ext not in ("jpg", "npy"):
                continue
            fobj = tf.extractfile(m)
            if fobj is None:
                continue
            parts.setdefault(key, {})[ext] = fobj.read()
            if "jpg" in parts[key] and "npy" in parts[key]:
                return parts[key]["jpg"], parts[key]["npy"]
    raise RuntimeError(f"No complete <key>.jpg+<key>.npy record in {tars[0]}")


class EnergonShardDataset:
    """Config-holder for one embodiment's Energon dataset.

    It is NOT a torch ``Dataset`` that gets wrapped in a ``DataLoader`` — Energon is the loader.
    :class:`EnergonMultiDataModuleWrapper` reads the fields here to build Energon's own
    ``SavableDataLoader``. This object only needs to (a) answer ``dataset[0]`` for trainHydra's
    shape inference and (b) carry the settings.

    Args:
        shard_dir: directory of ``*.tar`` shards WITH a ``.nv-meta`` sidecar (run ``energon
            prepare`` once).
        split_part: Energon split name (``train``/``val``); built once by ``energon prepare``.
        shuffle_buffer_size: reservoir size (samples) for Energon's ``ShuffleBufferDataset``.
        embodiment_name: embodiment for ``get_embodiment_id``.
        max_samples_per_sequence: max consecutive samples drawn from one shard before switching
            (None = drain the whole shard contiguously). A finite value interleaves across shards,
            which is what makes per-batch episode diversity independent of shard size — set it for
            large multi-episode shards so a worker doesn't emit a long correlated run from one shard.
        parallel_shard_iters: number of shards each worker opens simultaneously and shuffles between
            (None = Energon default, 16 in training). Higher = more distinct shards feeding the
            shuffle buffer = better mixing at a given buffer, at the cost of more open file handles.
        shuffle_over_epochs_multiplier: shuffle the shard-slice order over this many epochs (>=1).
    """

    def __init__(
        self,
        shard_dir: str,
        split_part: str = "train",
        shuffle_buffer_size: int = 10000,
        embodiment_name: str = EMBODIMENT_NAME,
        max_samples_per_sequence: int | None = None,
        parallel_shard_iters: int | None = None,
        shuffle_over_epochs_multiplier: int = 1,
    ):
        self.shard_dir = shard_dir
        self.split_part = split_part
        self.shuffle_buffer_size = shuffle_buffer_size
        self.embodiment_name = embodiment_name
        self.max_samples_per_sequence = max_samples_per_sequence
        self.parallel_shard_iters = parallel_shard_iters
        self.shuffle_over_epochs_multiplier = shuffle_over_epochs_multiplier
        self.embodiment_id = get_embodiment_id(embodiment_name)
        self.data_schematic = None
        self._epoch = 0

    # -- trainHydra / DataSchematic wiring (mirror the other datasets) ----------
    def set_data_schematic(self, data_schematic, bounds_slack: float = 0.0) -> None:
        self.data_schematic = data_schematic

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        # Approximate (Lightning progress only). Energon streams; exact length isn't needed.
        import os

        n_shards = len([f for f in os.listdir(self.shard_dir) if f.endswith(".tar")])
        return max(1, n_shards) * 16000

    def __getitem__(self, idx: int) -> dict:
        # trainHydra calls dataset[0] once for shape inference. Decode one record directly.
        jpg, npy = _first_record(self.shard_dir)
        return decode_record(jpg, npy)

    def encoder(self) -> MeckaEnergonEncoder:
        return MeckaEnergonEncoder()
