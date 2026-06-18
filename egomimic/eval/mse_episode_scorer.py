"""Per-episode action-MSE scoring for a pretrained pi / pi0.5 checkpoint.

This is the reusable, Modal-agnostic core behind the MSE data-filtering method
(``egomimic/modal/scoreMseModal.py``). It runs the policy's ``forward_eval``
(diffusion sampling) over every episode resolved from a data config and computes
the **per-episode unnormalized paired action MSE** — the exact metric
``PIEvalVideo`` logs as ``Valid/*_paired_mse_avg``
(``egomimic/eval/eval_pi.py``), but attributed to each episode by scoring one
episode at a time.

Design notes
------------
* Episodes are enumerated with the regular ``ModalEpisodeResolver``
  (``resolver.resolve(filters=...) -> {episode_hash: ZarrDataset}``), exactly as
  ``curateModal.py`` does. Each ``ZarrDataset`` is frame-level, so per-episode
  attribution is free — we never need an episode id inside a batch.
* The model is built outside Lightning via ``ModelWrapper`` so we reuse the same
  instantiation path (including the ``pytorch_weight_path`` safetensors load in
  ``PI.__init__``). A fine-tuned Lightning ``.ckpt`` is layered on top.
* The collate is taken from the data config's ``MultiDataModuleWrapper`` so the
  batch the model sees is bit-identical to training/eval.
* MSE is accumulated as sum-of-squared-error / element-count across an episode's
  frames (correct for uneven last batches), equivalent to ``MeanSquaredError``.
"""

from __future__ import annotations

import logging
import math
from contextlib import nullcontext

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id
from egomimic.rldb.zarr.utils import DataSchematic

logger = logging.getLogger(__name__)

_AUTOCAST_DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": None,
    "fp32": None,
    "none": None,
}


def register_omegaconf_resolvers() -> None:
    """Register the resolvers trainHydra registers, without importing trainHydra.

    Importing ``egomimic.trainHydra`` registers ``eval`` / ``multiply`` at module
    load and would raise if they were already registered; replicate them here
    (idempotent via ``replace=True``) so composing a data config that uses
    ``${multiply:...}`` resolves cleanly.
    """
    try:
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        OmegaConf.register_new_resolver(
            "multiply", lambda x, y: int(float(x)) * int(float(y)), replace=True
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not register OmegaConf resolvers: %r", exc)


def _build_model_config_tree(cfg: DictConfig) -> DictConfig:
    """Local copy of ``trainHydra._build_model_config_tree``.

    Nulls ``robomimic_model.data_schematic`` so ``ModelWrapper`` injects the real
    one (the YAML default is an interpolation placeholder).
    """
    import copy

    model_cfg = copy.deepcopy(cfg.model)
    if (
        "robomimic_model" in model_cfg
        and isinstance(model_cfg.robomimic_model, DictConfig)
        and "data_schematic" in model_cfg.robomimic_model
    ):
        model_cfg.robomimic_model.data_schematic = None
    return OmegaConf.create({"model": model_cfg})


def build_collate_fn(cfg: DictConfig):
    """Return the exact collate the data config's datamodule would build.

    Reads the same tokenizer params ``MultiDataModuleWrapper.__init__`` reads from
    ``cfg.data`` and builds the collate directly — so we reuse the config-driven
    collate (tokenized or ``annotation_collate``) WITHOUT instantiating any
    datasets (passing ``train_datasets={}`` to instantiate would be a no-op merge
    and would still try to build — and hit SQL for — the real datasets).
    """
    from egomimic.pl_utils.pl_data_utils import (
        annotation_collate,
        build_tokenized_collate,
    )

    d = cfg.data
    if not bool(OmegaConf.select(d, "use_tokenizer", default=False)):
        return annotation_collate
    return build_tokenized_collate(
        max_length=OmegaConf.select(d, "collate_max_length", default=128),
        model_name=OmegaConf.select(
            d, "model_name", default="google/paligemma-3b-mix-224"
        ),
        sampling_mode=OmegaConf.select(d, "sampling_mode", default="random"),
        annotation_key=OmegaConf.select(d, "annotation_key", default=None),
        default_prompt=OmegaConf.select(d, "default_prompt", default=""),
        proprio_keys=OmegaConf.select(d, "proprio_keys", default=None),
        state_num_bins=OmegaConf.select(d, "state_num_bins", default=256),
        proprio=OmegaConf.select(d, "proprio", default=False),
        embodiment_label=OmegaConf.select(d, "embodiment_label", default=False),
        control_mode=OmegaConf.select(d, "control_mode", default=None),
    )


def build_data_schematic(
    cfg: DictConfig,
    embodiment_name: str,
    *,
    precomputed_norm_path: str | None,
) -> DataSchematic:
    """Instantiate the DataSchematic and load (precomputed) norm stats.

    Norm stats MUST match the training run's: the per-episode MSE is on
    *unnormalized* actions, so different stats make scores incomparable to a
    training run's ``Valid/*_paired_mse_avg``. We require ``precomputed_norm_path``
    and raise otherwise (PI needs no shape inference — only norm stats and the
    key-type tables, both available without a data pass).
    """
    data_schematic: DataSchematic = hydra.utils.instantiate(cfg.data_schematic)
    if not precomputed_norm_path:
        raise ValueError(
            "precomputed_norm_path is required for MSE scoring so the unnormalized "
            "MSE matches the training run's norm stats. Pass "
            "norm_stats.precomputed_norm_path=precomputed_norm_stats/<run> "
            "(the same value used at training)."
        )
    # With a precomputed path set, infer_norm_from_dataset loads norm_stats.json
    # and returns without touching the dataset, so we can pass None.
    data_schematic.infer_norm_from_dataset(
        None, embodiment_name, precomputed_norm_path=precomputed_norm_path
    )
    emb_id = get_embodiment_id(embodiment_name)
    if not data_schematic.norm_stats.get(emb_id):
        raise ValueError(
            f"No norm stats loaded for embodiment '{embodiment_name}' (id={emb_id}) "
            f"from {precomputed_norm_path}/norm_stats.json — check the path and that "
            f"it contains stats for embodiment id {emb_id}."
        )
    return data_schematic


def build_pi_algo(
    cfg: DictConfig,
    data_schematic: DataSchematic,
    *,
    device: torch.device,
    ckpt_path: str | None = None,
):
    """Construct the PI algo outside Lightning, on ``device``, in eval mode.

    The pretrained π₀.₅ base loads automatically inside ``PI.__init__`` from
    ``model.robomimic_model.config.pytorch_weight_path``. A fine-tuned Lightning
    ``.ckpt`` (``ckpt_path``) is layered on top afterwards (it wins).
    """
    wrapper = ModelWrapper(
        config_tree=_build_model_config_tree(cfg),
        data_schematic_state=data_schematic.to_state(),
        viz_func={},
    )
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing, unexpected = wrapper.load_state_dict(ckpt["state_dict"], strict=False)
        logger.info(
            "Loaded fine-tuned weights from %s (%d missing, %d unexpected keys)",
            ckpt_path,
            len(missing),
            len(unexpected),
        )
    else:
        logger.info(
            "Using pretrained base weights from "
            "model.robomimic_model.config.pytorch_weight_path (no ckpt_path)."
        )

    algo = wrapper.model
    algo.device = device
    algo.nets.to(device)
    algo.nets.eval()
    return algo


def _frame_indices(n_frames: int, every_n: int, max_frames: int | None) -> list[int]:
    if n_frames <= 0:
        return []
    idx = list(range(0, n_frames, max(1, int(every_n))))
    if max_frames:
        idx = idx[: int(max_frames)]
    return idx


def score_one_episode(
    algo,
    zarr_ds,
    embodiment_name: str,
    collate_fn,
    *,
    batch_size: int = 16,
    every_n: int = 1,
    max_frames: int | None = None,
    device: torch.device | None = None,
    autocast_dtype: str | None = "bfloat16",
) -> tuple[float, int]:
    """Return ``(paired_mse, n_frames_scored)`` for one episode.

    Iterates the episode's frames in batches (direct indexing — GPU diffusion
    dominates, so a DataLoader per episode would only add worker-spawn overhead),
    wraps each batch as ``{embodiment_name: collated}`` (the embodiment is keyed
    by the dict key, mirroring ``CombinedLoader`` — see
    ``PI.process_batch_for_training``), runs ``forward_eval``, and accumulates
    SSE/count of the unnormalized predicted vs ground-truth action chunk.
    """
    device = device or algo.device
    indices = _frame_indices(len(zarr_ds), every_n, max_frames)
    if not indices:
        return float("nan"), 0

    ac_dtype = _AUTOCAST_DTYPES.get(str(autocast_dtype).lower(), torch.bfloat16)
    autocast_ctx = (
        torch.autocast("cuda", dtype=ac_dtype)
        if (device.type == "cuda" and ac_dtype is not None)
        else nullcontext()
    )

    sse = 0.0
    count = 0
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        samples = [zarr_ds[i] for i in chunk]
        collated = collate_fn(samples)
        batch = {embodiment_name: collated}

        with torch.inference_mode(), autocast_ctx:
            proc = algo.process_batch_for_training(batch)
            preds = algo.forward_eval(proc)

            for emb_id, _b in proc.items():
                ac_key = algo.ac_keys[emb_id]
                emb_name = get_embodiment(emb_id).lower()
                pred_key = f"{emb_name}_{ac_key}"
                if pred_key not in preds:
                    continue
                # Unnormalized paired MSE vs GT chunk — matches eval_pi.py.
                gt = algo.data_schematic.unnormalize_data(_b, emb_id)[ac_key]
                pred = preds[pred_key]
                diff = (pred.float().cpu() - gt.float().cpu()) ** 2
                sse += float(diff.sum())
                count += int(diff.numel())

    if count == 0:
        return float("nan"), 0
    return sse / count, len(indices)


def score_episode_map(
    algo,
    episodes: dict,
    embodiment_name: str,
    collate_fn,
    *,
    batch_size: int = 16,
    every_n: int = 1,
    max_frames: int | None = None,
    device: torch.device | None = None,
    autocast_dtype: str | None = "bfloat16",
    progress: str = "",
) -> dict[str, dict]:
    """Score every episode in ``{episode_hash: ZarrDataset}``.

    Per-episode failures are logged and skipped (a single bad episode must never
    abort a long scoring run). Returns ``{hash: {"mse": float, "n_frames": int}}``.
    """
    try:
        from tqdm import tqdm

        items = tqdm(episodes.items(), desc=progress or "scoring", total=len(episodes))
    except Exception:  # pragma: no cover - tqdm optional
        items = episodes.items()

    out: dict[str, dict] = {}
    for ep_hash, zarr_ds in items:
        try:
            mse, n_frames = score_one_episode(
                algo,
                zarr_ds,
                embodiment_name,
                collate_fn,
                batch_size=batch_size,
                every_n=every_n,
                max_frames=max_frames,
                device=device,
                autocast_dtype=autocast_dtype,
            )
            if math.isfinite(mse):
                out[ep_hash] = {"mse": float(mse), "n_frames": int(n_frames)}
            else:
                logger.warning("Episode %s scored non-finite MSE — skipping", ep_hash)
        except Exception as exc:
            logger.warning("Episode %s scoring FAILED (%r) — skipping", ep_hash, exc)
    return out
