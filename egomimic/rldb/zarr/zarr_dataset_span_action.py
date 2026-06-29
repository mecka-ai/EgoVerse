"""Span-level action dataset: one item = one annotation span's action trajectory.

Enumerates annotation spans across episodes (``ZarrDataset._load_annotations`` →
``{text, start_idx, end_idx}``), reads the per-frame transformed action trajectory for
each span's ``[start, end)`` frame range, and ActionNorms-normalizes it to a fixed length
``L``. Yields ``(L, action_dim)`` tensors so a sequence autoencoder (TemporalCNNAutoencoder)
can train on span shapes via the standard trainHydra pipeline.

Per-frame actions are obtained from the same transform path the curation build uses
(``_collect_curation_batched(load_images=False)``), so action_dim / transforms match the
rest of the pipeline. Pause removal must be OFF so annotation indices match the action frames.
"""

from __future__ import annotations

import numpy as np
import torch

from egomimic.algo.action_norms import ActionNorms, ActionNormsSettings


class SpanActionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        resolver,
        action_key: str = "actions_cartesian",
        image_key: str = "observations.images.front_img_1",
        fixed_length: int = 100,
        norms: dict | ActionNormsSettings | None = None,
        **kwargs,
    ) -> None:
        resolved = resolver.resolve()
        # resolver.resolve() → {episode: ZarrDataset}; MultiDataset exposes .datasets
        if hasattr(resolved, "datasets"):
            episodes = dict(resolved.datasets)
        elif isinstance(resolved, dict):
            episodes = resolved
        else:
            raise TypeError(f"Unsupported resolver.resolve() result: {type(resolved)!r}")

        self.episodes = episodes
        self.action_key = action_key
        self.image_key = image_key
        self.fixed_length = int(fixed_length)
        if isinstance(norms, ActionNormsSettings):
            ns = norms
        elif isinstance(norms, dict):
            ns = ActionNormsSettings(**norms)
        else:
            ns = ActionNormsSettings(enabled=True, resample_len=fixed_length)
        self.norms = ActionNorms(ns)
        self._action_cache: dict[str, np.ndarray] = {}

        # Enumerate (episode, start, end) over all valid annotation spans.
        self.index: list[tuple[str, int, int]] = []
        for ep, ds in episodes.items():
            try:
                anns = ds._load_annotations()
            except Exception:
                anns = []
            for ann in anns:
                if not isinstance(ann, dict):
                    continue
                s, e = int(ann.get("start_idx", -1)), int(ann.get("end_idx", -1))
                if s >= 0 and e > s:
                    self.index.append((ep, s, e))

        self.data_schematic = None

    def __len__(self) -> int:
        return len(self.index)

    def set_data_schematic(self, data_schematic, bounds_slack: float = 0.0) -> None:
        # Fixed-length items — no horizon bounds to clamp; just hold the reference.
        self.data_schematic = data_schematic

    # ------------------------------------------------------------------
    def _episode_actions(self, ep: str) -> np.ndarray:
        """Full per-frame transformed action trajectory ``(T_full, action_dim)`` for an episode."""
        if ep in self._action_cache:
            return self._action_cache[ep]
        ds = self.episodes[ep]
        actions, _, _ = ds._collect_curation_batched(
            action_key=self.action_key,
            image_key=self.image_key,
            image_decode_workers=0,
            load_images=False,
        )
        a = np.asarray(actions, dtype=np.float32)
        if a.ndim == 3:           # (T, horizon, D) → executed per-frame action = first step
            a = a[:, 0, :]
        elif a.ndim > 3:
            a = a.reshape(a.shape[0], -1)
        self._action_cache[ep] = a
        return a

    def __getitem__(self, idx: int) -> dict:
        ep, s, e = self.index[idx]
        full = self._episode_actions(ep)
        T = full.shape[0]
        s2, e2 = max(0, min(s, T)), max(0, min(e, T))
        traj = full[s2:e2] if e2 > s2 else full[:1]
        norm = self.norms.apply(traj)  # (L, D)
        return {self.action_key: torch.from_numpy(np.asarray(norm, dtype=np.float32))}
