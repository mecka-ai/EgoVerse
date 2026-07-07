"""The pluggable policy interface for YAM rollout/DAgger.

``PolicyRollout`` (yam_rollout.py) owns everything about the ROBOT —
embodiment, camera transforms, cam-frame<->base-frame conversion, chunk
resampling. A ``RolloutPolicy`` owns everything about the MODEL — checkpoint
format, batching, inference. Add a new model family (a real VLA, not pi0.5,
e.g. Molmo2 or another frontier model) by implementing this interface and
registering it in ``registry.py``; nothing in the rollout/DAgger loop needs to
change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import default_collate


class RolloutPolicy(ABC):
    @property
    def domains(self) -> list[str] | None:
        """Embodiment name(s) this checkpoint was trained on, if the model exposes one."""
        return None

    def use_6d_for(self, embodiment_id: int) -> bool:
        """Whether this embodiment's action representation is continuous-6D vs ypr euler."""
        return False

    def make_collate(self, default_prompt: str | None):
        """Build the collate_fn used to batch one transformed obs sample for this model."""
        return default_collate

    @abstractmethod
    def predict_chunk(self, collated_batch, embodiment_name: str) -> np.ndarray:
        """Run one forward pass on an already-collated batch; return the raw (T, D) action chunk."""

    def reset(self) -> None:
        """Clear any open-loop model state between episodes/interventions (default: no-op)."""
