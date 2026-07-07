"""Model registry: maps --model-type to a RolloutPolicy implementation.

Add a new model (Molmo2, or any other frontier model) by implementing
``RolloutPolicy`` (base.py) and registering it here — nothing in
yam_rollout.py or DAgger.py needs to change. Note that a model needs an
action-producing head to act as a rollout policy at all; a plain
vision-language model (points/text) isn't one on its own.
"""
from __future__ import annotations

from .pi05_policy import Pi05Policy

MODEL_REGISTRY = {
    "pi05": Pi05Policy,
}


def load_policy(model_type, policy_path, device=None):
    try:
        cls = MODEL_REGISTRY[model_type]
    except KeyError:
        raise ValueError(
            f"Unknown --model-type '{model_type}'. Available: {sorted(MODEL_REGISTRY)}"
        )
    return cls(policy_path, device=device)
