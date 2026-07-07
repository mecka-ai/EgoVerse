"""pi0.5 (openpi/robomimic ``ModelWrapper``) implementation of ``RolloutPolicy``.

Relocated verbatim from ``PolicyRollout`` in yam_rollout.py — same checkpoint
loading, torch.compile unwrap, and forward_eval call, just behind the
``RolloutPolicy`` interface so the rollout/DAgger loop doesn't need to know
it's pi0.5 underneath.
"""
from __future__ import annotations

import os

import torch

from egomimic.models.denoising_policy import DenoisingPolicy
from egomimic.pl_utils.pl_data_utils import build_tokenized_collate
from egomimic.pl_utils.pl_model import ModelWrapper

from .base import RolloutPolicy

# This file lives at <repo>/egomimic/robot/YAM/models/pi05_policy.py.
_THIS = os.path.dirname(os.path.abspath(__file__))                      # .../egomimic/robot/YAM/models
_EGOMIMIC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))  # .../egomimic


class Pi05Policy(RolloutPolicy):
    LOCAL_WEIGHT_PATH = os.path.join(
        _EGOMIMIC_DIR, "algo", "pi_checkpoints", "pi05_base_pytorch"
    )

    def __init__(self, policy_path, device=None):
        self.policy_path = policy_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = self._load_policy()

    @property
    def domains(self):
        return getattr(self.policy.model, "domains", None)

    def use_6d_for(self, embodiment_id):
        ac_key = self.policy.model.ac_keys[embodiment_id]
        conv = self.policy.model.action_registry.get(embodiment_id, ac_key)
        return "6D" in type(conv).__name__

    def make_collate(self, default_prompt):
        """Tokenizing collate with the SAME prompt format the checkpoint trained on.

        The pi0.5 training configs (e.g. data=yam_pick_hat_wrist_pi) set
        ``proprio: true`` + ``embodiment_label: true``, so every training prompt
        is ``"Task: <text>, Embodiment: <name>, State: <256-bin proprio>;\\nAction: "``.
        Rollout previously built the collate WITHOUT those flags, so the model
        was conditioned on a bare prompt it never saw in training — no Task
        anchor, no Embodiment block, and no discretized State splice (a proprio
        pathway the model learned to read). NOTE: proprio_keys must be passed
        explicitly here (this branch's collate has no default), and the batch's
        "embodiment" key must be the integer id for the Embodiment splice.
        """
        return build_tokenized_collate(
            max_length=128,
            model_name="google/paligemma-3b-mix-224",
            sampling_mode="first",
            annotation_key="annotations",
            default_prompt=default_prompt,
            proprio_keys=["observations.state.ee_pose"],
            state_num_bins=256,
            proprio=True,
            embodiment_label=True,
        )

    def predict_chunk(self, collated_batch, embodiment_name):
        batch = {embodiment_name: collated_batch}
        processed_batch = self.policy.model.process_batch_for_training(batch)
        preds = self.policy.model.forward_eval(processed_batch)[
            f"{embodiment_name}_actions_cartesian"
        ]
        return preds.detach().cpu().numpy().squeeze()

    def reset(self):
        self.policy.eval()

    @classmethod
    def _patch_checkpoint_paths(cls, ckpt_path):
        """Rewrite pytorch_weight_path in the checkpoint's saved config
        to point to the local base model weights."""
        import torch as _torch
        from omegaconf import DictConfig, OmegaConf
        ckpt = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ht = ckpt.get("hyper_parameters", {}).get("config_tree")
        if ht is None:
            return ckpt_path
        if isinstance(ht, DictConfig):
            cfg = OmegaConf.to_container(ht, resolve=True)
        else:
            cfg = ht
        # Navigate to pytorch_weight_path in the config
        robomimic = cfg.get("model", {}).get("robomimic_model", {})
        config = robomimic.get("config", {})
        old_path = config.get("pytorch_weight_path")
        if old_path is None or old_path == cls.LOCAL_WEIGHT_PATH:
            return ckpt_path
        print(f"[rollout] Patching pytorch_weight_path: {old_path} -> {cls.LOCAL_WEIGHT_PATH}")
        config["pytorch_weight_path"] = cls.LOCAL_WEIGHT_PATH
        ckpt["hyper_parameters"]["config_tree"] = OmegaConf.create(cfg)
        patched_path = ckpt_path + ".patched"
        _torch.save(ckpt, patched_path)
        print(f"[rollout] Patched checkpoint saved to {patched_path}")
        return patched_path

    def _load_policy(self):
        patched_path = self._patch_checkpoint_paths(self.policy_path)
        policy = ModelWrapper.load_from_checkpoint(
            patched_path, weights_only=False, map_location="cpu"
        )
        policy = policy.to(self.device)
        policy.eval()
        policy.model.device = self.device

        # Unwrap torch.compile on sample_actions to avoid massive first-call
        # compilation overhead (~50s). The compiled version (instance attribute)
        # shadows the original class method; deleting it restores the fast
        # uncompiled path which is sufficient for real-time rollout.
        pi0 = policy.model.nets["policy"]
        if "sample_actions" in vars(pi0):
            del pi0.sample_actions
            print("[rollout] Disabled torch.compile on sample_actions for rollout inference")

        # Verify model is on GPU
        try:
            p = next(pi0.parameters())
            print(f"[rollout] Model device: {p.device}, dtype: {p.dtype}")
            if not p.is_cuda:
                print("[rollout] WARNING: model is NOT on GPU — inference will be very slow!")
        except StopIteration:
            pass

        if getattr(policy.model, "diffusion", False):
            for head in policy.model.nets.policy.heads:
                if isinstance(policy.model.nets.policy.heads[head], DenoisingPolicy):
                    policy.model.nets.policy.heads[head].num_inference_steps = 10
        return policy
