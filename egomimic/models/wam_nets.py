"""
Network building blocks for WAM (World-Action Model) — the dreamzero port.

The model is the **CausalWanModel** (a Wan video DiT extended with action+state
register tokens): one joint token sequence ``[video latent | action | state]``,
flow-matched jointly so a single forward predicts BOTH the next video-frame
velocity and the action-chunk velocity. The whole WAN stack is vendored under
``egomimic/models/wan/``; this module is the builder/import surface the WAM algo
(``egomimic/algo/wam.py``) and Hydra configs use.

Builders (used as Hydra ``_target_``):
  - ``build_wam_dit``  -> CausalWanModel (loads the 1.3B Wan2.1 T2V DiT weights,
                          freezes the pretrained video blocks / optional LoRA,
                          leaves state/action/decoder modules trainable)
  - ``build_wan_vae``  -> frozen WanVideoVAE (frame <-> latent)

Re-exports: CausalWanModel, WANPolicyHead, FlowMatchScheduler,
FlowUniPCMultistepScheduler, WanVideoVAE, WanTextEncoder.
"""

import torch

from egomimic.models.wan.wan_video_dit import WanModel  # legacy (build_wan21_1_3b)

# --- WAN backbone (dreamzero) ----------------------------------------------
from egomimic.models.wan.wan_video_dit_action_casual_chunk import CausalWanModel
from egomimic.models.wan.wan_video_vae import WanVideoVAE

# ===========================================================================
# CausalWanModel (the WAM DiT) — 1.3B Wan2.1 T2V architecture preset
# ===========================================================================
# Canonical Wan2.1-T2V-1.3B DiT dims (dreamzero has no 1.3B config; these are
# the upstream 1.3B numbers, and they match the downloaded T2V-1.3B checkpoint).
WAM_DIT_1_3B = dict(
    model_type="t2v",
    patch_size=(1, 2, 2),
    text_len=512,
    in_dim=16,
    dim=1536,
    ffn_dim=8960,
    freq_dim=256,
    text_dim=4096,
    out_dim=16,
    num_heads=12,
    num_layers=30,
    qk_norm=True,
    cross_attn_norm=True,
    eps=1e-6,
    num_frame_per_block=1,
    concat_first_frame_latent=False,  # T2V: no first-frame channel concat
    hidden_size=1024,  # state/action encoder hidden width
    max_num_embodiments=1,  # single (human) embodiment
)


def build_wam_dit(
    checkpoint_path: str = None,
    action_dim: int = 12,
    max_state_dim: int = 12,
    num_action_per_block: int = 100,
    num_state_per_block: int = 1,
    frame_seqlen: int = 1560,
    freeze_video: bool = True,
    lora: bool = False,
    lora_rank: int = 4,
    lora_alpha: int = 4,
    **overrides,
):
    """Build the CausalWanModel WAM DiT at the 1.3B Wan2.1 T2V preset.

    - Loads the pretrained video DiT weights from ``checkpoint_path`` (the
      official T2V-1.3B safetensors), converted via the Wan civitai converter;
      the state/action/decoder register modules stay randomly initialized.
    - Trainability (memory): ``freeze_video`` freezes the pretrained blocks and
      trains only state/action/decoder (dreamzero keeps those trainable too);
      ``lora`` additionally injects LoRA adapters on the blocks (dreamzero's
      setup). ``num_action_per_block`` must equal the action horizon and
      ``num_state_per_block`` the state horizon (register-length invariant).
    """
    cfg = dict(WAM_DIT_1_3B)
    cfg.update(
        action_dim=action_dim,
        max_state_dim=max_state_dim,
        num_action_per_block=num_action_per_block,
        num_state_per_block=num_state_per_block,
        frame_seqlen=frame_seqlen,
    )
    cfg.update(overrides)
    model = CausalWanModel(**cfg)

    if checkpoint_path:
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            sd = load_file(checkpoint_path)
        else:
            sd = torch.load(checkpoint_path, map_location="cpu")
        try:  # official Wan weights -> WanModel naming (filters vace, matches hash)
            sd, _ = WanModel.state_dict_converter().from_civitai(sd)
        except Exception as e:  # noqa: BLE001
            print(f"[build_wam_dit] from_civitai skipped ({e}); loading raw")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(
            f"[build_wam_dit] loaded {checkpoint_path}: "
            f"{len(missing)} missing / {len(unexpected)} unexpected keys"
        )

    # --- trainability ------------------------------------------------------
    if lora:
        _inject_lora(model, lora_rank, lora_alpha)
        freeze_video = True  # lora handles the blocks; freeze the rest of them
    if freeze_video:
        model.requires_grad_(False)
        # The 3 robot modules are always trained (fresh, dreamzero-style).
        for mod in (model.state_encoder, model.action_encoder, model.action_decoder):
            mod.requires_grad_(True)
        if lora:  # re-enable LoRA adapter params
            for n, p in model.named_parameters():
                if "lora_" in n:
                    p.requires_grad_(True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[build_wam_dit] trainable params: {n_train/1e6:.1f}M "
        f"(lora={lora}, freeze_video={freeze_video})"
    )
    return model


def _inject_lora(model, rank, alpha, targets=("q", "k", "v", "o", "ffn.0", "ffn.2")):
    """Inject LoRA adapters on the DiT blocks (dreamzero: rank 4 / alpha 4 on
    attn q,k,v,o + the two FFN linears)."""
    from peft import LoraConfig, inject_adapter_in_model

    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=list(targets),
    )
    inject_adapter_in_model(cfg, model)


def build_wan_vae(checkpoint_path: str = None, z_dim: int = 16):
    """Build the vendored Wan 3D causal VAE (frozen) for the frame<->latent path.
    encode(video in [-1,1]) -> latent ; decode(latent) -> frames in [-1,1].
    """
    vae = WanVideoVAE(z_dim=z_dim)
    if checkpoint_path:
        sd = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        try:
            sd = WanVideoVAE.state_dict_converter().from_civitai(sd)
        except Exception:  # noqa: BLE001 — raw sd may already match
            pass
        missing, unexpected = vae.model.load_state_dict(sd, strict=False)
        print(
            f"[build_wan_vae] loaded {checkpoint_path}: "
            f"{len(missing)} missing / {len(unexpected)} unexpected keys"
        )
    vae.eval().requires_grad_(False)
    return vae


# ===========================================================================
# Legacy: plain (non-causal) WanModel builder for the simple action-only path.
# Kept for back-compat; the dreamzero WAM uses build_wam_dit above.
# ===========================================================================
WAN21_T2V_1_3B_CONFIG = dict(
    dim=1536,
    in_dim=16,
    ffn_dim=8960,
    out_dim=16,
    freq_dim=256,
    eps=1e-6,
    num_heads=12,
    num_layers=30,
    text_dim=4096,
    patch_size=[1, 2, 2],
    has_image_input=False,
)


def build_wan21_1_3b(checkpoint_path: str = None, **overrides):
    """(Legacy) plain Wan2.1 T2V-1.3B DiT (no action register tokens)."""
    model = WanModel(**{**WAN21_T2V_1_3B_CONFIG, **overrides})
    if checkpoint_path:
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            sd = load_file(checkpoint_path)
        else:
            sd = torch.load(checkpoint_path, map_location="cpu")
        try:
            sd, _ = WanModel.state_dict_converter().from_civitai(sd)
        except Exception as e:  # noqa: BLE001
            print(f"[build_wan21_1_3b] from_civitai skipped ({e}); loading raw sd")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(
            f"[build_wan21_1_3b] loaded {checkpoint_path}: "
            f"{len(missing)} missing / {len(unexpected)} unexpected keys"
        )
    return model
