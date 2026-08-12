"""
WAM (World-Action Model) — dreamzero port, integrated as an EgoMimic ``Algo``.

The model is dreamzero's **CausalWanModel**: a Wan video DiT in which the state
and (noised) action chunk are appended as register tokens, so ONE forward over
the joint sequence ``[video latent | action | state]`` predicts BOTH the video
flow-matching velocity and the action flow-matching velocity. Training is the
two-target rectified-flow loss; inference samples action + future frame jointly.

Integration mirrors HPT / Pi0.5:
  - ``WAMModel(nn.Module)`` owns the DiT + VAE + scheduler and the joint loss /
    sampler (this is the dreamzero ``WANPolicyHead`` logic, ported).
  - ``WAM(Algo)`` implements the 5 trainer hooks (process_batch_for_training,
    forward_training, forward_eval, compute_losses, log_info) and the
    per-embodiment batch plumbing — identical contract to HPT.

Hydra ``_target_: egomimic.algo.wam.WAM`` (see hydra_configs/model/wam_bc_human_wan21_1_3b.yaml).
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import override

from egomimic.algo.algo import Algo
from egomimic.models.wam_nets import FlowMatchScheduler
from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id


class WAMModel(nn.Module):
    """CausalWanModel + VAE + flow-matching loss/sampler (ported from WANPolicyHead)."""

    def __init__(
        self,
        dit=None,  # CausalWanModel (built by build_wam_dit)
        vae=None,  # frozen WanVideoVAE
        action_dim: int = 12,
        action_horizon: int = 100,
        state_dim: int = 12,
        frame_size: int = 256,  # square world-model input resolution (fallback)
        target_h: int = None,  # non-square override (e.g. 160 for Wan2.2 5B)
        target_w: int = None,  # non-square override (e.g. 320 for Wan2.2 5B)
        num_video_frames: int = 2,  # latent frames: 1 conditioning + 1 predicted
        text_len: int = 512,
        text_dim: int = 4096,
        world_loss_weight: float = 1.0,
        num_inference_steps: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.device = None
        self.dit = dit
        self.vae = vae
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.state_dim = state_dim
        self.frame_size = frame_size
        # Non-square target (Wan2.2 5B needs 160x320); default to the square frame_size.
        self.target_h = target_h or frame_size
        self.target_w = target_w or frame_size
        self.num_video_frames = num_video_frames
        self.text_len = text_len
        self.text_dim = text_dim
        self.world_loss_weight = world_loss_weight
        self.num_inference_steps = num_inference_steps

        # Rectified-flow scheduler (dreamzero settings); 1000 train timesteps.
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)

    # --- preprocessing ------------------------------------------------------
    def _frame_to_video(self, frames):
        """frames -> (B,C,T,H,W) in [-1,1] at frame_size. Accepts a single frame
        (B,C,H,W) or a WAM clip (B,T,C,H,W) from the windowed loader."""
        x = frames.float()
        if x.dim() == 4:
            x = x.unsqueeze(2)  # (B,C,H,W) -> (B,C,1,H,W)
        elif x.dim() == 5:
            x = x.permute(0, 2, 1, 3, 4)  # clip (B,T,C,H,W) -> (B,C,T,H,W)
        if x.max() > 1.5:
            x = x / 255.0
        B, C, T, H, W = x.shape
        x = F.interpolate(
            x.reshape(B * T, C, H, W),
            size=(self.target_h, self.target_w),
            mode="bilinear",
            align_corners=False,
        ).reshape(B, C, T, self.target_h, self.target_w)
        return (x * 2.0 - 1.0).to(self.device)

    @torch.no_grad()
    def _encode(self, frames):
        """frames -> VAE latent (B, Cz, F, h, w)."""
        return self.vae.encode(self._frame_to_video(frames))

    def _zero_context(self, B, device, dtype):
        # No-language BC: null text conditioning (DiT cross-attends to zeros).
        return torch.zeros(B, self.text_len, self.text_dim, device=device, dtype=dtype)

    def _prep_state_action(self, data):
        state = data["state"].float().to(self.device)  # (B, S, state_dim)
        if state.dim() == 2:
            state = state.unsqueeze(1)
        action = data["action"].float().to(self.device)  # (B, A, action_dim)
        action = action[:, : self.action_horizon, : self.action_dim]
        return state, action

    # --- training: two-target rectified-flow loss (dreamzero WANPolicyHead) -
    def compute_loss(self, batch):
        data = batch["data"]
        sched = self.scheduler
        latents = self._encode(data["video"])  # (B, Cz, F, h, w)
        # CausalWanModel needs >=2 latent frames (frame0 conditioning + frame1
        # predicted) so num_image_blocks == num_action_blocks == 1. Single obs
        # frame -> duplicate temporally (world target = obs frame until
        # multi-frame future clips are loaded).
        if latents.shape[2] < self.num_video_frames:
            latents = latents[:, :, :1].repeat(1, 1, self.num_video_frames, 1, 1)
        B, Cz, Fl, h, w = latents.shape
        device = latents.device
        state, action = self._prep_state_action(data)
        A = action.shape[1]

        # ---- timesteps (video per-frame; action coupled to the frame) ------
        ts_v_id = torch.randint(0, sched.num_train_timesteps, (B, Fl))
        ts_v = sched.timesteps[ts_v_id].to(device)  # (B, F) values
        ts_a_id = ts_v_id[:, :1].expand(B, A)  # coupled: share frame t
        ts_a = sched.timesteps[ts_a_id].to(device)  # (B, A)

        # ---- noise video latent + build velocity target --------------------
        lat_bf = latents.transpose(1, 2)  # (B, F, Cz, h, w)
        noise = torch.randn_like(lat_bf)
        noisy = sched.add_noise(
            lat_bf.flatten(0, 1), noise.flatten(0, 1), ts_v.flatten(0, 1)
        ).unflatten(0, (B, Fl))
        target_v = sched.training_target(lat_bf, noise, ts_v).transpose(
            1, 2
        )  # (B,Cz,F,h,w)
        x = noisy.transpose(1, 2)  # (B,Cz,F,h,w) model input

        # ---- noise action chunk + target -----------------------------------
        noise_a = torch.randn_like(action)
        noisy_a = sched.add_noise(
            action.flatten(0, 1), noise_a.flatten(0, 1), ts_a.flatten(0, 1)
        ).unflatten(0, (B, A))
        target_a = sched.training_target(action, noise_a, ts_a)  # (B,A,action_dim)

        seq_len = Fl * (h // 2) * (w // 2)
        context = self._zero_context(B, device, x.dtype)
        emb = torch.zeros(B, dtype=torch.long, device=device)

        # ---- joint DiT forward: video velocity + action velocity -----------
        # clean_x = the clean obs latents (teacher forcing): the model sees the
        # observed frames as causal context and learns to predict the FUTURE
        # frame velocity (dreamzero passes clean_x too), instead of unconditional
        # denoising. This is what anchors predictions to the real scene.
        video_pred, action_pred = self.dit(
            x,
            ts_v,
            ts_a,
            context,
            seq_len,
            action=noisy_a,
            state=state,
            embodiment_id=emb,
            clean_x=latents,
        )

        # ---- losses (rectified-flow MSE, timestep-weighted) ----------------
        if target_v.shape != video_pred.shape:  # guard odd dims
            target_v = target_v[..., : video_pred.shape[3], : video_pred.shape[4]]
        dyn = F.mse_loss(video_pred.float(), target_v.float(), reduction="none").mean(
            dim=(1, 3, 4)
        )
        w_dyn = sched.training_weight(ts_v.flatten()).reshape(B, Fl).to(device)
        world_loss = (dyn * w_dyn).mean()

        act = F.mse_loss(action_pred.float(), target_a.float(), reduction="none").mean(
            dim=2
        )
        w_act = sched.training_weight(ts_a.flatten()).reshape(B, A).to(device)
        action_loss = (act * w_act).mean()

        loss = self.world_loss_weight * world_loss + action_loss
        return loss, {"action_loss": action_loss, "world_loss": world_loss}

    # --- inference: jointly sample action chunk + future frame --------------
    @torch.no_grad()
    def sample(self, data):
        sched = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        sched.set_timesteps(self.num_inference_steps, training=False)
        latents0 = self._encode(data["video"])
        if latents0.shape[2] < self.num_video_frames:
            latents0 = latents0[:, :, :1].repeat(1, 1, self.num_video_frames, 1, 1)
        B, Cz, Fl, h, w = latents0.shape
        device = latents0.device
        state, _ = self._prep_state_action(data)
        seq_len = Fl * (h // 2) * (w // 2)
        context = self._zero_context(B, device, latents0.dtype)
        emb = torch.zeros(B, dtype=torch.long, device=device)

        # Image-to-video conditioning: frame 0 = the CLEAN observed latent; only
        # the future frames are sampled from noise (anchored to the real scene),
        # so the predicted video shows the actual observation evolving forward
        # rather than unconditional content.
        video = torch.randn_like(latents0)
        video[:, :, :1] = latents0[:, :, :1]
        action = torch.randn(B, self.action_horizon, self.action_dim, device=device)
        for t in sched.timesteps:
            ts_v = t.to(device).expand(B, Fl).clone()
            ts_v[:, 0] = 0.0  # frame 0 is clean (conditioning)
            ts_a = t.to(device).expand(B, self.action_horizon)
            v_vel, a_vel = self.dit(
                video,
                ts_v,
                ts_a,
                context,
                seq_len,
                action=action,
                state=state,
                embodiment_id=emb,
            )
            video = sched.step(v_vel, t, video)
            video[:, :, :1] = latents0[:, :, :1]  # keep the anchor clean
            action = sched.step(a_vel, t, action)
        frames = self.vae.decode(video)  # (B,C,F,H,W) in [-1,1]
        return action, frames


class WAM(Algo):
    """EgoMimic Algo adapter for the WAM (CausalWanModel) world-action model."""

    def __init__(
        self,
        norm_stats,
        camera_transforms=None,
        train_image_augs=None,
        eval_image_augs=None,
        dit=None,
        vae=None,
        action_dim: int = 12,
        action_horizon: int = 100,
        state_dim: int = 12,
        frame_size: int = 256,
        target_h: int = None,
        target_w: int = None,
        world_loss_weight: float = 1.0,
        num_inference_steps: int = 16,
        domains: list = None,
        ac_keys: dict = None,
        **kwargs,
    ):
        self.norm_stats = norm_stats
        self.camera_transforms = camera_transforms
        self.domains = domains.copy()
        self.ac_keys = ac_keys or {}
        self.device = kwargs.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        model = WAMModel(
            dit=dit,
            vae=vae,
            action_dim=action_dim,
            action_horizon=action_horizon,
            state_dim=state_dim,
            frame_size=frame_size,
            target_h=target_h,
            target_w=target_w,
            world_loss_weight=world_loss_weight,
            num_inference_steps=num_inference_steps,
        )
        model.device = self.device

        # per-embodiment key bookkeeping (same as HPT)
        self.camera_keys, self.proprio_keys = {}, {}
        for embodiment in self.domains:
            eid = get_embodiment_id(embodiment)
            self.camera_keys[eid] = [
                k
                for k in norm_stats.keys_of_type("camera_keys", eid)
                if norm_stats.is_key_with_embodiment(k, eid)
            ]
            self.proprio_keys[eid] = [
                k
                for k in norm_stats.keys_of_type("proprio_keys", eid)
                if norm_stats.is_key_with_embodiment(k, eid)
            ]
            if embodiment in self.ac_keys:
                self.ac_keys[eid] = self.ac_keys[embodiment]

        self.nets = nn.ModuleDict({"policy": model}).float().to(self.device)
        self._eval_frames = {}  # stashed predicted frames for the evaluator
        self.training_step = 0

    # --- Algo interface -----------------------------------------------------
    @override
    def process_batch_for_training(self, batch):
        processed = {}
        for embodiment_name, _batch in batch.items():
            eid = get_embodiment_id(embodiment_name)
            processed[eid] = {}
            for key, value in _batch.items():
                processed[eid][self.norm_stats.zarr_key_to_keyname(key, eid)] = value
            ac_key = self.ac_keys[eid]
            B, S, _ = processed[eid][ac_key].shape
            processed[eid]["pad_mask"] = torch.ones(B, S, 1, device=self.device)
            processed[eid]["embodiment"] = torch.tensor(
                [eid], device=self.device, dtype=torch.int64
            )
            for k, v in processed[eid].items():
                if isinstance(v, torch.Tensor):
                    v = v.to(self.device)
                    processed[eid][k] = v.float() if v.is_floating_point() else v
        return processed

    def _to_wam_data(self, eid, batch):
        """EgoMimic batch -> {video, state, action} for the CausalWanModel."""
        data = {}
        # video: first camera frame
        for key in self.camera_keys[eid]:
            if key in batch:
                data["video"] = batch[key]  # (B, C, H, W)
                break
        # state: concat proprio keys -> (B, S, state_dim)
        states = []
        for key in self.proprio_keys[eid]:
            if key in batch:
                s = batch[key]
                states.append(s.unsqueeze(1) if s.dim() == 2 else s)
        if states:
            data["state"] = torch.cat(states, dim=-1)
        data["action"] = batch[self.ac_keys[eid]]  # (B, A, action_dim)
        return data

    @override
    def forward_training(self, batch):
        predictions = OrderedDict()
        self.training_step += 1
        for eid, _batch in batch.items():
            name = get_embodiment(eid).lower()
            data = self._to_wam_data(eid, _batch)
            loss, parts = self.nets["policy"].compute_loss(
                {"domain": name, "data": data}
            )
            predictions[f"{name}_loss"] = loss
            for pk, pv in parts.items():
                predictions[f"{name}_{pk}"] = pv
        return predictions

    @override
    def forward_eval(self, batch):
        unnorm_preds = {}
        self._eval_frames = {}
        for eid, _batch in batch.items():
            name = get_embodiment(eid).lower()
            ac_key = self.ac_keys[eid]
            data = self._to_wam_data(eid, _batch)
            # val loss (same objective as training)
            loss, parts = self.nets["policy"].compute_loss(
                {"domain": name, "data": data}
            )
            unnorm_preds[f"{name}_loss"] = loss
            for pk, pv in parts.items():
                unnorm_preds[f"{name}_{pk}"] = pv
            # jointly sample action chunk + future frame
            action, frames = self.nets["policy"].sample(data)
            self._eval_frames[eid] = frames  # for WAMEvalVideo
            ref = _batch[ac_key]
            B, T, D = ref.shape
            preds = OrderedDict({ac_key: action[:, :T, :D]})
            for key, val in self.norm_stats.unnormalize(preds, eid).items():
                unnorm_preds[f"{name}_{key}"] = val
        return unnorm_preds

    @override
    def compute_losses(self, predictions, batch):
        total = torch.tensor(0.0, device=self.device)
        loss_dict = OrderedDict()
        for eid in batch:
            name = get_embodiment(eid).lower()
            total = total + predictions[f"{name}_loss"]
            for suffix in ("loss", "action_loss", "world_loss"):
                k = f"{name}_{suffix}"
                if k in predictions:
                    loss_dict[k] = predictions[k]
        loss_dict["action_loss"] = total / len(self.domains)
        return loss_dict

    @override
    def log_info(self, info):
        log = OrderedDict()
        log["Loss"] = info["losses"]["action_loss"].item()
        for k, v in info["losses"].items():
            log[k] = v.item() if torch.is_tensor(v) else v
        return log
