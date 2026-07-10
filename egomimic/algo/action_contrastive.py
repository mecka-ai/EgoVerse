"""InfoNCE self-supervised action-segment encoder.

Learns an embedding space where action segments with similar MOTION SHAPE cluster
across diverse episodes — without any labels. Supervision is instance
discrimination (SimCLR-style NT-Xent): the two augmented views of the same
annotation span are the positive pair; every other span in the batch is a
negative. The dataset's augmentations (crop / shared-hand rotation / noise, plus
ActionNorms' arc-resample + centroid + path-scale) define exactly which
differences the embedding must ignore — so the only signal left to match on is
the shape of the bimanual motion.

Telemetry (all surfaced to W&B via compute_losses — watch these like perplexity
on the VQ runs):
  pos_sim       mean cosine of positive pairs (alignment; -> 1 is good)
  neg_sim       mean cosine of negatives (uniformity; should stay well below pos)
  retrieval_acc in-batch top-1 view retrieval (the InfoNCE "accuracy")
  emb_std       mean per-dim std of normalized embeddings — COLLAPSE ALARM:
                near 0 means every segment maps to the same point.
"""

from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import override

from egomimic.algo.algo import Algo
from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id


class ActionEncoderTCN(nn.Module):
    """Temporal-CNN encoder: (B, L, D) action segment -> (B, embed_dim) embedding.

    Backbone (conv stack + linear) produces the embedding used downstream
    (curation, retrieval, clustering); a small MLP projection head is used ONLY
    for the contrastive loss, per SimCLR practice.
    """

    def __init__(
        self,
        action_dim: int,
        seq_len: int = 100,
        channels: tuple[int, ...] = (64, 128, 256),
        kernel_size: int = 5,
        embed_dim: int = 128,
        proj_dim: int = 128,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        layers, c_in = [], int(action_dim)
        for c_out in channels:
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size, stride=2, padding=pad),
                nn.GELU(),
            ]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        conv_len = int(seq_len)
        for _ in channels:
            conv_len = (conv_len + 1) // 2
        self.to_embed = nn.Linear(channels[-1] * conv_len, embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, D) -> (B, embed_dim) backbone embedding."""
        h = self.conv(x.transpose(1, 2)).flatten(start_dim=1)
        return self.to_embed(h)

    def project(self, emb: torch.Tensor) -> torch.Tensor:
        """Embedding -> L2-normalized projection for the InfoNCE loss."""
        return F.normalize(self.proj(emb), dim=-1)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float):
    """Symmetric NT-Xent (SimCLR) on L2-normalized projections z1, z2 (B, P).

    Returns (loss, pos_sim, neg_sim, retrieval_acc).
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2B, P)
    sim = z @ z.T / temperature  # (2B, 2B)
    eye = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(eye, float("-inf"))  # exclude self
    # positive of i is i+B (and vice versa)
    targets = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    loss = F.cross_entropy(sim, targets)

    with torch.no_grad():
        cos = z @ z.T  # cosine (z normalized)
        pos_sim = (z1 * z2).sum(dim=-1).mean()
        neg_mask = ~eye
        neg_mask[torch.arange(2 * B), targets] = False
        neg_sim = cos[neg_mask].mean()
        retrieval_acc = (sim.argmax(dim=-1) == targets).float().mean()
    return loss, pos_sim, neg_sim, retrieval_acc


class ActionContrastiveTrainer(Algo):
    """Train the action-segment encoder with InfoNCE over two-view span batches.

    Expects batches from ``ContrastiveSpanActionDataset``:
    ``{embodiment: {ac_key: (B, 2, L, D)}}``. Segments arrive already
    shape-normalized by ActionNorms, so no DataSchematic normalization is applied
    (both views must share the exact same input space).
    """

    def __init__(
        self,
        data_schematic,
        domains: List[str],
        ac_keys: Dict[str, str],
        encoder: ActionEncoderTCN,
        temperature: float = 0.15,
        viz_func: Optional[dict] = None,
        **kwargs,
    ):
        self.nets = nn.ModuleDict()
        self.data_schematic = data_schematic
        self.viz_func = viz_func
        self.domains = list(domains)
        self.ac_keys = dict(ac_keys)
        self.temperature = float(temperature)

        device_arg = kwargs.get("device")
        if device_arg is not None:
            self.device = torch.device(device_arg)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.ac_keys_by_id: Dict[int, str] = {}
        for embodiment in self.domains:
            embodiment_id = get_embodiment_id(embodiment)
            self.ac_keys_by_id[embodiment_id] = self.ac_keys[embodiment]

        self.nets["encoder"] = encoder
        self.nets = self.nets.float().to(self.device)
        self.training_step_count = 0

    @override
    def process_batch_for_training(self, batch):
        processed = {}
        for embodiment_name, _batch in batch.items():
            embodiment_id = get_embodiment_id(embodiment_name)
            ac_key = self.ac_keys_by_id[embodiment_id]
            if ac_key not in _batch:
                raise KeyError(
                    f"Action key {ac_key!r} missing for embodiment "
                    f"{embodiment_name!r}; available: {list(_batch.keys())}"
                )
            views = _batch[ac_key]
            if views.ndim != 4 or views.shape[1] < 2:
                raise ValueError(
                    f"Expected two-view batches (B, V>=2, L, D), got "
                    f"{tuple(views.shape)} — use ContrastiveSpanActionDataset."
                )
            processed[embodiment_id] = {
                ac_key: views.float().to(self.device),
                "embodiment": torch.tensor(
                    [embodiment_id], device=self.device, dtype=torch.int64
                ),
            }
        return processed

    @override
    def forward_training(self, batch):
        predictions = OrderedDict()
        self.training_step_count += 1
        enc = self.nets["encoder"]
        for embodiment_id, _batch in batch.items():
            name = get_embodiment(embodiment_id).lower()
            views = _batch[self.ac_keys_by_id[embodiment_id]]  # (B, V, L, D)
            B, V, L, D = views.shape
            emb = enc(views.reshape(B * V, L, D))  # (B*V, E)
            z = enc.project(emb).reshape(B, V, -1)
            loss, pos, neg, acc = nt_xent(z[:, 0], z[:, 1], self.temperature)
            predictions[f"{name}_infonce_loss"] = loss
            predictions[f"{name}_pos_sim"] = pos
            predictions[f"{name}_neg_sim"] = neg
            predictions[f"{name}_retrieval_acc"] = acc
            with torch.no_grad():
                predictions[f"{name}_emb_std"] = (
                    F.normalize(emb, dim=-1).std(dim=0).mean()
                )
        return predictions

    @override
    def forward_eval(self, batch):
        """Embeddings per view for downstream eval: {name_ackey: (B, V, E)}."""
        out = {}
        enc = self.nets["encoder"]
        with torch.inference_mode():
            for embodiment_id, _batch in batch.items():
                name = get_embodiment(embodiment_id).lower()
                ac_key = self.ac_keys_by_id[embodiment_id]
                views = _batch[ac_key]
                B, V, L, D = views.shape
                emb = enc(views.reshape(B * V, L, D)).reshape(B, V, -1)
                out[f"{name}_{ac_key}"] = emb
        return out

    def forward_eval_logging(self, batch):
        metrics: Dict[str, torch.Tensor] = {}
        for embodiment_id, _batch in batch.items():
            name = get_embodiment(embodiment_id).lower()
            views = _batch[self.ac_keys_by_id[embodiment_id]]
            with torch.inference_mode():
                enc = self.nets["encoder"]
                B, V, L, D = views.shape
                z = enc.project(enc(views.reshape(B * V, L, D))).reshape(B, V, -1)
                _, pos, neg, acc = nt_xent(z[:, 0], z[:, 1], self.temperature)
            metrics[f"{name}_pos_sim"] = pos
            metrics[f"{name}_retrieval_acc"] = acc
        return metrics, {}

    def visualize_preds(self, predictions, batch):
        return None

    @override
    def compute_losses(self, predictions, batch):
        total = torch.tensor(0.0, device=self.device)
        loss_dict = OrderedDict()
        for embodiment_id, _ in batch.items():
            name = get_embodiment(embodiment_id).lower()
            loss = predictions[f"{name}_infonce_loss"]
            loss_dict[f"{name}_infonce_loss"] = loss
            # telemetry (detached scalars for W&B)
            for k in ("pos_sim", "neg_sim", "retrieval_acc", "emb_std"):
                loss_dict[f"{name}_{k}"] = predictions[f"{name}_{k}"].detach()
            total = total + loss
        loss_dict["action_loss"] = total / max(len(self.domains), 1)
        return loss_dict

    @override
    def log_info(self, info):
        log = OrderedDict()
        log["Loss"] = info["losses"]["action_loss"].item()
        for k, v in info["losses"].items():
            log[k] = v.item()
        return log

    # ------------------------------------------------------------------
    # Curation interface (CheckpointActionEmbedder-compatible)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """Embed span trajectories ``(L, D)`` or ``(B, L, D)`` -> ``(E,)`` / ``(B, E)``.

        Input must be ActionNorms-normalized (applied upstream, same as training).
        Returns the backbone embedding (not the projection head output).
        """
        squeeze = actions.ndim == 2
        if squeeze:
            actions = actions.unsqueeze(0)
        emb = self.nets["encoder"](actions.float().to(self.device))
        return emb.squeeze(0) if squeeze else emb
