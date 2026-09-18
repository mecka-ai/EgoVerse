"""
Network building blocks for the cross-embodiment tactile encoder.

This module implements Phase 1 (architecture) of the tactile-encoder spec:
    - `PlatformAdapter`      : A_h / A_r, light platform-specific input adapters
    - `SharedBackboneEncoder`: E_shared, the shared spatio-temporal transformer backbone
    - `PlatformDecoder`      : D_h / D_r, light platform-specific reconstruction decoders
    - `ProjectionHead`       : g, the 2-layer GeLU alignment projection head
    - `TactileEncoderBackbone`: the frozen, decoder/projector-free module handed to a
      downstream policy at deployment time (Phase 4).

It also implements the pure-tensor math needed by Phase 2 (losses):
    - `patchify` / `unpatchify`      : spatio-temporal tube patch (de)tokenization
    - `random_masking`               : MAE-style random patch masking
    - `compute_proxy_features`       : non-dimensional proxy features phi(x)
    - `proxy_guided_cost_matrix`     : the C_ij cost matrix
    - `make_proxy_guided_cost_fn`    : the same cost, wrapped for geomloss.SamplesLoss

Kept dependency-light on purpose (torch + einops only): this is a standalone
representation-learning module, not a manipulation policy, so it does not need the
image/kinematics stack (cv2, timm, transformers, pytorch_kinematics, ...) pulled in
by `egomimic.models.hpt_nets` / `egomimic.utils.egomimicUtils`.
"""

import math
from typing import Optional, Tuple

import einops
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Patch (de)tokenization
# ---------------------------------------------------------------------------


def patchify(
    x: torch.Tensor, patch_t: int, patch_h: int, patch_w: int
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    Split a spatio-temporal tactile stream into flattened tube patches.

    Args:
        x: (B, T, C, H, W) raw tactile stream.
        patch_t, patch_h, patch_w: tube patch size along time / height / width.

    Returns:
        patches: (B, N, patch_dim) with N = (T/patch_t)*(H/patch_h)*(W/patch_w)
            and patch_dim = patch_t*patch_h*patch_w*C.
        grid: (nt, nh, nw), the number of patches along each axis (needed to
            invert the operation with `unpatchify`).
    """
    B, T, C, H, W = x.shape
    if T % patch_t or H % patch_h or W % patch_w:
        raise ValueError(
            f"Input shape (T={T}, H={H}, W={W}) is not divisible by patch size "
            f"(patch_t={patch_t}, patch_h={patch_h}, patch_w={patch_w})"
        )
    patches = einops.rearrange(
        x,
        "b (nt pt) c (nh ph) (nw pw) -> b (nt nh nw) (pt ph pw c)",
        pt=patch_t,
        ph=patch_h,
        pw=patch_w,
    )
    grid = (T // patch_t, H // patch_h, W // patch_w)
    return patches, grid


def unpatchify(
    patches: torch.Tensor,
    grid: Tuple[int, int, int],
    patch_t: int,
    patch_h: int,
    patch_w: int,
    channels: int,
) -> torch.Tensor:
    """Inverse of `patchify`: (B, N, patch_dim) -> (B, T, C, H, W)."""
    nt, nh, nw = grid
    return einops.rearrange(
        patches,
        "b (nt nh nw) (pt ph pw c) -> b (nt pt) c (nh ph) (nw pw)",
        nt=nt,
        nh=nh,
        nw=nw,
        pt=patch_t,
        ph=patch_h,
        pw=patch_w,
        c=channels,
    )


def sinusoidal_position_embedding(
    num_positions: int,
    dim: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Standard 1D sinusoidal position embedding table, shape (1, num_positions, dim)."""
    if dim % 2 != 0:
        raise ValueError(
            f"sinusoidal_position_embedding requires an even dim, got {dim}"
        )
    position = torch.arange(
        num_positions, dtype=torch.float32, device=device
    ).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / dim)
    )
    table = torch.zeros(num_positions, dim, dtype=torch.float32, device=device)
    table[:, 0::2] = torch.sin(position * div_term)
    table[:, 1::2] = torch.cos(position * div_term)
    return table.unsqueeze(0).to(dtype)


# ---------------------------------------------------------------------------
# MAE-style random masking
# ---------------------------------------------------------------------------


def random_masking(
    tokens: torch.Tensor,
    mask_ratio: float,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomly drop `mask_ratio` of the tokens per-sample (He et al. MAE masking).

    Args:
        tokens: (B, N, D) already position-embedded patch tokens.
        mask_ratio: fraction of tokens to mask out (e.g. 0.75).

    Returns:
        visible_tokens: (B, N_keep, D), the kept (unmasked) tokens.
        mask: (B, N) bool, True where the patch was masked out.
        ids_restore: (B, N) long, permutation restoring original patch order.
        ids_keep: (B, N_keep) long, indices (in original order) of the kept patches.
    """
    B, N, D = tokens.shape
    len_keep = max(1, N - int(round(N * mask_ratio)))

    noise = torch.rand(B, N, device=tokens.device, generator=generator)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    visible_tokens = torch.gather(
        tokens, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
    )

    mask = torch.ones(B, N, dtype=torch.bool, device=tokens.device)
    mask.scatter_(1, ids_keep, False)

    return visible_tokens, mask, ids_restore, ids_keep


# ---------------------------------------------------------------------------
# Proxy features + proxy-guided OT cost matrix
# ---------------------------------------------------------------------------


def compute_proxy_features(
    x: torch.Tensor, contact_threshold: float = 0.0, eps: float = 1e-6
) -> torch.Tensor:
    """
    Non-dimensional proxy features phi(x) used to guide the OT cost matrix.

    Args:
        x: (B, T, C, H, W) raw (non-negative, pressure-like) tactile stream.
        contact_threshold: taxel reading above which a cell counts as "in contact".
        eps: numerical stability constant for the relative-derivative ratio.

    Returns:
        phi: (B, 2) = [relative_dF_dt, contact_area_ratio]
            relative_dF_dt: mean_t(|dF/dt|) / mean_t(F), a dimensionless rate.
            contact_area_ratio: mean fraction of taxels in contact, in [0, 1].
    """
    B, T, C, H, W = x.shape
    total_force = x.sum(dim=(2, 3, 4))  # (B, T)

    if T > 1:
        dF_dt = total_force[:, 1:] - total_force[:, :-1]
        mean_abs_dF_dt = dF_dt.abs().mean(dim=1)
    else:
        mean_abs_dF_dt = torch.zeros(B, device=x.device, dtype=x.dtype)

    mean_force = total_force.mean(dim=1)
    relative_dF_dt = mean_abs_dF_dt / (mean_force.abs() + eps)

    in_contact = (x > contact_threshold).to(x.dtype)
    contact_area_ratio = in_contact.mean(dim=(1, 2, 3, 4))

    return torch.stack([relative_dF_dt, contact_area_ratio], dim=-1)


def proxy_guided_cost_matrix(
    z_h: torch.Tensor,
    z_r: torch.Tensor,
    phi_h: torch.Tensor,
    phi_r: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    C_ij = ||z_h_i - z_r_j||_2^2 + alpha * ||phi_h_i - phi_r_j||_2^2

    Args:
        z_h: (..., Bh, Dz) projected human latents.
        z_r: (..., Br, Dz) projected robot latents.
        phi_h: (..., Bh, Dp) human proxy features.
        phi_r: (..., Br, Dp) robot proxy features.
        alpha: weight on the proxy-feature term.

    Returns:
        C: (..., Bh, Br) cost matrix.

    Uses negative-axis broadcasting (rather than `unsqueeze(1)`/`unsqueeze(0)`)
    so this is correct both when called directly on unbatched (Bh, D) /
    (Br, D) tensors *and* when used as a `geomloss.SamplesLoss(cost=...)`
    callback, which invokes it on (1, Bh, D) / (1, Br, D) tensors with a
    prepended singleton batch axis -- with plain `unsqueeze(1)`/`unsqueeze(0)`
    that extra axis silently turns the intended (Bh, Br) outer product into a
    same-size elementwise comparison (wrong, but shape-compatible whenever
    Bh == Br) or a shape-mismatch crash whenever Bh != Br.
    """
    z_dist = ((z_h.unsqueeze(-2) - z_r.unsqueeze(-3)) ** 2).sum(dim=-1)
    phi_dist = ((phi_h.unsqueeze(-2) - phi_r.unsqueeze(-3)) ** 2).sum(dim=-1)
    return z_dist + alpha * phi_dist


def make_proxy_guided_cost_fn(d_proj: int, alpha: float):
    """
    Wrap `proxy_guided_cost_matrix` as a `cost(x, y)` callable for
    `geomloss.SamplesLoss(cost=...)`, where `x`/`y` are `[z_proj || phi]`
    concatenations (see `TactileEncoder.forward_training`, which builds them).
    """

    def cost_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z_x, phi_x = x[..., :d_proj], x[..., d_proj:]
        z_y, phi_y = y[..., :d_proj], y[..., d_proj:]
        return proxy_guided_cost_matrix(z_x, z_y, phi_x, phi_y, alpha)

    return cost_fn


# ---------------------------------------------------------------------------
# Phase 1 modules
# ---------------------------------------------------------------------------


class PlatformAdapter(nn.Module):
    """
    A_h / A_r: light platform-specific input adapter.

    Tokenizes a raw spatio-temporal tactile stream (B, T, C, H, W) into a
    sequence of patch tokens in the shared model dimension `d_model`, with
    additive sinusoidal position embeddings over the (fixed) patch grid.
    """

    def __init__(
        self,
        in_channels: int,
        patch_t: int,
        patch_h: int,
        patch_w: int,
        d_model: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_t = patch_t
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.d_model = d_model
        self.patch_dim = patch_t * patch_h * patch_w * in_channels

        self.proj = nn.Linear(self.patch_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

        # Lazily-built, cached position-embedding table. Not a persistent buffer:
        # it is a deterministic function of the (fixed, per-platform) patch grid,
        # so it is cheap to recompute and need not be checkpointed.
        self._pos_embed: Optional[torch.Tensor] = None
        self._pos_grid: Optional[Tuple[int, int, int]] = None

    def _position_embedding(
        self, grid: Tuple[int, int, int], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if self._pos_embed is None or self._pos_grid != grid:
            n = grid[0] * grid[1] * grid[2]
            self._pos_embed = sinusoidal_position_embedding(
                n, self.d_model, device=device, dtype=dtype
            )
            self._pos_grid = grid
        if self._pos_embed.device != device or self._pos_embed.dtype != dtype:
            self._pos_embed = self._pos_embed.to(device=device, dtype=dtype)
        return self._pos_embed

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
        """
        Args:
            x: (B, T, C, H, W) raw platform tactile stream.

        Returns:
            tokens: (B, N, d_model), position-embedded patch tokens.
            target_patches: (B, N, patch_dim), the raw flattened patch values
                (the MTAE reconstruction target -- pre-projection, pre-mask).
            grid: (nt, nh, nw) patch grid shape.
        """
        target_patches, grid = patchify(x, self.patch_t, self.patch_h, self.patch_w)
        tokens = self.norm(self.proj(target_patches))
        tokens = tokens + self._position_embedding(grid, x.device, tokens.dtype)
        return tokens, target_patches, grid


class SharedBackboneEncoder(nn.Module):
    """
    E_shared: the shared spatio-temporal transformer backbone.

    A single instance of this module is applied to *both* platforms' token
    sequences -- weight sharing is what makes the resulting latent space
    cross-embodiment. Pools a prepended learnable [CLS] token into the
    bottleneck latent z in R^{d_latent}.
    """

    def __init__(
        self,
        d_model: int,
        d_latent: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.bottleneck_proj = nn.Linear(d_model, d_latent)

    def forward(
        self, tokens: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (B, N, d_model) token sequence (already position-embedded).
            key_padding_mask: optional (B, N) bool, True at positions to ignore.

        Returns:
            z: (B, d_latent) bottleneck latent (from the [CLS] token).
            tokens_out: (B, 1+N, d_model) full encoder output sequence
                ([CLS] followed by the per-patch tokens), exposed in case a
                downstream consumer wants a token sequence rather than a
                single pooled vector (e.g. for cross-attention keys/values).
        """
        B = tokens.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)

        if key_padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=tokens.device)
            key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)

        tokens_out = self.encoder(seq, src_key_padding_mask=key_padding_mask)
        tokens_out = self.final_norm(tokens_out)
        z = self.bottleneck_proj(tokens_out[:, 0])
        return z, tokens_out


class PlatformDecoder(nn.Module):
    """
    D_h / D_r: light platform-specific reconstruction decoder.

    Reconstructs the *full* grid of patch values (visible and masked) for one
    platform from the shared bottleneck latent z, following the ACT-style
    idiom of concatenating a latent token onto a learnable query sequence and
    self-attending jointly.
    """

    def __init__(
        self,
        d_latent: int,
        d_model: int,
        patch_dim: int,
        num_patches: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        dim_feedforward = dim_feedforward or d_model * 4
        self.num_patches = num_patches

        self.z_proj = nn.Linear(d_latent, d_model)
        self.query_pos = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.query_pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, patch_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, d_latent) bottleneck latent.

        Returns:
            recon: (B, num_patches, patch_dim) reconstruction of every patch
                in the platform's grid (masked-patch selection happens outside,
                in the loss).
        """
        B = z.shape[0]
        z_tok = self.z_proj(z).unsqueeze(1)
        queries = self.query_pos.expand(B, -1, -1)
        seq = torch.cat([z_tok, queries], dim=1)
        out = self.decoder(seq)
        out = self.final_norm(out[:, 1:])
        return self.head(out)


class ProjectionHead(nn.Module):
    """g: 2-layer MLP with GeLU, mapping z -> z_proj in the alignment space."""

    def __init__(self, d_latent: int, d_hidden: int, d_proj: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_proj),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TactileEncoderBackbone(nn.Module):
    """
    Phase 4 hand-off module: input adapters (A_h, A_r) + shared backbone
    encoder (E_shared) only -- the reconstruction decoders and the alignment
    projection head are discarded.

    Frozen (all parameters `requires_grad_(False)`, permanently in `eval()`
    mode) so it can be embedded inside a downstream policy module (e.g.
    pi_0.5) without its stats drifting or its weights updating, and queried
    for the latent `z` (or the full token sequence) to feed into that
    policy's cross-attention layer alongside visual tokens.
    """

    def __init__(self, adapters: nn.ModuleDict, encoder: SharedBackboneEncoder):
        super().__init__()
        self.adapters = adapters
        self.encoder = encoder
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(
        self, platform: str, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            platform: key into `self.adapters` (e.g. "human" or "robot").
            x: (B, T, C, H, W) raw tactile stream for that platform.

        Returns:
            z: (B, d_latent) frozen bottleneck latent.
            tokens: (B, 1+N, d_model) full encoder token sequence ([CLS] + patches).
        """
        tokens, _, _ = self.adapters[platform](x)
        z, tokens_out = self.encoder(tokens)
        return z, tokens_out

    def train(self, mode: bool = True) -> "TactileEncoderBackbone":
        # Stay frozen/eval even if the enclosing policy module calls .train().
        return super().train(False)
