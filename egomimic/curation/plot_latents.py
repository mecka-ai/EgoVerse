"""Interactive latent-space visualizer for saved curation NPZ files.

Loads latents_{task_name}.npz (written by curateModal.py) and produces
an interactive Plotly HTML showing the joint [state ∥ action] space —
the exact space on which KSG operates.

Usage
-----
python -m egomimic.curation.plot_latents \\
    /mnt/outputs/latents/latents_fold_clothes.npz \\
    --out /tmp/fold_clothes_latents.html \\
    --method umap        # umap | tsne (default: umap if available, else tsne)
    --n-points 5000      # max points to plot (default: all)

The NPZ must contain at minimum: state (N, Ds), action (N, Da).
Optional: lengths (E,) episode lengths, hashes (E,) episode hash strings,
language (N, Dl), language_texts (N,).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _load_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    out: dict = {}
    for k in data.files:
        out[k] = data[k]
    return out


def _episode_labels(data: dict, n: int) -> np.ndarray:
    """Build per-frame episode index array from lengths, or arange fallback."""
    if "lengths" in data:
        lengths = data["lengths"].astype(int)
        return np.repeat(np.arange(len(lengths)), lengths)
    return np.arange(n)


def _reduce(X: np.ndarray, method: str, seed: int, n_components: int = 2) -> np.ndarray:
    if method == "umap":
        try:
            import umap  # type: ignore
            reducer = umap.UMAP(n_components=n_components, random_state=seed, n_jobs=1)
            return reducer.fit_transform(X.astype(np.float32))
        except ImportError:
            print("[plot_latents] umap-learn not installed — falling back to t-SNE")
            method = "tsne"

    from sklearn.manifold import TSNE  # type: ignore
    perplexity = min(30, max(5, len(X) // 10))
    return TSNE(
        n_components=n_components,
        random_state=seed,
        perplexity=perplexity,
        n_iter=1000,
    ).fit_transform(X.astype(np.float32))


def _episode_color_seq(n_episodes: int) -> list[str]:
    """Generate n_episodes distinct HSL colors."""
    import colorsys
    colors = []
    for i in range(n_episodes):
        h = i / max(n_episodes, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.85)
        colors.append(f"rgb({int(r*255)},{int(g*255)},{int(b*255)})")
    return colors


def _make_traces(
    coords: np.ndarray,
    ep_idx: np.ndarray,
    hashes: np.ndarray | None,
    frame_in_ep: np.ndarray,
    n_episodes: int,
):
    import plotly.graph_objects as go  # type: ignore

    palette = _episode_color_seq(n_episodes)
    traces = []
    for ep in range(n_episodes):
        mask = ep_idx == ep
        if not mask.any():
            continue
        ep_hash = hashes[ep][:8] if hashes is not None else str(ep)
        x, y = coords[mask, 0], coords[mask, 1]
        fi = frame_in_ep[mask]
        traces.append(
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    size=4,
                    color=fi,
                    colorscale="Viridis",
                    showscale=False,
                    line=dict(width=0),
                    opacity=0.7,
                ),
                name=f"ep {ep_hash}",
                text=[f"ep={ep_hash} frame={f}" for f in fi],
                hovertemplate="%{text}<extra></extra>",
                legendgroup=f"ep{ep}",
            )
        )
    return traces


def plot_latents(
    npz_path: str,
    out_path: str | None = None,
    method: str = "auto",
    n_points: int | None = None,
    seed: int = 42,
) -> str:
    try:
        import plotly.graph_objects as go  # type: ignore
        from plotly.subplots import make_subplots  # type: ignore
    except ImportError:
        print("plotly is required: pip install plotly")
        sys.exit(1)

    if method == "auto":
        try:
            import umap  # noqa: F401
            method = "umap"
        except ImportError:
            method = "tsne"

    print(f"[plot_latents] loading {npz_path}")
    data = _load_npz(npz_path)

    state = data["state"].astype(np.float32)       # (N, Ds)
    action = data["action"].astype(np.float32)     # (N, Da)
    n = len(state)

    ep_idx = _episode_labels(data, n)
    n_episodes = int(ep_idx.max()) + 1
    hashes: np.ndarray | None = data.get("hashes", None)

    # Per-episode frame index for temporal coloring
    frame_in_ep = np.zeros(n, dtype=np.int32)
    for ep in range(n_episodes):
        mask = ep_idx == ep
        frame_in_ep[mask] = np.arange(mask.sum())

    # Optional subsample
    if n_points is not None and n > n_points:
        rng = np.random.default_rng(seed)
        sel = rng.choice(n, n_points, replace=False)
        sel.sort()
        state = state[sel]
        action = action[sel]
        ep_idx = ep_idx[sel]
        frame_in_ep = frame_in_ep[sel]
        n = n_points
        print(f"[plot_latents] subsampled to {n} points")

    joint = np.hstack([state, action])             # (N, Ds+Da)
    has_lang = "language" in data and data["language"] is not None

    panels = [
        ("joint [state ∥ action]", joint),
        ("state", state),
        ("action", action),
    ]
    if has_lang:
        lang = data["language"].astype(np.float32)
        if n_points is not None and len(lang) > n_points:
            lang = lang[sel]
        panels.append(("language", lang))

    n_cols = len(panels)
    fig = make_subplots(
        rows=1,
        cols=n_cols,
        subplot_titles=[p[0] for p in panels],
        shared_xaxes=False,
        horizontal_spacing=0.04,
    )

    for col, (title, X) in enumerate(panels, start=1):
        print(f"[plot_latents] {method.upper()} on {title} ({X.shape}) ...")
        coords = _reduce(X, method, seed)
        traces = _make_traces(coords, ep_idx, hashes, frame_in_ep, n_episodes)
        show_legend = col == 1
        for trace in traces:
            trace.showlegend = show_legend
            fig.add_trace(trace, row=1, col=col)

    task_name = Path(npz_path).stem.removeprefix("latents_")
    fig.update_layout(
        title=dict(
            text=f"{task_name} — latent space ({method.upper()}, {n} points, {n_episodes} episodes)",
            font=dict(size=15),
        ),
        height=600,
        width=350 * n_cols,
        legend=dict(
            itemsizing="constant",
            tracegroupgap=0,
            font=dict(size=9),
        ),
        margin=dict(l=40, r=20, t=80, b=40),
        template="plotly_dark",
    )
    for col in range(1, n_cols + 1):
        fig.update_xaxes(showticklabels=False, row=1, col=col)
        fig.update_yaxes(showticklabels=False, row=1, col=col)

    if out_path is None:
        out_path = Path(npz_path).with_suffix(".html").name
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"[plot_latents] wrote {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive latent-space viewer")
    parser.add_argument("npz", help="Path to latents_{task}.npz")
    parser.add_argument("--out", default=None, help="Output HTML path")
    parser.add_argument(
        "--method",
        default="auto",
        choices=["auto", "umap", "tsne"],
        help="Dimensionality reduction method (default: umap if installed, else tsne)",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=None,
        help="Max points to plot (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    plot_latents(args.npz, args.out, args.method, args.n_points, args.seed)


if __name__ == "__main__":
    main()
