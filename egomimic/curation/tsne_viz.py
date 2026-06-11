"""Per-task t-SNE visualization of state/action latents.

Shared by the curation pipeline (curateModal.py) and the general latent-viz
export (latentVizModal.py) — nothing here depends on curation scoring.

For one task, projects the per-frame state and action latents to 2-D with t-SNE
and renders a scatter where:
  * each episode is a distinct hue (color-coded by episode), and
  * points darken with frame index within the episode (early = light, late =
    dark) — a temporal gradient.

Frames are subsampled every ``every_n``-th frame per episode to keep the plot
legible. One PNG is written per (task, modality), so a 14-task run yields 28
plots (state + action for each task).

Episode -> hue is derived from the episode's index in the latent list, and the
state and action lists are passed in the SAME episode order, so a given episode
is the SAME color in both its state and action plots.
"""

from __future__ import annotations

import colorsys
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _gather_points(latents_list: list, every_n: int, raw_index_lists: list | None = None):
    """Subsample every ``every_n``-th frame per episode.

    Returns ``(X (N, D), ep_index (N,), time_frac (N,), frame_idx (N,))`` where
    ``time_frac`` in [0, 1] is the frame's normalized position in its episode
    (0 = first frame, 1 = last) and ``frame_idx`` is the frame's RAW video-frame
    index when ``raw_index_lists`` (per-episode arrays row-aligned with the
    latents) is given, else the row index in the episode's latent sequence.
    Returns ``(None, None, None, None)`` if there are no frames.
    """
    pts: list = []
    ep_index: list = []
    time_frac: list = []
    frame_idx: list = []
    for i, lat in enumerate(latents_list):
        lat = np.asarray(lat)
        T = len(lat)
        if T == 0:
            continue
        raw = raw_index_lists[i] if raw_index_lists is not None else None
        for j in range(0, T, every_n):
            pts.append(lat[j])
            ep_index.append(i)
            time_frac.append(j / max(1, T - 1))
            frame_idx.append(int(raw[j]) if raw is not None else j)
    if not pts:
        return None, None, None, None
    return (
        np.asarray(pts, dtype=np.float64),
        np.asarray(ep_index, dtype=np.int64),
        np.asarray(time_frac, dtype=np.float64),
        np.asarray(frame_idx, dtype=np.int64),
    )


def _make_one(
    task_name: str,
    latents_list: list,
    modality: str,
    out_dir: Path,
    every_n: int,
    seed: int,
) -> str | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    X, ep_index, time_frac, _ = _gather_points(latents_list, every_n)
    if X is None or len(X) < 5:
        logger.warning(
            "tsne[%s/%s]: too few points (%s) — skipped",
            task_name, modality, 0 if X is None else len(X),
        )
        return None

    n = len(X)
    n_eps = len(latents_list)
    # t-SNE perplexity must be < n; scale with sample count, clamp to a sane range.
    perplexity = max(5.0, min(30.0, (n - 1) / 3.0))
    emb = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=seed,
    ).fit_transform(X)

    # Color: hue = episode (evenly spaced on the wheel), brightness darkens with
    # frame position (later frames darker). Saturation fixed.
    colors = [
        colorsys.hsv_to_rgb(
            ep_index[k] / max(1, n_eps),  # episode hue
            0.85,                          # saturation
            1.0 - 0.65 * time_frac[k],     # value: darker for later frames
        )
        for k in range(n)
    ]

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=14, linewidths=0)
    ax.set_title(
        f"t-SNE — {modality} latents — {task_name}\n"
        f"{n_eps} episodes · {n} frames (every {every_n}) · "
        f"hue = episode, darker = later frame"
    )
    ax.set_xticks([])
    ax.set_yticks([])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"tsne_{modality}_{task_name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        "tsne[%s/%s]: wrote %s (%d episodes, %d points, perplexity=%.1f)",
        task_name, modality, path, n_eps, n, perplexity,
    )
    return str(path)


def make_task_tsne_plots(
    task_name: str,
    state_latents: list,
    action_latents: list,
    out_dir: str | Path,
    every_n: int = 10,
    seed: int = 42,
) -> tuple[str | None, str | None]:
    """Render the state and action t-SNE PNGs for one task.

    Args:
        task_name: task label (used in title + filename).
        state_latents: list of per-episode ``(T, D)`` state-latent arrays.
        action_latents: list of per-episode ``(T, D)`` action-latent arrays,
            in the SAME episode order as ``state_latents`` (so episode i has the
            same hue in both plots).
        out_dir: directory to write the PNGs into.
        every_n: subsample stride over frames within each episode.
        seed: t-SNE random seed (deterministic embeddings).

    Returns:
        ``(state_png_path, action_png_path)``; either entry is ``None`` if that
        plot was skipped (too few points).
    """
    state_png = _make_one(task_name, state_latents, "state", out_dir, every_n, seed)
    action_png = _make_one(task_name, action_latents, "action", out_dir, every_n, seed)
    return state_png, action_png


def export_task_tsne3d(
    task_name: str,
    state_latents: list,
    action_latents: list,
    episode_hashes: list[str],
    out_dir: str | Path,
    every_n: int = 10,
    seed: int = 42,
    raw_index_lists: list | None = None,
) -> str | None:
    """Export 3-D t-SNE coords + point metadata for the interactive viewer.

    Writes ``tsne3d_<task>.json`` containing, for each modality (state/action),
    parallel arrays ``x/y/z`` (3-D t-SNE coords), ``ep`` (index into
    ``episodes``), ``frame``, and ``t`` (normalized time in [0, 1]).

    ``frame`` is the RAW video-frame index when ``raw_index_lists`` is given
    (one per-episode array of raw indices, row-aligned with the latents — the
    viewer seeks MP4s by raw frame, so pass it whenever the latents were
    pause/pose-filtered), else the row index in the episode's latent sequence.
    Episode order is identical across modalities so a viewer can color the
    same episode consistently.
    """
    import json

    from sklearn.manifold import TSNE

    out: dict = {"task": task_name, "episodes": list(episode_hashes), "every_n": every_n}
    for modality, latents_list in (("state", state_latents), ("action", action_latents)):
        X, ep_index, time_frac, frame_idx = _gather_points(latents_list, every_n, raw_index_lists)
        if X is None or len(X) < 5:
            logger.warning(
                "tsne3d[%s/%s]: too few points — skipped", task_name, modality
            )
            continue
        n = len(X)
        perplexity = max(5.0, min(30.0, (n - 1) / 3.0))
        emb = TSNE(
            n_components=3,
            perplexity=perplexity,
            init="pca",
            random_state=seed,
        ).fit_transform(X)
        out[modality] = {
            "x": [round(float(v), 3) for v in emb[:, 0]],
            "y": [round(float(v), 3) for v in emb[:, 1]],
            "z": [round(float(v), 3) for v in emb[:, 2]],
            "ep": ep_index.tolist(),
            "frame": frame_idx.tolist(),
            "t": [round(float(v), 4) for v in time_frac],
        }
        logger.info("tsne3d[%s/%s]: %d points embedded", task_name, modality, n)

    if "state" not in out and "action" not in out:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"tsne3d_{task_name}.json"
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    logger.info("tsne3d[%s]: wrote %s", task_name, path)
    return str(path)
