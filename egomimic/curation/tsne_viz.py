"""Per-task t-SNE visualization of curation latents (state, action, language).

Supports multiple latent views:
  * ``state`` — image/state embeddings (default episode hue + temporal gradient).
  * ``action`` — action embeddings.
  * ``state_lang`` — concatenated [z_s, z_l] (matches concat KSG conditioning).
  * ``language`` — language embeddings only.
  * ``state_by_lang`` — t-SNE on state only, colored by instruction cluster.

When language latents are provided, extra PNGs and 3-D JSON modalities are emitted
automatically (configurable via :class:`TsneVizSettings`).
"""

from __future__ import annotations

import colorsys
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_STANDARD_MODALITIES = ("state", "action")
_LANGUAGE_MODALITIES = ("state_lang", "language", "state_by_lang")


@dataclass(frozen=True)
class TsneVizSettings:
    """Controls which t-SNE views are generated when language is available."""

    every_n: int = 10
    seed: int = 42
    include_state_lang: bool = True
    include_language: bool = True
    include_state_by_lang: bool = True
    # How to color the standard ``state`` panel when instruction texts exist.
    # ``auto`` → ``language`` for stratified scoring runs, else ``episode``.
    state_color_by: str = "auto"


def _resolve_state_color_by(
    state_color_by: str,
    language_mode: str | None,
    has_language_texts: bool,
) -> str:
    if state_color_by != "auto":
        return state_color_by
    if has_language_texts and language_mode == "stratified":
        return "language"
    return "episode"


def _hstack_episode_latents(
    left: list[np.ndarray],
    right: list[np.ndarray],
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for a, b in zip(left, right):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        if len(a) != len(b):
            raise ValueError(
                f"episode length mismatch for joint latents: {len(a)} vs {len(b)}"
            )
        if len(a) == 0:
            out.append(np.empty((0, a.shape[1] + b.shape[1]), dtype=np.float32))
        else:
            out.append(np.hstack([a, b]))
    return out


def _language_cluster_ids(texts_by_episode: list[list[str]]) -> tuple[list[int], list[str]]:
    """Map each frame's instruction text to a stable cluster id + label list."""
    label_to_id: dict[str, int] = {}
    labels: list[str] = []
    ids: list[int] = []
    for ep_texts in texts_by_episode:
        for text in ep_texts:
            key = text if text else "<empty>"
            if key not in label_to_id:
                label_to_id[key] = len(label_to_id)
            ids.append(label_to_id[key])
            labels.append(key)
    return ids, [label_to_id[k] for k in sorted(label_to_id, key=lambda k: label_to_id[k])]


def _gather_points(
    latents_list: list,
    every_n: int,
    *,
    language_texts_by_episode: list[list[str]] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, list[str] | None]:
    """Subsample frames; optionally attach per-point instruction strings."""
    pts: list[np.ndarray] = []
    ep_index: list[int] = []
    time_frac: list[float] = []
    frame_idx: list[int] = []
    lang_text: list[str] = [] if language_texts_by_episode is not None else None

    for i, lat in enumerate(latents_list):
        lat = np.asarray(lat)
        t = len(lat)
        if t == 0:
            continue
        ep_texts = (
            language_texts_by_episode[i]
            if language_texts_by_episode is not None
            else None
        )
        for j in range(0, t, every_n):
            pts.append(lat[j])
            ep_index.append(i)
            time_frac.append(j / max(1, t - 1))
            frame_idx.append(j)
            if lang_text is not None:
                lang_text.append(
                    ep_texts[j] if ep_texts is not None and j < len(ep_texts) else ""
                )

    if not pts:
        return None, None, None, None, None
    return (
        np.asarray(pts, dtype=np.float64),
        np.asarray(ep_index, dtype=np.int64),
        np.asarray(time_frac, dtype=np.float64),
        np.asarray(frame_idx, dtype=np.int64),
        lang_text,
    )


def _colors_episode(ep_index: np.ndarray, time_frac: np.ndarray, n_eps: int) -> list[str]:
    return [
        _rgb(
            _hsv(
                ep_index[k] / max(1, n_eps),
                0.85,
                1.0 - 0.65 * time_frac[k],
            )
        )
        for k in range(len(ep_index))
    ]


def _colors_time(time_frac: np.ndarray) -> list[str]:
    return [
        _rgb(_lerp([170, 215, 255], [10, 40, 90], float(time_frac[k])))
        for k in range(len(time_frac))
    ]


def _colors_language(lang_text: list[str]) -> tuple[list[str], list[str]]:
    """Return (point colors, unique labels in cluster-id order)."""
    label_to_id: dict[str, int] = {}
    for text in lang_text:
        key = text if text else "<empty>"
        if key not in label_to_id:
            label_to_id[key] = len(label_to_id)
    ordered = [None] * len(label_to_id)
    for key, idx in label_to_id.items():
        ordered[idx] = key
    unique_labels: list[str] = [x for x in ordered if x is not None]
    n = len(unique_labels)
    colors = [
        _rgb(_hsv(label_to_id[text if text else "<empty>"] / max(1, n), 0.8, 0.9))
        for text in lang_text
    ]
    return colors, unique_labels


def _hsv(h: float, s: float, v: float) -> tuple[float, float, float]:
    return colorsys.hsv_to_rgb(h, s, v)


def _rgb(c: tuple[float, float, float]) -> str:
    return f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"


def _lerp(a: list[float], b: list[float], t: float) -> tuple[float, float, float]:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def _make_one(
    task_name: str,
    latents_list: list,
    modality: str,
    out_dir: Path,
    every_n: int,
    seed: int,
    *,
    color_by: str = "episode",
    language_texts_by_episode: list[list[str]] | None = None,
    subtitle: str = "",
) -> str | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    gather_lang = language_texts_by_episode if color_by == "language" else None
    X, ep_index, time_frac, _, lang_text = _gather_points(
        latents_list,
        every_n,
        language_texts_by_episode=gather_lang,
    )
    if X is None or len(X) < 5:
        logger.warning(
            "tsne[%s/%s]: too few points (%s) — skipped",
            task_name,
            modality,
            0 if X is None else len(X),
        )
        return None

    n = len(X)
    n_eps = len(latents_list)
    perplexity = max(5.0, min(30.0, (n - 1) / 3.0))
    emb = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=seed,
    ).fit_transform(X)

    if color_by == "language" and lang_text is not None:
        colors, unique_labels = _colors_language(lang_text)
        color_note = f"color = instruction ({len(unique_labels)} clusters)"
    elif color_by == "time":
        colors = _colors_time(time_frac)
        color_note = "color = time (light→dark)"
    else:
        colors = _colors_episode(ep_index, time_frac, n_eps)
        color_note = "hue = episode, darker = later frame"

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=14, linewidths=0)
    extra = f"\n{subtitle}" if subtitle else ""
    ax.set_title(
        f"t-SNE — {modality} — {task_name}\n"
        f"{n_eps} episodes · {n} frames (every {every_n}) · {color_note}{extra}"
    )
    ax.set_xticks([])
    ax.set_yticks([])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"tsne_{modality}_{task_name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        "tsne[%s/%s]: wrote %s (%d episodes, %d points, color_by=%s)",
        task_name,
        modality,
        path,
        n_eps,
        n,
        color_by,
    )
    return str(path)


def _embed_modality(
    latents_list: list,
    every_n: int,
    seed: int,
    n_components: int,
    *,
    language_texts_by_episode: list[list[str]] | None = None,
) -> dict[str, Any] | None:
    from sklearn.manifold import TSNE

    X, ep_index, time_frac, frame_idx, lang_text = _gather_points(
        latents_list,
        every_n,
        language_texts_by_episode=language_texts_by_episode,
    )
    if X is None or len(X) < 5:
        return None

    n = len(X)
    perplexity = max(5.0, min(30.0, (n - 1) / 3.0))
    emb = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        init="pca",
        random_state=seed,
    ).fit_transform(X)

    payload: dict[str, Any] = {
        "x": [round(float(v), 3) for v in emb[:, 0]],
        "y": [round(float(v), 3) for v in emb[:, 1]],
        "ep": ep_index.tolist(),
        "frame": frame_idx.tolist(),
        "t": [round(float(v), 4) for v in time_frac],
    }
    if n_components == 3:
        payload["z"] = [round(float(v), 3) for v in emb[:, 2]]
    if lang_text is not None:
        payload["lang"] = lang_text
        lang_ids, lang_labels = _language_cluster_ids_from_flat(lang_text)
        payload["lang_id"] = lang_ids
        payload["lang_labels"] = lang_labels
    return payload


def _language_cluster_ids_from_flat(lang_text: list[str]) -> tuple[list[int], list[str]]:
    label_to_id: dict[str, int] = {}
    ids: list[int] = []
    for text in lang_text:
        key = text if text else "<empty>"
        if key not in label_to_id:
            label_to_id[key] = len(label_to_id)
        ids.append(label_to_id[key])
    ordered = [None] * len(label_to_id)
    for key, idx in label_to_id.items():
        ordered[idx] = key
    return ids, [x for x in ordered if x is not None]


def make_task_tsne_plots(
    task_name: str,
    state_latents: list,
    action_latents: list,
    out_dir: str | Path,
    every_n: int = 10,
    seed: int = 42,
    *,
    language_latents: list | None = None,
    language_texts_by_episode: list[list[str]] | None = None,
    language_mode: str | None = None,
    settings: TsneVizSettings | None = None,
) -> dict[str, str | None]:
    """Render 2-D t-SNE PNGs for one task.

    Returns a map ``{modality: png_path}`` (value ``None`` if skipped).
    """
    cfg = settings or TsneVizSettings(every_n=every_n, seed=seed)
    every_n = cfg.every_n
    seed = cfg.seed
    out = Path(out_dir)
    has_lang = language_latents is not None and len(language_latents) > 0
    has_texts = bool(language_texts_by_episode)
    state_color = _resolve_state_color_by(
        cfg.state_color_by, language_mode, has_texts
    )

    paths: dict[str, str | None] = {}
    paths["state"] = _make_one(
        task_name,
        state_latents,
        "state",
        out,
        every_n,
        seed,
        color_by=state_color,
        language_texts_by_episode=language_texts_by_episode,
    )
    paths["action"] = _make_one(
        task_name, action_latents, "action", out, every_n, seed, color_by="episode"
    )

    if not has_lang:
        return paths

    if cfg.include_state_lang:
        joint = _hstack_episode_latents(state_latents, language_latents)
        paths["state_lang"] = _make_one(
            task_name,
            joint,
            "state_lang",
            out,
            every_n,
            seed,
            color_by="episode",
            subtitle="latent = [state ∥ language]",
        )

    if cfg.include_language:
        lang_color = "language" if has_texts else "episode"
        paths["language"] = _make_one(
            task_name,
            language_latents,
            "language",
            out,
            every_n,
            seed,
            color_by=lang_color,
            language_texts_by_episode=language_texts_by_episode,
        )

    if cfg.include_state_by_lang and has_texts:
        paths["state_by_lang"] = _make_one(
            task_name,
            state_latents,
            "state_by_lang",
            out,
            every_n,
            seed,
            color_by="language",
            language_texts_by_episode=language_texts_by_episode,
            subtitle="t-SNE on state, colored by instruction",
        )

    return paths


def export_task_tsne3d(
    task_name: str,
    state_latents: list,
    action_latents: list,
    episode_hashes: list[str],
    out_dir: str | Path,
    every_n: int = 10,
    seed: int = 42,
    *,
    language_latents: list | None = None,
    language_texts_by_episode: list[list[str]] | None = None,
    language_mode: str | None = None,
    settings: TsneVizSettings | None = None,
) -> str | None:
    """Export 3-D t-SNE JSON for the interactive viewer."""
    import json

    cfg = settings or TsneVizSettings(every_n=every_n, seed=seed)
    every_n = cfg.every_n
    seed = cfg.seed
    has_lang = language_latents is not None and len(language_latents) > 0
    has_texts = bool(language_texts_by_episode)

    out: dict[str, Any] = {
        "task": task_name,
        "episodes": list(episode_hashes),
        "every_n": every_n,
        "language_enabled": has_lang,
        "language_mode": language_mode,
    }

    if has_texts:
        _, lang_labels = _language_cluster_ids(language_texts_by_episode)
        out["language_labels"] = lang_labels

    modalities: list[tuple[str, list, list[list[str]] | None]] = [
        ("state", state_latents, language_texts_by_episode if has_texts else None),
        ("action", action_latents, None),
    ]
    if has_lang:
        if cfg.include_state_lang:
            modalities.append(
                (
                    "state_lang",
                    _hstack_episode_latents(state_latents, language_latents),
                    None,
                )
            )
        if cfg.include_language:
            modalities.append(
                ("language", language_latents, language_texts_by_episode)
            )
        if cfg.include_state_by_lang and has_texts:
            modalities.append(
                ("state_by_lang", state_latents, language_texts_by_episode)
            )

    for name, latents, texts in modalities:
        block = _embed_modality(
            latents,
            every_n,
            seed,
            n_components=3,
            language_texts_by_episode=texts,
        )
        if block is None:
            logger.warning("tsne3d[%s/%s]: too few points — skipped", task_name, name)
            continue
        out[name] = block
        logger.info("tsne3d[%s/%s]: %d points embedded", task_name, name, len(block["x"]))

    if "state" not in out and "action" not in out:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"tsne3d_{task_name}.json"
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    logger.info("tsne3d[%s]: wrote %s (modalities=%s)", task_name, path, list(out.keys()))
    return str(path)
