"""Alignment metrics for token/chunk/span action embeddings (metrics.json per run).

Turns "does the new tokenizer cluster better?" from eyeballing t-SNEs into numbers
trackable across checkpoints:

  language_nmi            k-means on span embeddings vs language-cluster labels (NMI)
  lang_knn_acc            kNN accuracy predicting the language cluster from the
                          embedding, at token / chunk / span level
  same_span_locality      fraction of a token's kNN sharing its span, with the
                          chance-level expectation and the lift (observed / expected)
  fsq                     codebook entropy (bits), perplexity, usage fraction
  tokidx_map_nmi          k-means on the projected map vs token position (NMI) — how
                          much the map encodes chunk position rather than content
"""

from __future__ import annotations

import numpy as np


def _subsample(n: int, cap: int, seed: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, cap, replace=False))


def _knn_indices(X: np.ndarray, k: int):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X))).fit(X)
    _, idx = nn.kneighbors(X)
    return idx[:, 1:]  # drop self


def _knn_label_acc(X: np.ndarray, y: np.ndarray, k: int, cap: int, seed: int):
    """Leave-one-out kNN majority-vote accuracy of labels y from embeddings X."""
    keep = np.flatnonzero(y >= 0)
    if len(keep) < k + 2 or len(np.unique(y[keep])) < 2:
        return None
    sel = keep[_subsample(len(keep), cap, seed)]
    Xs, ys = np.asarray(X[sel], dtype=np.float32), y[sel]
    nbr = ys[_knn_indices(Xs, k)]
    pred = np.array([np.bincount(row).argmax() for row in nbr])
    return float((pred == ys).mean())


def _locality(X: np.ndarray, groups: np.ndarray, k: int, cap: int, seed: int):
    """Observed same-group fraction among kNN vs chance; lift = observed / expected."""
    if len(X) < k + 2:
        return None
    sel = _subsample(len(X), cap, seed)
    Xs, gs = np.asarray(X[sel], dtype=np.float32), groups[sel]
    same = float((gs[_knn_indices(Xs, k)] == gs[:, None]).mean())
    _, counts = np.unique(gs, return_counts=True)
    n = len(gs)
    expected = float((counts * (counts - 1)).sum() / max(1, n * (n - 1)))
    return {"observed": same, "expected": expected,
            "lift": (same / expected) if expected > 0 else None}


def _kmeans_nmi(X: np.ndarray, labels: np.ndarray, seed: int, cap: int = 20000):
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score
    keep = np.flatnonzero(labels >= 0)
    ks = np.unique(labels[keep])
    if len(keep) < 10 or len(ks) < 2:
        return None
    sel = keep[_subsample(len(keep), cap, seed)]
    pred = KMeans(n_clusters=len(ks), n_init=4, random_state=seed).fit_predict(
        np.asarray(X[sel], dtype=np.float32))
    return float(normalized_mutual_info_score(labels[sel], pred))


def compute_alignment_metrics(
    *,
    span_emb: np.ndarray,
    span_lang: np.ndarray,
    chunk_emb: np.ndarray | None = None,
    chunk_lang: np.ndarray | None = None,
    token_emb: np.ndarray | None = None,
    token_lang: np.ndarray | None = None,
    token_span: np.ndarray | None = None,
    coords: np.ndarray | None = None,
    coords_tok_idx: np.ndarray | None = None,
    codes: np.ndarray | None = None,
    codebook_size: int | None = None,
    seed: int = 0,
    knn_k: int = 10,
    sample_cap: int = 20000,
) -> dict:
    """Compute every applicable alignment metric; inapplicable ones come back None."""
    out: dict = {"knn_k": knn_k, "sample_cap": sample_cap}

    out["language_nmi"] = _kmeans_nmi(span_emb, span_lang, seed, sample_cap)
    out["lang_knn_acc"] = {
        "span": _knn_label_acc(span_emb, span_lang, knn_k, sample_cap, seed),
        "chunk": (_knn_label_acc(chunk_emb, chunk_lang, knn_k, sample_cap, seed)
                  if chunk_emb is not None and chunk_lang is not None else None),
        "token": (_knn_label_acc(token_emb, token_lang, knn_k, sample_cap, seed)
                  if token_emb is not None and token_lang is not None else None),
    }
    out["same_span_locality"] = (
        _locality(token_emb, token_span, knn_k, sample_cap, seed)
        if token_emb is not None and token_span is not None else None
    )

    if codes is not None and len(codes):
        flat = np.asarray(codes).reshape(-1)
        counts = np.bincount(flat, minlength=int(codebook_size or flat.max() + 1))
        p = counts[counts > 0] / counts.sum()
        entropy = float(-(p * np.log2(p)).sum())
        out["fsq"] = {
            "entropy_bits": entropy,
            "perplexity": float(2 ** entropy),
            "codebook_size": int(codebook_size) if codebook_size else None,
            "codes_used": int((counts > 0).sum()),
            "usage": (float((counts > 0).sum() / codebook_size) if codebook_size else None),
        }
    else:
        out["fsq"] = None

    if coords is not None and coords_tok_idx is not None and (coords_tok_idx >= 0).any():
        from sklearn.cluster import KMeans
        from sklearn.metrics import normalized_mutual_info_score
        keep = np.flatnonzero(coords_tok_idx >= 0)
        sel = keep[_subsample(len(keep), sample_cap, seed)]
        ntok = len(np.unique(coords_tok_idx[sel]))
        if ntok >= 2:
            pred = KMeans(n_clusters=ntok, n_init=4, random_state=seed).fit_predict(
                np.asarray(coords[sel], dtype=np.float32))
            out["tokidx_map_nmi"] = float(
                normalized_mutual_info_score(coords_tok_idx[sel], pred))
        else:
            out["tokidx_map_nmi"] = None
    else:
        out["tokidx_map_nmi"] = None

    return out
