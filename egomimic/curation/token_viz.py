"""Token-level span visualization pipeline for chunk tokenizers (QueST).

Modular stages, each independently cacheable on the outputs volume:

1. chunk plan  — snap-to-nearest-full-chunk tiling of annotation spans, deduped by
                 ``(episode, chunk_frame)`` across overlapping spans (no span ever
                 produces zero chunks; nothing is padded — a tile that would run past
                 the span end or the last stored chunk snaps back to a real full chunk).
2. chunk read  — one process per episode (spawn pool; the zarr transform pipeline is
                 CPU-bound so threads serialize on the GIL). Each worker slices only
                 the chunk frames its spans need and discards the full episode array;
                 the raw ``(T, horizon, D)`` chunk array is cached per episode under
                 ``<cache_dir>/chunks/<data_tag>/<episode>.npy``.
3. pool        — token | chunk | span point granularity. Pooling path (implemented
                 once): per-position standardize → mean over tokens (chunk embedding)
                 → mean over a span's chunks (span embedding).
4. preproc     — configurable stages before projection: center_by_position (token
                 granularity), l2norm, whiten, pre-PCA.
5. project     — tsne | umap | pca to 2-D/3-D.

Cache tiers (``cache_key`` of the exact upstream content):
  chunks      key = resolver config + action key          (per episode)
  tokens      key = chunk identity + checkpoint + horizon
  projection  key = tokens key + granularity + preproc + method/dims/seed/cap/balance
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Cache keys + npz cache helpers
# ---------------------------------------------------------------------------


def cache_key(obj) -> str:
    """Stable 16-hex content key for any JSON-serializable object."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def array_key(*arrays) -> str:
    """Stable 16-hex content key over raw array bytes (order-sensitive)."""
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()[:16]


def cache_load_npz(cache_dir: str, kind: str, key: str) -> dict | None:
    """Load ``<cache_dir>/<kind>/<key>.npz`` as a dict of arrays, or None."""
    if not cache_dir:
        return None
    p = Path(cache_dir) / kind / f"{key}.npz"
    if not p.exists():
        return None
    z = np.load(str(p), allow_pickle=True)
    return {k: z[k] for k in z.files}


def cache_save_npz(cache_dir: str, kind: str, key: str, **arrays) -> str:
    """Atomically write arrays to ``<cache_dir>/<kind>/<key>.npz``."""
    d = Path(cache_dir) / kind
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".{key}.{os.getpid()}.tmp.npz"
    np.savez_compressed(str(tmp), **arrays)
    final = d / f"{key}.npz"
    os.replace(str(tmp), str(final))
    return str(final)


# ---------------------------------------------------------------------------
# 1. Chunk plan — snap-to-nearest-full-chunk
# ---------------------------------------------------------------------------


def plan_span_chunks(start: int, end: int, n_valid: int, horizon: int):
    """Chunk-start frames covering span ``[start, end)`` with full-chunk snapping.

    The stored chunk at frame ``f`` covers ``[f, f + horizon)``; only frames
    ``0 .. n_valid-1`` hold a full stored chunk. Tiles at ``horizon`` stride from the
    span start; any tile that would run past the span end (or past the last stored
    chunk) snaps back so the window is a real full chunk — never padded. Every span
    with any valid data yields ≥ 1 chunk.

    Returns ``(frames, snapped, would_drop)`` where ``would_drop`` flags spans the old
    ``range(start, min(end, n_valid), horizon)`` tiling silently produced zero chunks for.
    """
    last = n_valid - 1
    if last < 0:
        return [], False, True
    hi = max(end - horizon, start)  # latest chunk start that keeps the window in the span
    frames, snapped = [], False
    for pos in range(start, max(end, start + 1), horizon):
        f = max(0, min(pos, hi, last))
        snapped |= f != pos
        frames.append(f)
    return sorted(set(frames)), snapped, start >= n_valid


def extract_span_sequence(arr: np.ndarray, s: int, e: int) -> np.ndarray:
    """Contiguous per-frame action sequence for span ``[s, e)`` from the stored chunk
    array ``(T, H, D)``.

    Global frame ``g`` lives at ``arr[min(g, T-1), g - min(g, T-1)]`` — NOT column 0
    (step-0 is a degenerate near-identity delta in wrist-frame representations). The
    last stored row covers the episode tail, so frames up to ``T-1+H`` are reachable.
    """
    n_valid, H = arr.shape[0], arr.shape[1]
    end_max = n_valid - 1 + H            # one past the last representable frame
    e2 = max(min(e, end_max), 1)
    s2 = min(max(s, 0), e2 - 1)
    g = np.arange(s2, e2)
    rows = np.minimum(g, n_valid - 1)
    return np.asarray(arr[rows, g - rows], dtype=np.float32)  # (T_span, D)


def resample_sequence(seq: np.ndarray, L: int) -> np.ndarray:
    """Uniform temporal resample (linear interp per dim) of ``(T, D)`` to ``(L, D)``.

    The temporal normalization: spans of any duration map to a fixed L-step sequence,
    removing duration/speed so the tokenizer sees one same-length input per span."""
    seq = np.asarray(seq, dtype=np.float32)
    T = len(seq)
    if T == L:
        return seq
    if T == 1:
        return np.repeat(seq, L, axis=0)
    src = np.linspace(0.0, T - 1.0, L)
    i0 = np.floor(src).astype(np.int64)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (src - i0).astype(np.float32)[:, None]
    return ((1.0 - w) * seq[i0] + w * seq[i1]).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Chunk read — spawn process pool, per-episode cache, per-episode slicing
# ---------------------------------------------------------------------------

_W: dict = {}  # per-worker-process state (set by _worker_init)


def _worker_init(repo_dir: str, resolver_cfg_json: str, action_key: str,
                 image_key: str, chunk_cache_dir: str, span_resample: bool,
                 arclen_distance: float | None) -> None:
    import sys
    if repo_dir and repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    _W.update(
        resolver_cfg_json=resolver_cfg_json,
        action_key=action_key,
        image_key=image_key,
        chunk_cache_dir=chunk_cache_dir,
        span_resample=span_resample,
        arclen_distance=arclen_distance,
    )


def _wrist_arc_cumsum(ds) -> np.ndarray:
    """Cumulative combined wrist-travel arc length per raw frame (metres).

    Mirrors ArcLengthResampleChunks' metric: the R^6 norm of both wrists' world
    translation deltas. Zero/invalid pose rows (tracking dropouts) hold the previous
    position so they add no distance — matching the transforms' zero-quat sanitizer.
    """
    store = ds.episode_reader._store
    sides = []
    for side in ("left", "right"):
        p = np.asarray(store[f"{side}.obs_wrist_pose"][:], dtype=np.float64)
        xyz = p[:, :3].copy()
        bad = ~np.isfinite(p).all(axis=1) | (np.abs(p).sum(axis=1) < 1e-9)
        for i in range(1, len(xyz)):
            if bad[i]:
                xyz[i] = xyz[i - 1]
        sides.append(xyz)
    xyz = np.concatenate(sides, axis=-1)
    d = np.linalg.norm(np.diff(xyz, axis=0), axis=-1)
    return np.concatenate([[0.0], np.cumsum(d)])


def _worker_episodes() -> dict:
    """Instantiate the resolver once per worker process, lazily (cache hits skip it)."""
    if "episodes" not in _W:
        import hydra
        from omegaconf import OmegaConf
        rc = OmegaConf.create(json.loads(_W["resolver_cfg_json"]))
        resolver = hydra.utils.instantiate(rc)
        resolved = resolver.resolve()
        _W["episodes"] = dict(resolved.datasets) if hasattr(resolved, "datasets") else dict(resolved)
    return _W["episodes"]


def _worker_read(job):
    """Read one episode's chunk array (cache-first), plan + slice its spans' chunks.

    job: ``(ep_hash, [(span_idx, start, end), ...], horizon)``
    Returns ``(ep, {key: (horizon, D) float32}, {span_idx: [keys]}, stats, err)`` where
    ``key`` is ``(frame,)`` for snap-tiled chunks or ``(start, end)`` for span-resampled
    chunks (one temporally-normalized chunk per span).
    """
    ep, spans, horizon = job
    try:
        arclen = _W["arclen_distance"]
        cache_dir = _W["chunk_cache_dir"]
        cache_p = Path(cache_dir) / f"{ep}.npz" if cache_dir else None
        if cache_p is not None and cache_p.exists():
            z = np.load(str(cache_p))
            arr, kept = z["arr"], z["kept"]
            s_cum = z["s_cum"] if "s_cum" in z.files else None
        else:
            episodes = _worker_episodes()
            if ep not in episodes:
                raise KeyError(f"episode not resolvable ({len(episodes)} episodes resolved)")
            ds = episodes[ep]
            actions, _, _ = ds._collect_curation_batched(
                action_key=_W["action_key"],
                image_key=_W["image_key"],
                image_decode_workers=0,
                load_images=False,
            )
            arr = np.asarray(actions, dtype=np.float32)  # (T_kept, stored_horizon, D)
            if arr.ndim != 3:
                raise ValueError(f"expected (T, horizon, D) chunks, got shape {arr.shape}")
            # Row i of arr is the chunk anchored at ORIGINAL frame kept[i] — transforms
            # (e.g. arclen still-anchor rejection, zero-pose filtering) drop rows, so
            # row index != frame index in general.
            kept = np.asarray(
                getattr(ds, "_curation_kept_indices", np.arange(len(arr))), dtype=np.int64)
            if len(kept) != len(arr):
                raise ValueError(f"kept-index map {len(kept)} != rows {len(arr)}")
            s_cum = _wrist_arc_cumsum(ds) if arclen else None
            if cache_p is not None:
                cache_p.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache_p.with_suffix(f".{os.getpid()}.tmp.npz")
                extra = {} if s_cum is None else {"s_cum": s_cum}
                np.savez(str(tmp), arr=arr, kept=kept, **extra)
                os.replace(str(tmp), str(cache_p))
        if arr.shape[1] < horizon:
            raise ValueError(f"stored horizon {arr.shape[1]} < requested {horizon}")
        if arclen and s_cum is None:
            raise ValueError("arclen cache entry missing s_cum — delete stale cache")

        n_valid = arr.shape[0]
        span_map: dict[int, list[tuple]] = {}
        chunks: dict[tuple, np.ndarray] = {}
        n_snapped = n_dropped_old = 0

        def _row_chunk(pos: int) -> np.ndarray:
            return np.array(arr[pos, :horizon], dtype=np.float32)

        if _W["span_resample"]:
            if arclen:
                raise ValueError("span_resample is incompatible with arc-length chunks "
                                 "(rows are arc samples, not frames)")
            # Temporal normalization: one chunk per span — the span's exact per-frame
            # sequence uniformly resampled to `horizon` steps. No chunk sharing between
            # spans, no out-of-span frames.
            for span_idx, s, e in spans:
                key = (int(s), int(e))
                if key not in chunks:
                    chunks[key] = resample_sequence(
                        extract_span_sequence(arr, int(s), int(e)), horizon)
                span_map[int(span_idx)] = [key]
        elif arclen:
            # Arc-length chunks: anchor at kept frames, each covering [anchor, end)
            # where end is the first frame reaching +`arclen` metres of combined wrist
            # travel. Tile a span end-to-end by true window extents (variable frames).
            for span_idx, s, e in spans:
                pos = int(np.searchsorted(kept, int(s), side="left"))
                if pos >= n_valid or kept[pos] >= int(e):
                    pos = max(0, min(pos, n_valid - 1) - (1 if pos >= n_valid or (pos > 0 and kept[pos] >= int(e)) else 0))
                    n_snapped += 1
                keys = []
                while True:
                    f = int(kept[pos])
                    end = int(np.searchsorted(s_cum, s_cum[min(f, len(s_cum) - 1)] + arclen, side="left"))
                    end = min(max(end, f + 1), len(s_cum) - 1)
                    key = (f, end)
                    if key not in chunks:
                        chunks[key] = _row_chunk(pos)
                    keys.append(key)
                    if end >= int(e):
                        break
                    nxt = int(np.searchsorted(kept, end, side="left"))
                    if nxt <= pos:
                        nxt = pos + 1
                    if nxt >= n_valid:
                        break
                    pos = nxt
                span_map[int(span_idx)] = keys
        else:
            # Position-space snap tiling: spans index ORIGINAL frames; map to kept-row
            # positions first (kept == arange when nothing was dropped).
            needed: set[int] = set()
            pos_of: dict[int, int] = {}
            for span_idx, s, e in spans:
                pos_lo = int(np.searchsorted(kept, int(s), side="left"))
                pos_hi = int(np.searchsorted(kept, int(e), side="left"))
                positions, snapped, would_drop = plan_span_chunks(
                    pos_lo, max(pos_hi, pos_lo + 1), n_valid, horizon)
                span_map[int(span_idx)] = [(int(kept[p]),) for p in positions]
                for pp in positions:
                    pos_of[int(kept[pp])] = pp
                n_snapped += int(snapped)
                n_dropped_old += int(would_drop)
                needed.update(int(kept[p]) for p in positions)
            chunks = {(f,): _row_chunk(pos_of[f]) for f in sorted(needed)}
        stats = {"spans": len(spans), "snapped": n_snapped, "dropped_old": n_dropped_old,
                 "n_valid": int(n_valid)}
        return ep, chunks, span_map, stats, None
    except Exception as exc:  # surfaced (with episode hash) by the parent — never silent
        return ep, None, None, None, f"{type(exc).__name__}: {exc}"


def read_span_chunks(resolver_cfg: dict, repo_dir: str, span_meta: list[dict],
                     horizon: int, cache_dir: str, tag: str,
                     action_key: str = "actions_cartesian",
                     image_key: str = "observations.images.front_img_1",
                     workers: int | None = None, span_resample: bool = False,
                     arclen_distance: float | None = None):
    """Read every span's action chunks with snapping + dedupe + caching.

    With ``span_resample`` each span becomes exactly ONE chunk: its per-frame sequence
    uniformly resampled to ``horizon`` steps (temporal normalization — no chunk sharing,
    no out-of-span frames). Otherwise spans are tiled at ``horizon`` stride with
    snap-to-nearest-full-chunk and deduped by ``(episode, frame)``.

    Returns ``(chunks, chunk_ep, chunk_frame, chunk_end, chunk_owner, span_chunks, info)``:
      chunks       (Nc, horizon, D) float32 — unique chunks across all spans
      chunk_ep     list[str] (Nc,)          — episode hash per chunk
      chunk_frame  (Nc,) int32              — chunk start frame per chunk
      chunk_end    (Nc,) int32              — chunk end frame (exclusive)
      chunk_owner  (Nc,) int32              — first span (index into span_meta) claiming it
      span_chunks  {span_idx: [chunk idx]}  — every span → its (possibly shared) chunks
      info         dict                     — data_tag, failures, snap/dedupe counters
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    t0 = time.perf_counter()
    data_tag = cache_key({"resolver": resolver_cfg, "action_key": action_key,
                          "arclen": arclen_distance})
    chunk_cache_dir = str(Path(cache_dir) / "chunks" / data_tag) if cache_dir else ""

    by_ep: dict[str, list] = {}
    for i, m in enumerate(span_meta):
        by_ep.setdefault(m["episode"], []).append((i, int(m["start"]), int(m["end"])))
    jobs = [(ep, spans, horizon) for ep, spans in sorted(by_ep.items())]

    n_cached = sum(1 for ep in by_ep if chunk_cache_dir and (Path(chunk_cache_dir) / f"{ep}.npz").exists())
    n_workers = workers or min(12, os.cpu_count() or 8)
    print(f"{tag} chunk read: {len(jobs)} episodes ({n_cached} cached, tag={data_tag}), "
          f"{n_workers} worker processes")

    # spawn (not fork): the parent may already hold CUDA; workers re-import via PYTHONPATH.
    if repo_dir:
        os.environ["PYTHONPATH"] = repo_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    ctx = mp.get_context("spawn")
    results = []
    with ProcessPoolExecutor(
        max_workers=n_workers, mp_context=ctx, initializer=_worker_init,
        initargs=(repo_dir, json.dumps(resolver_cfg), action_key, image_key,
                  chunk_cache_dir, span_resample, arclen_distance),
    ) as pool:
        results = list(pool.map(_worker_read, jobs))

    key2idx: dict[tuple, int] = {}
    chunk_rows, chunk_ep, chunk_frame, chunk_end, chunk_owner = [], [], [], [], []
    span_chunks: dict[int, list[int]] = {}
    failed, spans_lost = [], 0
    tot_snapped = tot_dropped_old = tot_refs = 0
    for ep, chunks_d, span_map, stats, err in results:
        if err is not None:
            failed.append((ep, err))
            spans_lost += len(by_ep[ep])
            continue
        tot_snapped += stats["snapped"]
        tot_dropped_old += stats["dropped_old"]
        for span_idx, keys in span_map.items():
            idxs = []
            for key in keys:
                k = (ep, *key)
                if k not in key2idx:
                    key2idx[k] = len(chunk_rows)
                    chunk_rows.append(chunks_d[key])
                    chunk_ep.append(ep)
                    chunk_frame.append(int(key[0]))
                    chunk_end.append(int(key[1]) if len(key) > 1 else int(key[0]) + horizon)
                    chunk_owner.append(span_idx)
                idxs.append(key2idx[k])
            span_chunks[span_idx] = idxs
            tot_refs += len(idxs)

    for ep, err in failed:
        print(f"{tag}   episode read FAILED {ep}: {err}")
    if not chunk_rows:
        raise RuntimeError(f"no chunks read ({len(failed)} episode failures)")
    chunks = np.stack(chunk_rows).astype(np.float32)
    mode = "span-resampled (temporal norm)" if span_resample else \
        f"{tot_snapped} snapped, {tot_dropped_old} previously dropped now recovered"
    print(
        f"{tag} chunk read: {len(jobs) - len(failed)}/{len(jobs)} episodes ok in "
        f"{time.perf_counter() - t0:.1f}s — {len(span_chunks)} spans mapped "
        f"({mode}, {spans_lost} lost to episode failures); {len(chunks)} unique chunks "
        f"({tot_refs - len(chunks)} duplicate refs deduped)"
    )
    info = {"data_tag": data_tag, "failed": failed, "spans_lost": spans_lost,
            "snapped": tot_snapped, "dropped_old": tot_dropped_old,
            "deduped": tot_refs - len(chunks), "span_resample": span_resample}
    return chunks, chunk_ep, np.asarray(chunk_frame, dtype=np.int32), \
        np.asarray(chunk_end, dtype=np.int32), \
        np.asarray(chunk_owner, dtype=np.int32), span_chunks, info


# ---------------------------------------------------------------------------
# 3. Pooling — the one pooling path (per-position standardize → token → chunk means)
# ---------------------------------------------------------------------------


def pool_chunks(tok_emb: np.ndarray) -> np.ndarray:
    """(Nc, ntok, D) token embeddings → (Nc, D) chunk embeddings.

    Per-position standardize (kills the token-position signature that otherwise
    dominates the map) then mean over the token dimension."""
    mu = tok_emb.mean(axis=0, keepdims=True)
    sd = tok_emb.std(axis=0, keepdims=True) + 1e-6
    return ((tok_emb - mu) / sd).mean(axis=1).astype(np.float32)


def pool_spans(chunk_emb: np.ndarray, span_chunks: dict[int, list[int]]):
    """(Nc, D) chunk embeddings → (Ns, D) span embeddings (mean over the span's chunks).

    Returns ``(span_emb, span_order)`` with ``span_order`` the span indices (into
    span_meta) row-aligned to ``span_emb``."""
    span_order = sorted(span_chunks)
    span_emb = np.stack(
        [chunk_emb[span_chunks[s]].mean(axis=0) for s in span_order]
    ).astype(np.float32)
    return span_emb, np.asarray(span_order, dtype=np.int32)


# ---------------------------------------------------------------------------
# 4. Preproc stages
# ---------------------------------------------------------------------------


def preprocess(X: np.ndarray, *, center_by_position: bool = False,
               pos_ids: np.ndarray | None = None, l2norm: bool = False,
               whiten: bool = False, pca_dim: int = 0, seed: int = 0) -> np.ndarray:
    """Configurable embedding preprocessing before projection.

    Order: center_by_position (token granularity only; needs ``pos_ids``) → l2norm →
    PCA(pca_dim, whiten). These change what the map shows more than the projection
    method does — keep them explicit per run."""
    X = np.asarray(X, dtype=np.float32)
    if center_by_position and pos_ids is not None:
        X = X.copy()
        for p in np.unique(pos_ids):
            m = pos_ids == p
            X[m] -= X[m].mean(axis=0, keepdims=True)
    if l2norm:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    n_comp = min(pca_dim, X.shape[1], len(X)) if pca_dim and pca_dim > 0 else 0
    if (n_comp and X.shape[1] > n_comp) or whiten:
        from sklearn.decomposition import PCA
        X = PCA(n_components=n_comp or min(X.shape), whiten=whiten,
                random_state=seed).fit_transform(X).astype(np.float32)
    return X


# ---------------------------------------------------------------------------
# 5. Balanced subsample + projection
# ---------------------------------------------------------------------------


def balanced_subsample(groups: np.ndarray, cap: int, seed: int = 0) -> np.ndarray:
    """Cap total points with a fair per-group allocation (waterfilling).

    Uniform sampling lets long spans / chunk-rich spans dominate the map; this grants
    each group an equal share, redistributing what small groups don't use."""
    groups = np.asarray(groups)
    N = len(groups)
    if N <= cap:
        return np.arange(N)
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    buckets = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    order = np.argsort([len(b) for b in buckets])  # smallest first: leftovers flow to big groups
    sel_parts, remaining = [], cap
    for rank, gi in enumerate(order):
        share = remaining // (len(buckets) - rank)
        t = min(len(buckets[gi]), share)
        remaining -= t
        if t == len(buckets[gi]):
            sel_parts.append(buckets[gi])
        elif t > 0:
            sel_parts.append(rng.choice(buckets[gi], t, replace=False))
    return np.sort(np.concatenate(sel_parts))


def project_points(X: np.ndarray, method: str = "tsne", dims: int = 3,
                   seed: int = 0) -> np.ndarray:
    """Project embeddings to ``dims``-D (2 or 3). method: 'tsne' | 'umap' | 'pca'.

    No hidden preprocessing — dimensionality reduction before tsne/umap is the
    ``preprocess`` pca_dim stage. t-SNE uses openTSNE (FFT interpolation, ~10-30x
    faster than sklearn for large N); openTSNE/umap-learn pip-install on demand."""
    from sklearn.decomposition import PCA
    X = np.asarray(X, dtype=np.float32)
    if method == "pca":
        return PCA(n_components=dims, random_state=seed).fit_transform(X)
    if method == "umap":
        try:
            import umap
        except ImportError:
            import subprocess, sys
            print("[project] installing umap-learn…")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "umap-learn"], check=True)
            import umap
        return umap.UMAP(n_components=dims, n_neighbors=15, min_dist=0.1,
                         random_state=seed).fit_transform(X)
    perplexity = min(50, max(5, len(X) // 100))
    try:
        import openTSNE
    except ImportError:
        import subprocess, sys
        print("[project] installing openTSNE…")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openTSNE"], check=True)
        import openTSNE
    tsne = openTSNE.TSNE(n_components=dims, perplexity=perplexity, n_jobs=-1,
                         initialization="pca", random_state=seed)
    return np.asarray(tsne.fit(X), dtype=np.float32)
