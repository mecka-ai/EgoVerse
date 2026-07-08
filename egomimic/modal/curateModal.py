"""DemInf curation pipeline — three-phase orchestrator.

Phase 1: build MP4+NPZ shards from zarr episodes (50 parallel workers)
Phase 2: embed state (L40S GPU) + action (CPU) in parallel per task
Phase 3: KSG mutual-information scoring from pre-computed latents

Usage:
  python egomimic/modal/curateModal.py \\
    name=<run_name> description=<desc> \\
    data=mecka_all_zarr model=deminf_default trainer=ddp_modal

Replaces the legacy WDS tar-sharding curation pipeline with a decoupled format that
separates state/action embedding onto different hardware and stores latents in a
provenance-first store (flat arrays + manifest + annotation spans).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import modal
from modal_setup import (
    CFG,
    CURATE_ORCHESTRATOR,
    MODAL_COMPUTE_ARG_MAP,
    ModalCompute,
    _boot_container,
    _local_hf_token,
    _prepare_repo,
    _resolve_git_state,
    app,
    app_name_from_hydra_args,
    deminf_v2_volume,
    DEMINF_V2_MOUNT,
    image,
    launch_detached,
    pop_init_submodules,
    training_outputs_volume,
    zarr_volume,
)
from build_deminf_shards import build_shards_for_task, _build_shards_worker  # noqa: F401
from embed_deminf_shards import _embed_state_shards, _embed_action_shards  # noqa: F401

_SHARED_SECRETS = [modal.Secret.from_name(name) for name in CFG.secret_names]

# Per-task KSG scoring: L40S GPU for GPU-accelerated marginal counting; 8 CPUs for tree build.
_SCORE_COMPUTE = ModalCompute(gpu="L40S", cpu=8, memory_mb=32768)


def _load_cfg(hydra_args: tuple[str, ...]):
    import hydra as _hydra
    with _hydra.initialize_config_dir(
        config_dir=f"{CFG.remote_repo_dir}/egomimic/hydra_configs",
        version_base="1.3",
    ):
        return _hydra.compose("curate", overrides=list(hydra_args))


def _clean_latent_key(key: str) -> str:
    """Recover the bare episode hash from a latent key.

    Older shards keyed latents by the mangled ``np.bytes_(b'<hash>')`` repr of the
    episode_hash. Episode hashes are 24-char Mongo ObjectIds, so extract that; for
    already-clean keys this returns the key unchanged.
    """
    import re as _re
    m = _re.search(r"[0-9a-fA-F]{24}", key)
    return m.group(0) if m else key


def _read_episode_spans(zarr_root: Path, ep_hash: str) -> list[dict]:
    """Read decoded ``{text, start_idx, end_idx}`` annotation spans from an episode's zarr."""
    import json as _json

    import zarr

    def _decode(value):
        import numpy as _np
        if isinstance(value, _np.void):
            value = value.item()
        if isinstance(value, memoryview):
            value = value.tobytes()
        if isinstance(value, bytearray):
            value = bytes(value)
        if isinstance(value, bytes):
            return _json.loads(value.decode("utf-8"))
        if isinstance(value, str):
            return _json.loads(value)
        return value

    for cand in (zarr_root / ep_hash, zarr_root / f"{ep_hash}.zarr"):
        if not cand.exists():
            continue
        store = zarr.open(str(cand), mode="r", zarr_format=3)
        if "annotations" not in store:
            return []
        raw = store["annotations"][:]
        return [d for d in (_decode(x) for x in raw) if isinstance(d, dict)]
    return []


def _read_span_action_trajectories(cfg, span_meta: list[dict], tag: str) -> list:
    """Read the raw per-frame action trajectory for each annotation span from zarr.

    For the TCN span action embedder: each span's ``actions_cartesian[start:end)`` is
    read via the SAME transform path the span autoencoder trained on
    (``SpanActionDataset`` → ``ZarrDataset._collect_curation_batched``), with pause
    removal forced OFF so ``start``/``end`` index raw zarr frames (matching the curation
    store, which hardcodes ``pause_eps=None``, and the TCN training config).

    Returns a list of ``(T_i, action_dim)`` float32 arrays aligned to ``span_meta``.
    """
    import hydra as _hydra
    import numpy as _np
    from omegaconf import OmegaConf as _OC

    ds_cfg = next(iter(cfg.data.train_datasets.values()))
    resolver_cfg = _OC.create(_OC.to_container(ds_cfg.resolver, resolve=True))
    resolver_cfg["pause_removal_epsilon"] = None  # raw frame indices (match store + TCN training)
    resolver = _hydra.utils.instantiate(resolver_cfg)
    resolved = resolver.resolve()
    episodes = dict(resolved.datasets) if hasattr(resolved, "datasets") else dict(resolved)
    print(f"{tag} TCN span read: resolver returned {len(episodes)} episodes")

    from concurrent.futures import ThreadPoolExecutor

    unique_eps = sorted({m["episode"] for m in span_meta})
    missing = [ep for ep in unique_eps if ep not in episodes]
    if missing:
        raise KeyError(
            f"{len(missing)} span episode(s) not resolvable for TCN action read, "
            f"e.g. {missing[:3]}; available e.g. {list(episodes)[:3]}"
        )

    def _read_one(ep: str):
        actions, _, _ = episodes[ep]._collect_curation_batched(
            action_key="actions_cartesian",
            image_key="observations.images.front_img_1",
            image_decode_workers=0,
            load_images=False,
        )
        a = _np.asarray(actions, dtype=_np.float32)
        if a.ndim == 3:           # (T, horizon, D) → executed per-frame action = first step
            a = a[:, 0, :]
        elif a.ndim > 3:
            a = a.reshape(a.shape[0], -1)
        return ep, a

    # Read each unique episode's full action trajectory once, in parallel — zarr reads
    # release the GIL (mirrors the curation episode loader's thread pool).
    cache: dict[str, _np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for ep, a in pool.map(_read_one, unique_eps):
            cache[ep] = a
    print(f"{tag} TCN span read: cached {len(cache)} episodes")

    trajectories: list = []
    for m in span_meta:
        ep, s, e = m["episode"], int(m["start"]), int(m["end"])
        full = cache[ep]
        T = full.shape[0]
        s2, e2 = max(0, min(s, T)), max(0, min(e, T))
        trajectories.append(full[s2:e2] if e2 > s2 else full[:1])
    return trajectories


def _build_quest_token_tsne(
    cfg, span_meta, span_ids, span_cluster, cluster_labels, ae,
    output_dir, task_name, tag, device, seed,
):
    """Token/chunk/span-level QueST projection for the cluster viewer.

    Modular pipeline (egomimic.curation.token_viz): snap-to-full-chunk span tiling
    deduped by (episode, frame) → process-pool cached chunk read → QueST encode
    (+ FSQ codes) → granularity pooling + preproc → balanced subsample → projection.
    Chunk / token / projection tiers are content-key cached under token_viz.cache_dir,
    so re-projections (new method/dims/granularity) skip the zarr read and re-encode.
    Writes tsne3d/spans_tsne3d.json (explicit per-point tok_idx / chunk_frame /
    chunk_idx / sid fields) and metrics.json.
    """
    import numpy as _np
    from omegaconf import OmegaConf as _OCq

    from egomimic.curation import token_viz as tv
    from egomimic.curation.config import select_token_viz_settings
    from egomimic.curation.embedders import QuestTokenEmbedder

    tvs = select_token_viz_settings(cfg)
    H = int(_OCq.select(cfg, "model.action_embedder.quest_horizon", default=100))  # block size fed to QueST
    method = str(_OCq.select(cfg, "model.projection", default="tsne")).lower()
    dims = int(_OCq.select(cfg, "model.projection_dims", default=3))
    cache_dir = tvs.cache_dir if tvs.cache else ""

    # 1+2. Chunk plan + read (snap-to-full-chunk, dedupe, per-episode cache, process pool).
    ds_cfg = next(iter(cfg.data.train_datasets.values()))
    resolver_cfg = _OCq.to_container(ds_cfg.resolver, resolve=True)
    resolver_cfg["pause_removal_epsilon"] = None  # raw frame indices (match the latent store)
    chunks, chunk_ep, chunk_frame, chunk_end, chunk_owner, span_chunks, read_info = tv.read_span_chunks(
        resolver_cfg, CFG.remote_repo_dir, span_meta, H, cache_dir, tag,
        span_resample=tvs.span_resample,
    )
    Nc = len(chunks)

    # 3. QueST encode + FSQ codes (cached on chunk identity + checkpoint + horizon).
    ep_order = sorted(set(chunk_ep))
    ep_idx_of = {e: i for i, e in enumerate(ep_order)}
    chunk_ep_idx = _np.asarray([ep_idx_of[e] for e in chunk_ep], dtype=_np.int32)
    tok_key = tv.cache_key({
        "data": read_info["data_tag"], "eps": ep_order,
        "chunks": tv.array_key(chunk_ep_idx, chunk_frame, chunk_end),
        "ckpt": str(ae.checkpoint_path), "horizon": H,
        "span_resample": tvs.span_resample,
    })
    cached = tv.cache_load_npz(cache_dir, "tokens", tok_key)
    if cached is not None:
        emb = cached["emb"].astype(_np.float32)
        codes = cached["codes"] if "codes" in cached else None
        codebook_size = int(cached["codebook_size"]) if "codebook_size" in cached else None
        print(f"{tag} QueST: token cache HIT ({tok_key}) → {emb.shape}")
    else:
        qt = QuestTokenEmbedder(ae.checkpoint_path, device=device)
        qt.fit()
        emb, codes, codebook_size = qt.embed_chunks_with_codes(chunks)
        if cache_dir:
            extra = {} if codes is None else {"codes": codes, "codebook_size": codebook_size}
            p = tv.cache_save_npz(cache_dir, "tokens", tok_key, emb=emb, **extra)
            print(f"{tag} QueST: token cache write → {p}")
    ntok = emb.shape[1]
    print(f"{tag} QueST: horizon={H}, {Nc} chunks × {ntok} tok from {len(span_chunks)} spans "
          f"(granularity={tvs.granularity})")

    # 4. Granularity pooling (the one pooling path) + point metadata.
    chunk_emb = tv.pool_chunks(emb)                              # (Nc, D)
    span_emb, span_order = tv.pool_spans(chunk_emb, span_chunks)  # (Ns, D)
    if tvs.granularity == "token":
        X = emb.reshape(-1, emb.shape[2]).astype(_np.float32)
        pt_chunk = _np.repeat(_np.arange(Nc, dtype=_np.int32), ntok)
        pt_tok = _np.tile(_np.arange(ntok, dtype=_np.int32), Nc)
        pt_span = chunk_owner[pt_chunk]
    elif tvs.granularity == "chunk":
        X = chunk_emb
        pt_chunk = _np.arange(Nc, dtype=_np.int32)
        pt_tok = _np.full(Nc, -1, dtype=_np.int32)
        pt_span = chunk_owner
    elif tvs.granularity == "span":
        X = span_emb
        pt_chunk = _np.full(len(span_order), -1, dtype=_np.int32)
        pt_tok = _np.full(len(span_order), -1, dtype=_np.int32)
        pt_span = span_order
    else:
        raise ValueError(f"unknown token_viz.granularity: {tvs.granularity!r}")

    X = tv.preprocess(
        X, center_by_position=tvs.center_by_position,
        pos_ids=(pt_tok if tvs.granularity == "token" else None),
        l2norm=tvs.l2norm, whiten=tvs.whiten, pca_dim=tvs.pca_dim, seed=seed,
    )

    # 5. Balanced subsample (per-span / per-cluster fair shares) + cached projection.
    span_cid = _np.asarray([int(span_cluster.get(span_ids[i], -1)) for i in range(len(span_meta))])
    if tvs.balance == "cluster":
        groups = span_cid[pt_span]
    elif tvs.balance == "none":
        groups = _np.zeros(len(X), dtype=_np.int32)
    else:
        groups = pt_span
    sel = tv.balanced_subsample(groups, tvs.cap, seed)
    if len(sel) < len(X):
        print(f"{tag} QueST: subsampled {len(X)} → {len(sel)} points "
              f"(cap={tvs.cap}, balance={tvs.balance})")

    proj_key = tv.cache_key({
        "tokens": tok_key, "granularity": tvs.granularity, "method": method, "dims": dims,
        "seed": seed, "cap": tvs.cap, "balance": tvs.balance,
        "preproc": {"center_by_position": tvs.center_by_position, "l2norm": tvs.l2norm,
                    "whiten": tvs.whiten, "pca_dim": tvs.pca_dim},
    })
    cached = tv.cache_load_npz(cache_dir, "proj", proj_key)
    if cached is not None and len(cached["coords"]) == len(sel):
        proj = cached["coords"].astype(_np.float32)
        print(f"{tag} QueST: projection cache HIT ({proj_key})")
    else:
        print(f"{tag} QueST: projecting {len(sel)} {tvs.granularity} points with {method} ({dims}D)")
        proj = tv.project_points(X[sel], method=method, dims=dims, seed=seed)
        if cache_dir:
            tv.cache_save_npz(cache_dir, "proj", proj_key, coords=proj, sel=sel)

    # 6. Alignment metrics (metrics.json, and embedded in the viewer JSON as chips).
    metrics = None
    if tvs.metrics:
        from egomimic.curation.token_metrics import compute_alignment_metrics
        tok_flat = emb.reshape(-1, emb.shape[2])
        tok_span_all = chunk_owner[_np.repeat(_np.arange(Nc), ntok)]
        metrics = compute_alignment_metrics(
            span_emb=span_emb, span_lang=span_cid[span_order],
            chunk_emb=chunk_emb, chunk_lang=span_cid[chunk_owner],
            token_emb=tok_flat, token_lang=span_cid[tok_span_all], token_span=tok_span_all,
            coords=proj, coords_tok_idx=pt_tok[sel],
            codes=codes, codebook_size=codebook_size, seed=seed,
        )
        metrics.update({
            "granularity": tvs.granularity, "method": method, "dims": dims,
            "n_chunks": int(Nc), "n_tokens_per_chunk": int(ntok),
            "n_spans_mapped": len(span_chunks),
            "chunks_snapped_spans": read_info["snapped"],
            "spans_previously_dropped": read_info["dropped_old"],
            "chunk_refs_deduped": read_info["deduped"],
            "episodes_failed": [e for e, _ in read_info["failed"]],
        })
        mpath = Path(output_dir) / "metrics.json"
        mpath.parent.mkdir(parents=True, exist_ok=True)
        with open(mpath, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"{tag} QueST: metrics → {mpath}: "
              f"lang_nmi={metrics['language_nmi']}, knn={metrics['lang_knn_acc']}, "
              f"locality={metrics['same_span_locality']}, fsq={metrics['fsq']}, "
              f"tokidx_map_nmi={metrics['tokidx_map_nmi']}")

    # 7. Viewer JSON with explicit per-point fields.
    coords = {"x": proj[:, 0].astype(float).tolist(), "y": proj[:, 1].astype(float).tolist(),
              "span_idx": list(range(len(proj)))}
    if dims >= 3:
        coords["z"] = proj[:, 2].astype(float).tolist()
    spans_list, cids = [], []
    for k in sel:
        si = int(pt_span[k])
        m, sid = span_meta[si], span_ids[si]
        ci, tj = int(pt_chunk[k]), int(pt_tok[k])
        cid = int(span_cid[si])
        cids.append(cid)
        # start/end: the chunk window for token/chunk points (frame-precise hover +
        # time coloring), the span itself for span points.
        s0, e0 = ((int(chunk_frame[ci]), int(chunk_end[ci])) if ci >= 0
                  else (int(m["start"]), int(m["end"])))
        rec = {
            "id": f"{sid}#c{ci}t{tj}" if ci >= 0 else sid,
            "sid": sid, "ep": m["episode"], "start": s0, "end": e0, "text": m["text"],
            "score": None, "cluster": cid, "tok_idx": tj, "chunk_idx": ci,
            "chunk_frame": (int(chunk_frame[ci]) if ci >= 0 else -1),
        }
        if tvs.granularity == "span":
            rec["n_chunks"] = len(span_chunks.get(si, []))
        spans_list.append(rec)
    cids = _np.asarray(cids)
    clusters_meta = {str(c): {"label": cluster_labels.get(str(c), ""), "n_spans": int((cids == c).sum())}
                     for c in sorted(set(cids.tolist()))}
    out = {"n_clusters": len(clusters_meta), "clusters": clusters_meta, "spans": spans_list,
           "action": coords, "method": method, "dims": dims,
           "level": tvs.granularity, "ntok": int(ntok)}
    if metrics is not None:
        out["metrics"] = metrics
    tsne_dir = Path(output_dir) / "tsne3d"; tsne_dir.mkdir(parents=True, exist_ok=True)
    with open(tsne_dir / "spans_tsne3d.json", "w") as f:
        json.dump(out, f)
    print(f"{tag} QueST: wrote {tvs.granularity}-level {method} → {tsne_dir/'spans_tsne3d.json'} "
          f"({len(sel)} points plotted)")


# ---------------------------------------------------------------------------
# Latent store: flat memmap arrays + provenance manifest + annotation spans
# ---------------------------------------------------------------------------


def load_latent_store(lat_dir: Path):
    """Load an assembled latent store.

    Returns ``(state_mmap, action_mmap, manifest, spans)`` where the arrays are
    memmapped (N, D) float32, ``manifest`` maps each episode to its row range, and
    ``spans`` carries annotation spans with precomputed ``row_start``/``row_count``.
    """
    import json as _json

    import numpy as _np

    manifest = _json.loads((lat_dir / "manifest.json").read_text())
    spans_path = lat_dir / "spans.json"
    spans = _json.loads(spans_path.read_text()) if spans_path.exists() else []
    state = _np.load(str(lat_dir / "state.npy"), mmap_mode="r")
    action = _np.load(str(lat_dir / "action.npy"), mmap_mode="r")
    return state, action, manifest, spans


def assemble_latent_store(output_dir: str, task_name: str, shard_dir: str) -> bool:
    """Merge the embedder flats into the canonical provenance-first latent store.

    Inputs (written by the embedders): ``latents/<task>/_state.npy`` +
    ``_state_manifest.json`` and ``_action.npy`` + ``_action_manifest.json``.
    Build sidecars on the shard volume (``<hash>.meta.json`` spans, ``<hash>.npz``
    ``orig_frames``) supply annotation spans and original-frame provenance.

    Writes ``state.npy``, ``action.npy``, ``orig_frame.npy``, ``manifest.json``,
    ``spans.json`` under ``latents/<task>/`` in sorted-hash canonical row order, then
    removes the embedder temporaries. Returns True on success.
    """
    import json as _json

    import numpy as _np

    lat_dir = Path(output_dir) / "latents" / task_name
    sm_path = lat_dir / "_state_manifest.json"
    am_path = lat_dir / "_action_manifest.json"
    if not sm_path.exists() or not am_path.exists():
        print(f"[{task_name}][assemble] missing embedder flats — skipping")
        return False

    sm = _json.loads(sm_path.read_text())
    am = _json.loads(am_path.read_text())
    state_flat = _np.load(str(lat_dir / "_state.npy"), mmap_mode="r")
    action_flat = _np.load(str(lat_dir / "_action.npy"), mmap_mode="r")

    s_idx = {e["hash"]: e for e in sm["episodes"]}
    a_idx = {e["hash"]: e for e in am["episodes"]}
    common = sorted(set(s_idx) & set(a_idx))
    if not common:
        print(f"[{task_name}][assemble] no episodes present in both modalities")
        return False

    episodes: list[dict] = []
    n_rows = 0
    for h in common:
        T = int(min(s_idx[h]["n_frames"], a_idx[h]["n_frames"]))
        episodes.append({"hash": h, "row_start": n_rows, "n_frames": T})
        n_rows += T

    Ds, Da = int(sm["dim"]), int(am["dim"])
    state = _np.lib.format.open_memmap(
        str(lat_dir / "state.npy"), mode="w+", dtype=_np.float32, shape=(n_rows, Ds)
    )
    action = _np.lib.format.open_memmap(
        str(lat_dir / "action.npy"), mode="w+", dtype=_np.float32, shape=(n_rows, Da)
    )
    orig_frame = _np.empty(n_rows, dtype=_np.int32)
    shard = Path(shard_dir)
    spans_out: list[dict] = []

    for ep in episodes:
        h, rs, T = ep["hash"], ep["row_start"], ep["n_frames"]
        state[rs : rs + T] = state_flat[s_idx[h]["row_start"] : s_idx[h]["row_start"] + T]
        action[rs : rs + T] = action_flat[a_idx[h]["row_start"] : a_idx[h]["row_start"] + T]

        of = None
        npz_p = shard / f"{h}.npz"
        if npz_p.exists():
            try:
                of = _np.asarray(_np.load(str(npz_p), allow_pickle=True)["orig_frames"], dtype=_np.int64)
            except Exception:
                of = None
        if of is None or len(of) < T:
            of = _np.arange(T, dtype=_np.int64)
        of = of[:T]
        orig_frame[rs : rs + T] = of.astype(_np.int32)

        meta_p = shard / f"{h}.meta.json"
        if not meta_p.exists():
            continue
        try:
            spans = _json.loads(meta_p.read_text()).get("spans", [])
        except Exception:
            continue
        for sp in spans:
            s0, e0 = int(sp.get("start_idx", -1)), int(sp.get("end_idx", -1))
            txt = str(sp.get("text", "")).strip()
            if not txt or s0 < 0 or e0 <= s0:
                continue
            lo = int(_np.searchsorted(of, s0, side="left"))
            hi = int(_np.searchsorted(of, e0, side="left"))
            if hi <= lo:
                continue
            spans_out.append(
                {"episode": h, "start": s0, "end": e0, "text": txt,
                 "row_start": rs + lo, "row_count": hi - lo}
            )

    state.flush()
    action.flush()
    del state, action
    _np.save(str(lat_dir / "orig_frame.npy"), orig_frame)

    manifest = {"task": task_name, "fps": 30, "n_rows": n_rows,
                "state_dim": Ds, "action_dim": Da, "episodes": episodes}
    (lat_dir / "manifest.json").write_text(_json.dumps(manifest))
    (lat_dir / "spans.json").write_text(_json.dumps(spans_out))

    for tmp in ("_state.npy", "_action.npy", "_state_manifest.json", "_action_manifest.json"):
        try:
            (lat_dir / tmp).unlink()
        except Exception:
            pass

    print(
        f"[{task_name}][assemble] store: {n_rows} rows, {len(episodes)} episodes, "
        f"{len(spans_out)} spans"
    )
    return True


def _score_task_clustered(
    task_name: str,
    cfg,
    output_dir: str,
    tag: str,
    t_start: float,
    latents_source: str = "",
) -> tuple[str, dict]:
    """Span-granularity clustered scoring (model.language_conditioning.mode=clustered).

    Reads the latent store (flat arrays + spans.json with row ranges) — from
    ``latents_source`` when given, else ``output_dir`` — clusters spans by language,
    and scores each cluster. Two action representations are supported:

    * default — pool the per-frame action latents from the store over each span;
    * ``action_embedder.type=tcn`` — encode each span's raw action trajectory with a
      trained temporal-CNN span autoencoder (ActionNorms → bottleneck), giving one
      shape-preserving latent per span.

    Language embeddings are reused from a prior run's ``<task>_clustered_spans.npz``
    under ``latents_source`` when present (skips the Qwen3 pass). All fresh outputs
    (scores, span npz, 3-D t-SNE) are written under ``output_dir``.
    """
    import time as _time

    import numpy as _np
    import torch as _torch

    from egomimic.curation.config import (
        select_action_embedder_settings,
        select_language_conditioning_settings,
        select_seed,
    )
    from egomimic.curation.embedders import LanguageEmbedder, TCNActionEmbedder
    from egomimic.curation.scoring import trajectory_scorer_from_cfg

    lang = select_language_conditioning_settings(cfg)
    src_dir = Path(latents_source) if latents_source else Path(output_dir)
    lat_dir = src_dir / "latents" / task_name
    state, action, manifest, spans = load_latent_store(lat_dir)

    span_state, span_action, span_ids, span_texts, span_meta = [], [], [], [], []
    for sp in spans:
        rs, rc = int(sp["row_start"]), int(sp["row_count"])
        if rc < 1:
            continue
        span_state.append(_np.asarray(state[rs : rs + rc]))
        span_action.append(_np.asarray(action[rs : rs + rc]))
        span_ids.append(f"{sp['episode']}::{sp['start']}-{sp['end']}")
        span_texts.append(sp["text"])
        span_meta.append(
            {"episode": sp["episode"], "start": int(sp["start"]),
             "end": int(sp["end"]), "text": sp["text"]}
        )

    # Optional episode-set span filter (model.span_filter.episodes_json): restrict
    # clustering/scoring/viz to spans from the listed episodes.
    from omegaconf import OmegaConf as _OCf
    filt_json = _OCf.select(cfg, "model.span_filter.episodes_json", default=None)
    if filt_json:
        doc = json.loads(Path(str(filt_json)).read_text())
        if isinstance(doc, dict) and "sets" in doc:
            ep_set = {e["episode_id"]: s["name"] for s in doc["sets"] for e in s["episodes"]}
        else:
            ep_set = {str(e): "set" for e in doc}
        keep = [i for i, m in enumerate(span_meta) if m["episode"] in ep_set]
        per_set: dict[str, int] = {}
        for i in keep:
            per_set[ep_set[span_meta[i]["episode"]]] = per_set.get(ep_set[span_meta[i]["episode"]], 0) + 1
        span_state = [span_state[i] for i in keep]
        span_action = [span_action[i] for i in keep]
        span_ids = [span_ids[i] for i in keep]
        span_texts = [span_texts[i] for i in keep]
        span_meta = [span_meta[i] for i in keep]
        print(f"{tag} span filter {filt_json}: kept {len(keep)} spans from "
              f"{len({m['episode'] for m in span_meta})} episodes {per_set}")
        if not keep:
            raise ValueError(f"span filter {filt_json} matched no spans in the store")

    print(
        f"{tag} clustered: {len(span_ids)} annotation spans "
        f"from {len(manifest['episodes'])} episodes (store: {src_dir})"
    )
    if not span_ids:
        print(f"{tag} no annotation spans found — skipping clustered scoring")
        return task_name, {}

    device = "cuda" if _torch.cuda.is_available() else "cpu"

    # Per-span action representation. With a trained temporal-CNN span autoencoder
    # (action_embedder.type=tcn) encode each span's RAW action trajectory
    # (ActionNorms-normalized) into one shape-preserving latent — instead of pooling the
    # per-frame action latents from the store. State stays per-frame (pooled downstream).
    ae = select_action_embedder_settings(cfg)
    if ae.type == "tcn":
        span_vecs = None
        # Reuse precomputed TCN span latents from a prior run when provided — the action
        # shapes don't change between re-cluster runs, so this skips the episode resolve +
        # zarr read + encode entirely.
        if ae.reuse_latents_npz and Path(ae.reuse_latents_npz).exists():
            z = _np.load(ae.reuse_latents_npz, allow_pickle=True)
            if "span_ids" in z and "action_emb" in z:
                id2vec = {str(sid): z["action_emb"][i] for i, sid in enumerate(z["span_ids"])}
                if all(sid in id2vec for sid in span_ids):
                    span_vecs = _np.stack([id2vec[sid] for sid in span_ids]).astype(_np.float32)
                    print(f"{tag} reusing TCN action latents from {ae.reuse_latents_npz} → {span_vecs.shape}")
                else:
                    print(f"{tag} reuse npz action span_ids mismatch — re-encoding from zarr")
        if span_vecs is None:
            if not ae.checkpoint_path:
                raise ValueError("action_embedder.type=tcn requires action_embedder.checkpoint_path")
            print(f"{tag} TCN span action embedder: {ae.checkpoint_path} (norms={ae.norms})")
            trajectories = _read_span_action_trajectories(cfg, span_meta, tag)
            tcn = TCNActionEmbedder(ae.checkpoint_path, norms=ae.norms, device=device)
            tcn.fit()
            span_vecs = tcn.embed_spans(trajectories)  # (n_spans, tcn_latent_dim)
        span_action = [span_vecs[i][None, :] for i in range(len(span_vecs))]
        print(f"{tag} TCN span action latents: {span_vecs.shape}")

    # Naive rule-based clustering (language_conditioning.naive_parse): exact
    # (verb, hand, direction) triples parsed from the text — no Qwen3, no k-means;
    # within a cluster all three words align perfectly by construction.
    from omegaconf import OmegaConf as _OCn
    naive_parse = bool(_OCn.select(cfg, "model.language_conditioning.naive_parse", default=False))

    # Language embeddings for clustering. Reuse a prior run's Qwen3 embeddings when
    # present (k-means re-runs deterministically → identical clusters), else embed now.
    text_embeddings = None
    reuse_npz = src_dir / "scores" / f"{task_name}_clustered_spans.npz"
    if not naive_parse and lang.reuse_clusters and reuse_npz.exists():
        z = _np.load(str(reuse_npz), allow_pickle=True)
        if "span_ids" in z and "lang_emb" in z:
            id2emb = {str(sid): z["lang_emb"][i] for i, sid in enumerate(z["span_ids"])}
            if all(sid in id2emb for sid in span_ids):
                text_embeddings = _np.stack([id2emb[sid] for sid in span_ids]).astype(_np.float32)
                print(f"{tag} reusing language embeddings from {reuse_npz} → {text_embeddings.shape}")
            else:
                print(f"{tag} reuse npz span_ids mismatch — recomputing language embeddings")
    if text_embeddings is None and not naive_parse:
        lemb = LanguageEmbedder(
            source="qwen3",
            latent_dim=4096,  # >= Qwen3 hidden size → no projection, full embedding for clustering
            device=device,
            model_name=lang.model_name,
            max_length=lang.max_length,
            batch_size=lang.batch_size,
            dtype=lang.dtype,
            seed=select_seed(cfg),
            instruction=lang.cluster_instruction,  # steer toward verbs + handedness, not objects
        )
        if lang.cluster_instruction:
            print(f"{tag} clustering instruction: {lang.cluster_instruction}")
        lemb.fit()
        text_embeddings = lemb.embed(span_texts)

    from omegaconf import OmegaConf as _OC2
    disable_scoring = bool(_OC2.select(cfg, "model.cluster_scoring.disable", default=False)) or ae.type == "quest_tokens"
    t_ksg = _time.perf_counter()
    if naive_parse:
        from egomimic.curation.naive_lang import naive_language_clusters
        clustered = naive_language_clusters(span_ids, span_texts, span_meta)
        print(f"{tag} naive (verb|hand|direction) clustering: {len(clustered)} exact-triple "
              f"clusters over {len(span_ids)} spans; largest: "
              + ", ".join(f"{v['label']} ({len(v['spans'])})" for v in list(clustered.values())[:5]))
    else:
        scorer = trajectory_scorer_from_cfg(cfg)
        clustered = scorer.score_clusters(
            span_state, span_action, span_ids, span_texts, span_meta, text_embeddings,
            score=not disable_scoring,
        )
        print(
            f"{tag} clustered {'(scoring DISABLED) ' if disable_scoring else ''}"
            f"done in {_time.perf_counter() - t_ksg:.1f}s — {len(clustered)} clusters"
        )

    scores_dir = Path(output_dir) / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    out_path = scores_dir / f"{task_name}_clustered_scores.json"
    with open(out_path, "w") as f:
        json.dump(clustered, f, indent=2)

    # Per-span artifact for the latent viewer: pooled state/action latents, the
    # Qwen3 language embedding, cluster id, score, and the (episode, start, end)
    # needed to play each span's video clip. One row per span, aligned arrays.
    span_cluster: dict[str, int] = {}
    span_score: dict[str, float] = {}
    cluster_labels: dict[str, str] = {}
    for ck, cv in clustered.items():
        cid = int(ck.split("_")[1])
        cluster_labels[str(cid)] = cv["label"]
        for sid, rec in cv["spans"].items():
            span_cluster[sid] = cid
            span_score[sid] = rec.get("score")

    with open(scores_dir / f"{task_name}_cluster_labels.json", "w") as f:
        json.dump(cluster_labels, f, indent=2)

    # QueST token mode: one point per QueST token (multiple per span, variable by span
    # length), colored by the span's language cluster. Distinct token-level t-SNE artifact.
    if ae.type == "quest_tokens":
        _build_quest_token_tsne(
            cfg, span_meta, span_ids, span_cluster, cluster_labels, ae,
            output_dir, task_name, tag, device, select_seed(cfg),
        )
        training_outputs_volume.commit()
        print(f"{tag} quest_tokens total: {_time.perf_counter() - t_start:.1f}s")
        return task_name, {}

    state_pool = _np.stack([s.mean(axis=0) for s in span_state]).astype(_np.float32)
    action_pool = _np.stack([a.mean(axis=0) for a in span_action]).astype(_np.float32)
    lang_emb = _np.asarray(text_embeddings, dtype=_np.float32)
    cluster_ids = _np.asarray([span_cluster[s] for s in span_ids], dtype=_np.int32)
    scores_arr = _np.asarray(
        [(_np.nan if span_score[s] is None else span_score[s]) for s in span_ids],
        dtype=_np.float32,
    )
    spans_npz = scores_dir / f"{task_name}_clustered_spans.npz"
    _np.savez_compressed(
        spans_npz,
        span_ids=_np.asarray(span_ids, dtype=object),
        episodes=_np.asarray([m["episode"] for m in span_meta], dtype=object),
        starts=_np.asarray([m["start"] for m in span_meta], dtype=_np.int32),
        ends=_np.asarray([m["end"] for m in span_meta], dtype=_np.int32),
        texts=_np.asarray(span_texts, dtype=object),
        cluster_ids=cluster_ids,
        scores=scores_arr,
        state_emb=state_pool,
        action_emb=action_pool,
        lang_emb=lang_emb,
    )
    with open(scores_dir / f"{task_name}_cluster_labels.json", "w") as f:
        json.dump(cluster_labels, f, indent=2)
    print(
        f"{tag} wrote span viewer artifact → {spans_npz} "
        f"(state{state_pool.shape} action{action_pool.shape} lang{lang_emb.shape})"
    )

    # 3D t-SNE per modality → tsne3d/spans_tsne3d.json (consumed by the span viewer:
    # egomimic/modal/latent_viz_app.py::view_spans and scripts/build_span_viz.py).
    seed = select_seed(cfg)

    def _tsne3d(X) -> "_np.ndarray":
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        Xf = _np.asarray(X, dtype=_np.float32)
        if Xf.shape[1] > 50:
            Xf = PCA(n_components=50, random_state=seed).fit_transform(Xf)
        return TSNE(
            n_components=3, init="pca", perplexity=30, random_state=seed
        ).fit_transform(Xf)

    coords = {}
    for name, X in (("state", state_pool), ("action", action_pool), ("language", lang_emb)):
        t_t = _time.perf_counter()
        e = _tsne3d(X)
        coords[name] = {
            "x": e[:, 0].astype(float).tolist(),
            "y": e[:, 1].astype(float).tolist(),
            "z": e[:, 2].astype(float).tolist(),
            "span_idx": list(range(len(e))),
        }
        print(f"{tag} t-SNE {name}: {len(e)} pts in {_time.perf_counter() - t_t:.1f}s")

    spans_list = [
        {
            "id": span_ids[i],
            "ep": span_meta[i]["episode"],
            "start": int(span_meta[i]["start"]),
            "end": int(span_meta[i]["end"]),
            "text": span_texts[i],
            "score": (None if not _np.isfinite(scores_arr[i]) else float(scores_arr[i])),
            "cluster": int(cluster_ids[i]),
        }
        for i in range(len(span_ids))
    ]
    clusters_meta = {
        str(cid): {
            "label": cluster_labels[str(cid)],
            "n_spans": int((cluster_ids == cid).sum()),
        }
        for cid in sorted(set(cluster_ids.tolist()))
    }
    spans_tsne = {
        "n_clusters": len(clusters_meta),
        "clusters": clusters_meta,
        "spans": spans_list,
        **coords,
    }
    tsne_dir = Path(output_dir) / "tsne3d"
    tsne_dir.mkdir(parents=True, exist_ok=True)
    with open(tsne_dir / "spans_tsne3d.json", "w") as f:
        json.dump(spans_tsne, f)
    print(f"{tag} wrote {tsne_dir / 'spans_tsne3d.json'} ({len(spans_list)} spans)")

    training_outputs_volume.commit()

    flat = {
        sid: rec["score"]
        for cl in clustered.values()
        for sid, rec in cl["spans"].items()
        if rec.get("score") is not None
    }
    print(
        f"{tag} clustered total: {_time.perf_counter() - t_start:.1f}s → {out_path} "
        f"({len(flat)} spans scored)"
    )
    return task_name, flat


# ---------------------------------------------------------------------------
# Phase 3: KSG scoring from pre-computed latents
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu=_SCORE_COMPUTE.gpu,
    cpu=_SCORE_COMPUTE.cpu,
    memory=_SCORE_COMPUTE.memory_mb,
    timeout=86400,
    secrets=_SHARED_SECRETS,
    volumes={
        CFG.output_mount_path: training_outputs_volume,
        CFG.volume_mount_path: zarr_volume,
    },
)
def _score_task(
    task_name: str,
    run_name: str,
    hydra_args: tuple[str, ...],
    action_mean: list,
    action_std: list,
    output_dir: str,
    git_remote: str,
    git_commit: str,
    hf_token: str = "",
    latents_source: str = "",
) -> tuple[str, dict[str, float]]:
    """Load pre-computed latents and run scoring. Returns (task_name, scores).

    Reads the latent store from ``latents_source`` (when given) and writes all fresh
    outputs under ``output_dir`` — letting a run reuse an existing store non-destructively.
    """
    import time as _time
    import numpy as _np

    _boot_container(git_remote, git_commit, hf_token)

    from egomimic.curation.config import apply_curation_seed, select_seed
    from egomimic.curation.scoring import trajectory_scorer_from_cfg
    from egomimic.utils.aws.aws_data_utils import load_env

    load_env()
    cfg = _load_cfg(hydra_args)
    apply_curation_seed(select_seed(cfg))

    tag = f"[{task_name}][score]"
    t_start = _time.perf_counter()

    # Run provenance: resolved config + git commit + inputs + seed, next to scores/ —
    # a run must be reproducible from its output dir alone.
    import datetime as _dt
    from omegaconf import OmegaConf as _OCprov
    prov = _OCprov.create({
        "provenance": {
            "run_name": run_name, "task": task_name,
            "git_remote": git_remote, "git_commit": git_commit,
            "hydra_args": list(hydra_args),
            "latents_source": latents_source or None, "output_dir": output_dir,
            "seed": select_seed(cfg),
            "written_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        },
        "config": _OCprov.to_container(cfg, resolve=True),
    })
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "run_config.yaml").write_text(_OCprov.to_yaml(prov))
    print(f"{tag} provenance → {Path(output_dir) / 'run_config.yaml'}")

    # QueST token embedder needs the external/quest submodule (SkillVAE); the score
    # worker boots without submodules, so init it on demand here.
    from egomimic.curation.config import select_action_embedder_settings as _sae
    if _sae(cfg).type == "quest_tokens":
        import subprocess as _sp
        print(f"{tag} initializing external/quest submodule for QueST tokenizer…")
        _sp.run(["git", "-C", CFG.remote_repo_dir, "submodule", "update", "--init",
                 "external/quest"], check=True)
        _qd = f"{CFG.remote_repo_dir}/external/quest"
        if Path(_qd).is_dir() and _qd not in sys.path:
            sys.path.insert(0, _qd)  # make 'quest' importable in THIS process (.pth only helps next start)
            print(f"{tag} added {_qd} to sys.path")

    src_dir = Path(latents_source) if latents_source else Path(output_dir)
    lat_dir = src_dir / "latents" / task_name
    if not (lat_dir / "state.npy").exists() or not (lat_dir / "manifest.json").exists():
        print(f"{tag} latent store not found at {lat_dir} — skipping scoring")
        return task_name, {}

    # Clustered mode: atomic unit is the annotation span, not the episode.
    from egomimic.curation.config import select_language_conditioning_settings

    lang = select_language_conditioning_settings(cfg)
    if lang.enabled and lang.mode == "clustered":
        return _score_task_clustered(task_name, cfg, output_dir, tag, t_start, latents_source)

    state, action, manifest, _spans = load_latent_store(lat_dir)
    episodes = manifest["episodes"]
    if not episodes:
        print(f"{tag} empty latent store — returning empty")
        return task_name, {}

    episode_hashes = [e["hash"] for e in episodes]
    ep_lengths = [int(e["n_frames"]) for e in episodes]
    state_latents_list = [
        _np.asarray(state[e["row_start"] : e["row_start"] + e["n_frames"]]) for e in episodes
    ]
    action_latents_list = [
        _np.asarray(action[e["row_start"] : e["row_start"] + e["n_frames"]]) for e in episodes
    ]
    s_all = _np.asarray(state)
    a_all = _np.asarray(action)
    n_total = s_all.shape[0]

    print(
        f"{tag} KSG: {len(episode_hashes)} episodes, {n_total} timesteps, "
        f"state_dim={s_all.shape[1]}, action_dim={a_all.shape[1]}"
    )

    t_ksg = _time.perf_counter()
    scorer = trajectory_scorer_from_cfg(cfg)
    scores = scorer.score_latents(s_all, a_all, episode_hashes, ep_lengths)
    print(f"{tag} KSG done in {_time.perf_counter() - t_ksg:.1f}s — {len(scores)} episodes scored")

    # t-SNE visualization (non-fatal)
    try:
        from egomimic.curation.tsne_viz import (
            TsneVizSettings,
            export_task_tsne3d,
            make_task_tsne_plots,
        )
        from egomimic.curation.config import select_tsne_viz_config

        tsne_cfg = select_tsne_viz_config(cfg)
        viz_settings = TsneVizSettings(
            every_n=tsne_cfg.every_n,
            seed=select_seed(cfg),
            include_state_lang=tsne_cfg.include_state_lang,
            include_language=tsne_cfg.include_language,
            include_state_by_lang=tsne_cfg.include_state_by_lang,
            state_color_by=tsne_cfg.state_color_by,
        )
        tsne_dir = Path(output_dir) / "tsne"
        tsne3d_dir = Path(output_dir) / "tsne3d"
        make_task_tsne_plots(
            task_name,
            state_latents_list,
            action_latents_list,
            tsne_dir,
            settings=viz_settings,
        )
        export_task_tsne3d(
            task_name,
            state_latents_list,
            action_latents_list,
            episode_hashes,
            tsne3d_dir,
            settings=viz_settings,
        )
    except Exception as exc:
        import traceback
        print(f"{tag} t-SNE viz FAILED (non-fatal): {exc}")
        traceback.print_exc()

    # Write per-task scores
    scores_dir = Path(output_dir) / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    with open(scores_dir / f"{task_name}_scores.json", "w") as f:
        json.dump(dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True)), f, indent=2)
    training_outputs_volume.commit()

    print(f"{tag} total: {_time.perf_counter() - t_start:.1f}s")
    return task_name, scores


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu=CURATE_ORCHESTRATOR.gpu,
    cpu=CURATE_ORCHESTRATOR.cpu,
    memory=CURATE_ORCHESTRATOR.memory_mb,
    timeout=CFG.timeout_seconds,
    secrets=_SHARED_SECRETS,
    volumes={
        CFG.volume_mount_path: zarr_volume,
        DEMINF_V2_MOUNT: deminf_v2_volume,
        CFG.output_mount_path: training_outputs_volume,
    },
)
def run_curate(
    hydra_args: tuple[str, ...],
    git_remote: str,
    git_commit: str,
    run_name: str,
    hf_token: str = "",
) -> str:
    """Three-phase DemInf orchestrator: build shards → embed → KSG."""
    import sys as _sys
    import time as _time
    import numpy as _np

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    _prepare_repo(git_remote=git_remote, git_commit=git_commit)
    _sys.path.insert(0, CFG.remote_repo_dir)
    os.chdir(CFG.remote_repo_dir)
    os.environ["MODAL_IS_REMOTE"] = "1"
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")

    import hydra as _hydra
    from omegaconf import OmegaConf
    from egomimic.curation.config import (
        apply_curation_seed,
        load_action_norm_stats,
        select_seed,
        select_tensor_keys,
    )
    from egomimic.utils.aws.aws_data_utils import load_env

    load_env()

    with _hydra.initialize_config_dir(
        config_dir=f"{CFG.remote_repo_dir}/egomimic/hydra_configs",
        version_base="1.3",
    ):
        cfg = _hydra.compose("curate", overrides=list(hydra_args))

    apply_curation_seed(select_seed(cfg))

    # ── 1. SQL task lookup ────────────────────────────────────────────────────
    print("Running SQL task lookup …")
    from egomimic.utils.aws.aws_sql import episode_table_to_df, create_default_engine

    engine = create_default_engine()
    full_df = episode_table_to_df(engine)
    if "is_deleted" in full_df.columns:
        full_df = full_df[full_df["is_deleted"] != True]  # noqa: E712

    hash_to_task: dict[str, str] = {}
    if "task" in full_df.columns:
        hash_to_task = dict(
            zip(full_df["episode_hash"], full_df["task"].fillna("unknown"))
        )

    # ── 2. Resolve episodes via data-config resolver ──────────────────────────
    by_task: dict[str, list[str]] = {}  # task → list of zarr episode dirs

    zarr_root = Path(CFG.volume_mount_path)
    for ds_name, ds_cfg in cfg.data.train_datasets.items():
        resolver = _hydra.utils.instantiate(ds_cfg.resolver)
        dataset_filter = (
            _hydra.utils.instantiate(ds_cfg.filters) if "filters" in ds_cfg else None
        )
        resolved = resolver.resolve(filters=dataset_filter)
        print(f"[{ds_name}] {len(resolved)} episodes after resolver")

        for episode_hash in resolved:
            task = hash_to_task.get(episode_hash) or "unknown"
            if str(task) in ("nan", "None", ""):
                task = "unknown"
            # Resolve zarr dir path on volume
            ep_dir: str | None = None
            for cand in (zarr_root / episode_hash, zarr_root / f"{episode_hash}.zarr"):
                if cand.is_dir():
                    ep_dir = str(cand)
                    break
            if ep_dir is None:
                continue
            by_task.setdefault(task, []).append(ep_dir)

    total_episodes = sum(len(v) for v in by_task.values())
    print(
        f"Episode partition: {total_episodes} episodes across {len(by_task)} tasks — "
        + ", ".join(f"{t}:{len(h)}" for t, h in sorted(by_task.items())[:5])
        + ("…" if len(by_task) > 5 else "")
    )
    if total_episodes == 0:
        print("No episodes found — check data config resolver")
        return ""

    # ── 3. Load norm stats ────────────────────────────────────────────────────
    from egomimic.curation.config import select_action_embedder_settings as _sae_cfg
    _ae_type = _sae_cfg(cfg).type
    action_key, _ = select_tensor_keys(cfg)
    if _ae_type == "quest_tokens":
        # QueST encoder normalises internally — no external norm stats needed.
        print(f"action_embedder.type=quest_tokens — skipping norm stats load")
        action_mean: list = []
        action_std: list = []
    else:
        try:
            action_mean_arr, action_std_arr = load_action_norm_stats(
                cfg,
                action_key,
                search_roots=[CFG.output_mount_path, CFG.remote_repo_dir, Path.cwd()],
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load norm stats: {exc}") from exc
        action_mean = action_mean_arr.tolist()
        action_std = action_std_arr.tolist()

    # ── 4. Output directory ───────────────────────────────────────────────────
    timestamp = _time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(CFG.output_mount_path) / cfg.name / f"{cfg.description}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_str = str(output_dir)

    print(f"Output dir: {output_dir_str}")

    # ── 5. Per-task fan-out ───────────────────────────────────────────────────
    print(f"Processing {len(by_task)} task(s) …")
    t0 = _time.time()

    scores_by_task: dict[str, dict[str, float]] = {}
    n_failures = 0

    for task_name, ep_dirs in sorted(by_task.items()):
        print(f"\n── [{task_name}] {len(ep_dirs)} episodes ──")
        t_task = _time.perf_counter()

        # Phase 1: build MP4+NPZ shards
        print(f"[{task_name}] Phase 1: building shards …")
        shard_pairs = build_shards_for_task(
            ep_dirs,
            task_name,
            run_name,
            hydra_args,
            git_remote,
            git_commit,
            hf_token,
        )
        if not shard_pairs:
            print(f"[{task_name}] No shards built — skipping")
            n_failures += 1
            continue
        print(f"[{task_name}] Phase 1 done: {len(shard_pairs)} shards")

        # Phase 2: parallel state + action embedding
        print(f"[{task_name}] Phase 2: spawning state + action embedders …")
        state_handle = _embed_state_shards.spawn(
            shard_pairs, task_name, run_name, hydra_args,
            action_mean, action_std, git_remote, git_commit, output_dir_str, hf_token,
        )
        action_handle = _embed_action_shards.spawn(
            shard_pairs, task_name, run_name, hydra_args,
            action_mean, action_std, git_remote, git_commit, output_dir_str, hf_token,
        )
        try:
            state_path = state_handle.get(timeout=14400)
            action_path = action_handle.get(timeout=14400)
        except Exception as exc:
            print(f"[{task_name}] embedding FAILED: {exc}")
            n_failures += 1
            continue
        print(f"[{task_name}] Phase 2 done: state={state_path}, action={action_path}")

        # Phase 2.5: assemble the provenance-first latent store (flat arrays + manifest + spans).
        training_outputs_volume.reload()
        deminf_v2_volume.reload()
        shard_dir = str(Path(DEMINF_V2_MOUNT) / run_name / "shards" / task_name)
        if not assemble_latent_store(output_dir_str, task_name, shard_dir):
            print(f"[{task_name}] latent-store assembly FAILED — skipping")
            n_failures += 1
            continue
        training_outputs_volume.commit()

        # Phase 3: KSG scoring (skipped for quest_tokens — embedding-only run)
        if _ae_type == "quest_tokens":
            print(f"[{task_name}] Phase 3: skipped (quest_tokens embedding-only)")
            scores_by_task[task_name] = {}
            print(f"[{task_name}] complete in {_time.perf_counter() - t_task:.1f}s")
            continue

        print(f"[{task_name}] Phase 3: KSG scoring …")
        try:
            _, task_scores = _score_task.remote(
                task_name, run_name, hydra_args,
                action_mean, action_std, output_dir_str,
                git_remote, git_commit, hf_token,
            )
            scores_by_task[task_name] = task_scores
        except Exception as exc:
            print(f"[{task_name}] KSG FAILED: {exc}")
            n_failures += 1
            continue

        print(
            f"[{task_name}] complete — {len(task_scores)} scores "
            f"in {_time.perf_counter() - t_task:.1f}s"
        )

    elapsed = _time.time() - t0

    # ── 6. Aggregate + save ───────────────────────────────────────────────────
    flat_scores: dict[str, float] = {}
    for t_scores in scores_by_task.values():
        flat_scores.update(t_scores)

    all_vals = _np.array([s for s in flat_scores.values() if _np.isfinite(s)])

    per_task_stats: dict[str, dict] = {}
    for t_name, t_scores in scores_by_task.items():
        vals = _np.array([s for s in t_scores.values() if _np.isfinite(s)])
        per_task_stats[t_name] = {
            "count":     len(t_scores),
            "mi_mean":   float(_np.nanmean(vals))   if len(vals) else float("nan"),
            "mi_std":    float(_np.nanstd(vals))    if len(vals) else float("nan"),
            "mi_median": float(_np.nanmedian(vals)) if len(vals) else float("nan"),
            "mi_min":    float(_np.nanmin(vals))    if len(vals) else float("nan"),
            "mi_max":    float(_np.nanmax(vals))    if len(vals) else float("nan"),
        }

    def _sort_scores(d: dict) -> dict:
        return dict(sorted(d.items(), key=lambda kv: kv[1] if _np.isfinite(kv[1]) else float("-inf"), reverse=True))

    sorted_flat = _sort_scores(flat_scores)
    sorted_by_task = {t: _sort_scores(s) for t, s in scores_by_task.items()}

    with open(output_dir / "scores.json", "w") as f:
        json.dump(sorted_flat, f, indent=2)
    with open(output_dir / "scores_by_task.json", "w") as f:
        json.dump(sorted_by_task, f, indent=2)
    with open(output_dir / "kept_hashes.json", "w") as f:
        json.dump(list(sorted_flat.keys()), f, indent=2)
    with open(output_dir / "curation_stats.json", "w") as f:
        json.dump({
            "total_input":     total_episodes,
            "n_tasks":         len(scores_by_task),
            "n_task_failures": n_failures,
            "scored":          len(flat_scores),
            "elapsed_seconds": round(elapsed, 1),
            "mi_mean":   float(all_vals.mean())       if len(all_vals) else float("nan"),
            "mi_std":    float(all_vals.std())        if len(all_vals) else float("nan"),
            "mi_median": float(_np.median(all_vals))  if len(all_vals) else float("nan"),
            "per_task":  per_task_stats,
        }, f, indent=2)

    # ── 7. t-SNE export for latent viewer (from the latent store) ────────────
    print("\nPhase 4: exporting t-SNE for latent viewer …")
    from egomimic.curation.tsne_viz import export_task_tsne3d

    tsne_dir = output_dir / "tsne3d"
    tsne_dir.mkdir(parents=True, exist_ok=True)

    all_hashes: list[str] = []
    for t_name in sorted(scores_by_task.keys()):
        lat_dir = output_dir / "latents" / t_name
        if not (lat_dir / "state.npy").exists() or not (lat_dir / "manifest.json").exists():
            print(f"[{t_name}] latent store missing — skipping t-SNE")
            continue
        try:
            state, action, manifest, _spans = load_latent_store(lat_dir)
            eps = manifest["episodes"]
            common = [e["hash"] for e in eps]
            all_hashes.extend(common)
            json_path = export_task_tsne3d(
                t_name,
                [_np.asarray(state[e["row_start"] : e["row_start"] + e["n_frames"]]) for e in eps],
                [_np.asarray(action[e["row_start"] : e["row_start"] + e["n_frames"]]) for e in eps],
                common,
                tsne_dir,
            )
            print(f"[{t_name}] tsne3d → {json_path} ({len(common)} episodes)")
        except Exception as exc:
            print(f"[{t_name}] t-SNE FAILED: {exc}")

    # ── 8. Episode-preview MP4s so the viewer's /video + /frame resolve ───────
    if all_hashes:
        print(f"\nPhase 5: rendering {len(all_hashes)} episode previews …")
        try:
            render_episode = modal.Function.from_name(
                "egoverse-episode-preview-render", "render_episode"
            )
            n_prev = sum(1 for _ in render_episode.map(sorted(set(all_hashes))))
            print(f"Episode previews done — {n_prev} episodes")
        except Exception as exc:
            print(f"Episode preview render FAILED (non-fatal): {exc}")

    zarr_volume.commit()
    training_outputs_volume.commit()

    print(
        f"\nDemInf curation done — scored={len(flat_scores)} episodes "
        f"across {len(scores_by_task)}/{len(by_task)} tasks "
        f"in {elapsed:.1f}s\nOutput: {output_dir_str}"
    )
    return output_dir_str


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def submit_curate(*hydra_args: str) -> None:
    """Fire-and-forget: spawn a DemInf curation job."""
    hydra_args, _ = pop_init_submodules(hydra_args)
    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print("Warning: local repo has uncommitted changes.")

    # Derive run_name from hydra args (name= key)
    run_name = "deminf_v2"
    for arg in hydra_args:
        key, sep, val = arg.lstrip("+").partition("=")
        if sep and key == "name":
            run_name = val.strip()
            break

    print(f"Submitting DemInf curation: run_name={run_name} at commit {git_commit[:12]}")
    handle = run_curate.spawn(
        tuple(hydra_args), git_remote, git_commit, run_name,
        hf_token=_local_hf_token(),
    )
    _env = os.environ.get("MODAL_ENVIRONMENT", "robotics")
    _app = os.environ.get("MODAL_APP_NAME", "egomimic-training")
    print(f"Submitted: {handle.object_id}")
    print(f"Monitor: https://modal.com/apps/mecka/{_env}/apps/{_app}")


# ---------------------------------------------------------------------------
# Resume: score-only (Phase 3) against pre-computed latents
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    timeout=CFG.timeout_seconds,
    secrets=_SHARED_SECRETS,
    volumes={CFG.output_mount_path: training_outputs_volume},
)
def run_score(
    output_dir: str,
    hydra_args: tuple[str, ...],
    git_remote: str,
    git_commit: str,
    run_name: str,
    score_task: str = "",
    hf_token: str = "",
    latents_source: str = "",
) -> str:
    """Resume: run ONLY Phase 3 (scoring) against pre-computed latents.

    Discovers tasks under ``<latents_source or output_dir>/latents/`` (or just
    ``score_task``) and invokes ``_score_task`` per task — no shard build, no embedding.
    When ``latents_source`` differs from ``output_dir`` the store is read from the source
    and all fresh outputs are written under ``output_dir`` (non-destructive reuse).
    """
    import sys as _sys
    from pathlib import Path as _Path

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    _prepare_repo(git_remote=git_remote, git_commit=git_commit)
    _sys.path.insert(0, CFG.remote_repo_dir)
    os.chdir(CFG.remote_repo_dir)
    os.environ["MODAL_IS_REMOTE"] = "1"

    src_dir = latents_source or output_dir
    lat_root = _Path(src_dir) / "latents"
    if not lat_root.is_dir():
        print(f"No latents dir under {src_dir} — nothing to score")
        return ""

    if score_task:
        tasks = [score_task]
    else:
        tasks = sorted(p.name for p in lat_root.iterdir() if p.is_dir())
    print(
        f"Resume scoring {len(tasks)} task(s): store={src_dir} → output={output_dir}: {tasks}"
    )

    for task_name in tasks:
        print(f"\n── [{task_name}] scoring …")
        try:
            _score_task.remote(
                task_name, run_name, hydra_args,
                [], [], output_dir, git_remote, git_commit, hf_token, latents_source,
            )
        except Exception as exc:
            print(f"[{task_name}] scoring FAILED: {exc}")

    training_outputs_volume.commit()
    print(f"\nResume scoring done — {output_dir}")
    return output_dir


@app.local_entrypoint()
def submit_score(*args: str) -> None:
    """Fire-and-forget: resume KSG scoring on an existing run's latents.

    Required: ``score_output_dir=<output dir under /root/EgoVerse/logs>``
    Optional: ``score_task=<task>`` (default: all tasks under latents/)
    Optional: ``score_latents_source=<dir>`` — read the latent store + reuse artifacts
              (e.g. a prior run's language clusters) from this dir while writing fresh
              outputs under ``score_output_dir`` (non-destructive reuse).
    Remaining args are normal hydra overrides (model=…, data=…, language_conditioning…).
    """
    args, _ = pop_init_submodules(args)
    output_dir = ""
    score_task = ""
    latents_source = ""
    hydra_args: list[str] = []
    for a in args:
        key, sep, val = a.lstrip("+").partition("=")
        if sep and key == "score_output_dir":
            output_dir = val.strip()
        elif sep and key == "score_task":
            score_task = val.strip()
        elif sep and key == "score_latents_source":
            latents_source = val.strip()
        else:
            hydra_args.append(a)
    if not output_dir:
        raise SystemExit("score_output_dir=<path> is required for score-only resume")

    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print("Warning: local repo has uncommitted changes.")

    run_name = "deminf_v2"
    for a in hydra_args:
        key, sep, val = a.lstrip("+").partition("=")
        if sep and key == "name":
            run_name = val.strip()
            break

    print(
        f"Resume scoring: output_dir={output_dir}"
        + (f" (store from {latents_source})" if latents_source else "")
        + f" at commit {git_commit[:12]}"
    )
    handle = run_score.spawn(
        output_dir, tuple(hydra_args), git_remote, git_commit, run_name, score_task,
        hf_token=_local_hf_token(), latents_source=latents_source,
    )
    _env = os.environ.get("MODAL_ENVIRONMENT", "robotics")
    _app = os.environ.get("MODAL_APP_NAME", "egomimic-training")
    print(f"Submitted: {handle.object_id}")
    print(f"Monitor: https://modal.com/apps/mecka/{_env}/apps/{_app}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    modal_env = os.environ.copy()
    hydra_args: list[str] = []
    _MODAL_FLAGS = {"--detach", "--env"}
    for arg in sys.argv[1:]:
        if arg in _MODAL_FLAGS:
            continue
        key, sep, val = arg.lstrip("+").partition("=")
        if sep and key in MODAL_COMPUTE_ARG_MAP:
            modal_env[MODAL_COMPUTE_ARG_MAP[key]] = val
        else:
            hydra_args.append(arg)

    # Route to score-only resume when score_output_dir= is supplied.
    entrypoint = (
        "submit_score"
        if any(a.startswith("score_output_dir=") for a in hydra_args)
        else "submit_curate"
    )

    modal_env["MODAL_APP_NAME"] = app_name_from_hydra_args(hydra_args)
    print(f"DemInf — app: {modal_env['MODAL_APP_NAME']} (entrypoint: {entrypoint})")
    launch_detached(Path(__file__).resolve(), entrypoint, hydra_args, modal_env)
