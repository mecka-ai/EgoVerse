"""Ingest ELMO GT annotations from R2 into the elmo zarr episodes + rebuild store spans.

The elmo aria/eva zarrs on ``rldb/processed_v3`` ship with EMPTY ``annotations`` arrays;
the ground-truth span annotations live as sidecar JSONs at
``s3://rldb/processed_annotations/<episode_id>_annotations.json`` — a flat list of
``{"text", "start_idx", "end_idx"}`` with ~13 paraphrase rewordings per span. This script:

1. downloads each episode's JSON, dedupes to one annotation per unique
   ``(start_idx, end_idx)`` (first text), and rewrites the zarr ``annotations`` array
   in the standard ZarrWriter encoding (JSON bytes, VariableLengthBytes dtype);
2. rebuilds ``spans.json`` for an already-assembled latent store (the expensive
   state/action embed does NOT need re-running — with pause removal off, span row
   ranges follow directly from the store manifest + orig_frame maps).

Usage
-----
    MODAL_ENVIRONMENT=robotics modal run egomimic/modal/ingest_elmo_annotations.py -- \\
        store_dir=deminf_elmo_aria/deminf_curation_2026-07-07_18-40-10 \\
        <episode_id> [<episode_id> ...]

R2 credentials come from ~/.egoverse_env_old (the legacy rldb bucket keys).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

os.environ.setdefault("MODAL_ENVIRONMENT", "robotics")

_image = modal.Image.debian_slim().pip_install("zarr==3.1.5", "boto3")
_zarr_volume = modal.Volume.from_name("elmo_data_v2")
_outputs_volume = modal.Volume.from_name("egoverse-training-outputs")
_app = modal.App("egomimic-ingest-elmo-annotations", image=_image)

_ZARR_MOUNT = "/mnt/zarr-data"
_OUT_MOUNT = "/mnt/outputs"
_ANN_PREFIX = "processed_annotations"


def _load_r2_creds() -> dict[str, str]:
    env_file = Path("~/.egoverse_env_old").expanduser()
    creds: dict[str, str] = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            creds.setdefault(k.strip(), v.strip().strip("'").strip('"'))
    missing = [k for k in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT_URL")
               if not creds.get(k)]
    if missing:
        raise RuntimeError(f"Missing R2 creds in ~/.egoverse_env_old: {missing}")
    return {k: creds[k] for k in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT_URL")}


@_app.function(volumes={_ZARR_MOUNT: _zarr_volume}, timeout=3600)
def ingest_annotations(episodes: list[str], r2: dict) -> dict:
    """Download annotation JSONs and rewrite each episode's zarr annotations array."""
    import shutil

    import boto3
    import numpy as np
    import zarr
    from zarr.core.dtype import VariableLengthBytes

    s3 = boto3.client(
        "s3", endpoint_url=r2["R2_ENDPOINT_URL"],
        aws_access_key_id=r2["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=r2["R2_SECRET_ACCESS_KEY"], region_name="auto",
    )
    ok, missing, failed = [], [], []
    for ep in episodes:
        ep_dir = Path(_ZARR_MOUNT) / ep
        if not ep_dir.is_dir():
            failed.append((ep, "zarr dir missing on volume"))
            continue
        try:
            obj = s3.get_object(Bucket="rldb", Key=f"{_ANN_PREFIX}/{ep}_annotations.json")
            raw = json.loads(obj["Body"].read())
        except s3.exceptions.NoSuchKey:
            missing.append(ep)
            continue
        except Exception as exc:
            failed.append((ep, f"{type(exc).__name__}: {exc}"))
            continue

        # One annotation per unique (start, end): the JSONs carry ~13 paraphrases per span.
        spans: dict[tuple[int, int], str] = {}
        for a in raw:
            key = (int(a["start_idx"]), int(a["end_idx"]))
            if key[1] > key[0] and str(a.get("text", "")).strip():
                spans.setdefault(key, str(a["text"]).strip())
        if not spans:
            missing.append(ep)
            continue

        ordered = sorted(spans.items())
        encoded = np.array(
            [json.dumps({"text": t, "start_idx": s0, "end_idx": e0},
                        ensure_ascii=False, separators=(",", ":")).encode("utf-8")
             for (s0, e0), t in ordered],
            dtype=object,
        )
        ann_dir = ep_dir / "annotations"
        if ann_dir.exists():
            shutil.rmtree(ann_dir)
        store = zarr.open(str(ep_dir), mode="a", zarr_format=3)
        n = len(encoded)
        store.create_array("annotations", shape=(n,), chunks=(n,), shards=(n,),
                           dtype=VariableLengthBytes())
        store["annotations"][:] = encoded
        feats = dict(store.attrs.get("features", {}))
        if "annotations" in feats:
            feats["annotations"] = {**feats["annotations"], "shape": [n]}
            store.attrs["features"] = feats
        ok.append((ep, n))
        print(f"[{ep}] wrote {n} annotation spans ({len(raw)} raw entries deduped)")

    _zarr_volume.commit()
    print(f"annotations: {len(ok)} episodes written, {len(missing)} without GT JSON, "
          f"{len(failed)} failed")
    for ep, err in failed:
        print(f"  FAILED {ep}: {err}")
    return {"ok": ok, "missing": missing, "failed": failed}


@_app.function(volumes={_ZARR_MOUNT: _zarr_volume, _OUT_MOUNT: _outputs_volume}, timeout=1800)
def rebuild_spans(store_dir: str, task: str = "unknown") -> int:
    """Rebuild spans.json for an assembled latent store from the (now-populated)
    zarr annotations — no re-embed needed."""
    import numpy as np
    import zarr

    lat_dir = Path(_OUT_MOUNT) / store_dir / "latents" / task
    manifest = json.loads((lat_dir / "manifest.json").read_text())
    orig_frame = np.load(str(lat_dir / "orig_frame.npy"))

    spans_out: list[dict] = []
    for ep in manifest["episodes"]:
        h, rs, T = ep["hash"], int(ep["row_start"]), int(ep["n_frames"])
        ep_dir = Path(_ZARR_MOUNT) / h
        if not ep_dir.is_dir():
            print(f"  [WARN] {h}: zarr missing — skipping")
            continue
        store = zarr.open(str(ep_dir), mode="r")
        if "annotations" not in store:
            continue
        of = orig_frame[rs : rs + T]
        for x in store["annotations"][:]:
            try:
                d = json.loads(bytes(x).decode("utf-8"))
            except Exception:
                continue
            s0, e0 = int(d.get("start_idx", -1)), int(d.get("end_idx", -1))
            txt = str(d.get("text", "")).strip()
            if not txt or s0 < 0 or e0 <= s0:
                continue
            lo = int(np.searchsorted(of, s0, side="left"))
            hi = int(np.searchsorted(of, e0, side="left"))
            if hi <= lo:
                continue
            spans_out.append({"episode": h, "start": s0, "end": e0, "text": txt,
                              "row_start": rs + lo, "row_count": hi - lo})

    (lat_dir / "spans.json").write_text(json.dumps(spans_out))
    _outputs_volume.commit()
    print(f"rebuilt spans.json → {lat_dir / 'spans.json'} ({len(spans_out)} spans, "
          f"{len({s['episode'] for s in spans_out})} episodes)")
    return len(spans_out)


@_app.local_entrypoint()
def main(*args: str) -> None:
    store_dir = ""
    episodes: list[str] = []
    for a in args:
        if a.startswith("store_dir="):
            store_dir = a.split("=", 1)[1]
        else:
            episodes.append(a)
    if not episodes:
        raise SystemExit("pass episode ids (and optional store_dir=<run dir>)")

    r2 = _load_r2_creds()
    result = ingest_annotations.remote(episodes, r2)
    print(f"ingested annotations for {len(result['ok'])} episodes")
    if store_dir:
        n = rebuild_spans.remote(store_dir)
        print(f"spans.json rebuilt: {n} spans")
