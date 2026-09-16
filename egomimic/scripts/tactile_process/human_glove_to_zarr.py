"""
Pull a handful of human tactile-glove episodes and write them to Zarr.

Data source: MongoDB `mecka-ai.episodes` docs referenced by
`tactile_data/tactile_200h_episodes.csv` (a 200-hour, tactile-annotated
subset of the MECKA human episode corpus). Each episode doc carries
`tactileLeftKey` / `tactileRightKey` -- R2 storage keys pointing at
per-hand glove CSVs (`glove_left.csv`, `glove_right.csv`) that are NOT yet
wired into the existing MECKA->Zarr pipeline
(`egomimic/scripts/mecka_process/mecka_to_zarr.py`, `modal_mecka_to_zarr.py`):
those only extract video/hand-pose, not the glove tactile stream.

Each glove CSV is a plain per-frame table:
    frame_index, log_time_ns, t0, t1, ..., t459
i.e. 460 raw taxel readings per hand per frame, sampled independently per
hand (left/right are NOT guaranteed to have matching frame counts or
timestamps). This script stores that raw (T, 460) layout as-is -- it does
NOT reshape to a [C, H, W] spatial grid, because the physical taxel->pixel
layout of this glove is unknown; fabricating one would silently corrupt any
later spatial patchifying (see `egomimic/models/tactile_nets.py:patchify`).
Resampling/aligning left+right and mapping taxels to a real 2D layout is a
follow-up step once that layout is known.

This is a small-scale pilot (a handful of episodes), not a bulk downloader:
- each glove CSV is ~10 MB for a ~45s episode (video/hand assets are NOT
  downloaded at all here, keeping this fast and disk-light).
- point `--n-episodes` at a bigger number for a bigger pull later, once you
  know how much disk/time you actually want to spend (7733 episodes'
  worth of glove data alone would be on the order of tens of GB).

Usage:
    python human_glove_to_zarr.py \\
        --csv /home/mecka/EgoVerse/tactile_data/tactile_200h_episodes.csv \\
        --env ~/mecka/.env \\
        --out-dir /home/mecka/EgoVerse/zarr/tactile_human \\
        --n-episodes 5
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
from pathlib import Path

import numpy as np
import zarr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MONGODB_DB = "mecka-ai"
MONGODB_EPISODES_COLLECTION = "episodes"
DEFAULT_BUCKET = "data"


def load_env(path: str) -> None:
    """Minimal .env loader: KEY=VALUE lines, no external dependency."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f".env file not found: {p}")
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'").strip('"'))


def parse_storage_key(storage_key: str, default_bucket: str) -> tuple[str, str]:
    if storage_key.startswith("r2://"):
        without_scheme = storage_key[len("r2://"):]
        bucket, _, key = without_scheme.partition("/")
        return bucket, key
    return default_bucket, storage_key


def get_r2_client():
    import boto3

    endpoint = os.environ.get("R2_ENDPOINT") or os.environ.get("R2_ENDPOINT_URL")
    access_key = os.environ.get("R2_ACCESS_KEY") or os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_KEY") or os.environ.get("R2_SECRET_ACCESS_KEY")
    if not (endpoint and access_key and secret_key):
        raise RuntimeError(
            "Missing R2 credentials. Expected R2_ENDPOINT / R2_ACCESS_KEY / "
            "R2_SECRET_KEY (or the *_URL / *_ID / *_ACCESS_KEY variants) in the env file."
        )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=boto3.session.Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def fetch_glove_csv(r2_client, storage_key: str, bucket: str) -> np.ndarray:
    """Download one glove_{left,right}.csv and return it as (T, 462) float32
    [frame_index, log_time_ns, t0..t459]."""
    b, key = parse_storage_key(storage_key, bucket)
    obj = r2_client.get_object(Bucket=b, Key=key)
    body = obj["Body"].read()
    arr = np.loadtxt(io.StringIO(body.decode("utf-8")), delimiter=",", skiprows=1)
    return arr.astype(np.float32)


def convert_episode(r2_client, mongo_doc: dict, out_dir: Path, bucket: str) -> Path:
    episode_id = str(mongo_doc["_id"])
    left_key = mongo_doc.get("tactileLeftKey")
    right_key = mongo_doc.get("tactileRightKey")
    if not left_key or not right_key:
        raise ValueError(f"Episode {episode_id} missing tactileLeftKey/tactileRightKey")

    left = fetch_glove_csv(r2_client, left_key, bucket)
    right = fetch_glove_csv(r2_client, right_key, bucket)

    dest = out_dir / f"{episode_id}.zarr"
    if dest.exists():
        import shutil

        shutil.rmtree(dest)
    root = zarr.open_group(str(dest), mode="w")

    root.create_array("tactile_left", data=np.ascontiguousarray(left[:, 2:], dtype=np.float32))
    root.create_array("tactile_left_frame_index", data=left[:, 0].astype(np.int64))
    root.create_array("tactile_left_time_ns", data=left[:, 1].astype(np.int64))

    root.create_array("tactile_right", data=np.ascontiguousarray(right[:, 2:], dtype=np.float32))
    root.create_array("tactile_right_frame_index", data=right[:, 0].astype(np.int64))
    root.create_array("tactile_right_time_ns", data=right[:, 1].astype(np.int64))

    root.attrs["episode_id"] = episode_id
    root.attrs["task"] = mongo_doc.get("task_desc", "")
    root.attrs["scene"] = str(mongo_doc.get("scene_id", ""))
    root.attrs["environment"] = str(mongo_doc.get("environment_id", ""))
    root.attrs["duration"] = mongo_doc.get("duration")
    root.attrs["num_taxels_per_hand"] = 460
    root.attrs["source"] = "mecka-ai.episodes (human tactile glove, pilot pull)"
    root.attrs["note"] = (
        "tactile_{left,right} are raw (T, 460) taxel readings, NOT reshaped "
        "to a spatial grid -- physical taxel layout unknown. left/right are "
        "sampled independently and are not frame-aligned."
    )
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description="Human glove tactile -> Zarr (pilot)")
    ap.add_argument("--csv", required=True, help="tactile_200h_episodes.csv path")
    ap.add_argument("--env", default="~/mecka/.env", help="Path to credentials .env")
    ap.add_argument("--out-dir", required=True, help="Output directory for per-episode .zarr")
    ap.add_argument("--n-episodes", type=int, default=5, help="How many episodes to pull")
    ap.add_argument("--bucket", default=DEFAULT_BUCKET, help="R2 bucket fallback for bare keys")
    args = ap.parse_args()

    load_env(args.env)
    from pymongo import MongoClient
    from bson import ObjectId

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    logger.info(f"{len(rows)} episodes listed in {args.csv}; pulling first {args.n_episodes}")

    client = MongoClient(os.environ["MONGODB_URI"], serverSelectionTimeoutMS=15000)
    db = client[MONGODB_DB]
    r2 = get_r2_client()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok, failed = [], []
    for row in rows[: args.n_episodes]:
        eid = row["episode_id"]
        try:
            doc = db[MONGODB_EPISODES_COLLECTION].find_one({"_id": ObjectId(eid)})
            if doc is None:
                raise ValueError("not found in MongoDB")
            dest = convert_episode(r2, doc, out_dir, args.bucket)
            logger.info(f"[{eid}] wrote {dest}")
            ok.append(eid)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{eid}] FAILED: {e}")
            failed.append((eid, str(e)))

    logger.info(f"Done. ok={len(ok)} failed={len(failed)}")
    for eid, err in failed:
        logger.info(f"  FAILED {eid}: {err}")


if __name__ == "__main__":
    main()
