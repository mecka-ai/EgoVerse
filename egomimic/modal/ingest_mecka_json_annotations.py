"""Convert raw Mecka episodes from R2 → zarr on the data volume, using language
annotations from a local JSON file instead of the episode's own annotations.

Unlike ``ingest_zarr.py`` (which downloads *already-processed* zarr stores listed
in the legacy SQL table), these episodes are unprocessed: they exist only as raw
files in the Mecka ``data`` R2 bucket, referenced from the ``mecka-ai.episodes``
MongoDB collection. So this script runs the real raw→zarr conversion
(``mecka_to_zarr.MeckaDatasetConverter``) per episode, with ONE change:

    The annotations baked into the zarr come from the supplied JSON file's
    ``preannotations`` (label + start_seconds/end_seconds), NOT from the
    episode's own ``annotations.csv``.

Per episode (one Modal container each):
  1. Look up the Mongo doc → ``s3_base_path`` + ``framesKey``.
  2. Download hands.csv, egomotion.txt, frames.csv, and the aligned ultrawide
     video from the ``data`` R2 bucket into a temp dir.
  3. Write ``annotations.csv`` from the JSON ``preannotations`` (columns
     label/start_time/end_time) — this is the swap. MeckaDatasetConverter then
     converts label,start_time,end_time → (label, int(start*30), int(end*30))
     exactly as it does for real annotations.
  4. Run the converter and move ``<episode_id>.zarr`` onto the volume.

The video must be the one frame-aligned with frames.csv/hands.csv — that is the
``ultrawide_video_<eptag>.mp4`` under ``s3_base_path`` (matching framesKey's ep
tag), NOT the trimmed ``final_video.mp4``.

Usage:
    modal run --detach --env robotics \\
        egomimic/modal/ingest_mecka_json_annotations.py -- \\
        --json-path ~/Downloads/preannotations-trusted-labelers.json --limit 200
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
    _boot_container as _boot_container_fn,
    _local_hf_token,
    _resolve_git_state,
    app,
    image,
    zarr_volume,
)

# mecka_to_zarr needs cv2/pandas/scipy/zarr (all in the base training image);
# we additionally need pymongo to resolve episode_id → R2 paths.
_image = image.pip_install("pymongo")

_SECRETS = [
    modal.Secret.from_name("egoverse-r2"),
    modal.Secret.from_name("egoverse-mongodb"),
]

_MOUNT = CFG.volume_mount_path  # /mnt/zarr-data
_MAX_CONTAINERS = 100


def _boot_container(git_remote: str, git_commit: str, hf_token: str) -> None:
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    _boot_container_fn(git_remote, git_commit, hf_token)


# ---------------------------------------------------------------------------
# Per-episode worker
# ---------------------------------------------------------------------------


@app.function(
    image=_image,
    cpu=4,
    memory=16384,
    timeout=3600,
    max_containers=_MAX_CONTAINERS,
    secrets=_SECRETS,
    volumes={_MOUNT: zarr_volume},
)
def _convert_one(
    episode_id: str,
    annotations_json: str,
    git_remote: str,
    git_commit: str,
    hf_token: str = "",
) -> dict:
    import re
    import shutil
    import tempfile

    import boto3
    import pandas as pd

    dest_zarr = Path(_MOUNT) / f"{episode_id}.zarr"
    if dest_zarr.exists():
        return {"episode_id": episode_id, "status": "skipped (exists)"}

    _boot_container(git_remote, git_commit, hf_token)

    from bson import ObjectId
    from pymongo import MongoClient

    from egomimic.scripts.mecka_process.mecka_to_zarr import MeckaDatasetConverter

    # --- 1. Resolve R2 paths from Mongo -------------------------------------
    db = MongoClient(os.environ["MONGODB_URI"], serverSelectionTimeoutMS=30000)
    doc = db.get_default_database()["episodes"].find_one({"_id": ObjectId(episode_id)})
    if doc is None:
        raise RuntimeError(f"{episode_id}: not found in mecka-ai.episodes")

    s3_base = (doc.get("s3_base_path") or "").rstrip("/")
    frames_key = doc.get("framesKey") or ""
    if not s3_base or not frames_key:
        raise RuntimeError(
            f"{episode_id}: missing s3_base_path/framesKey "
            f"(s3_base_path={s3_base!r}, framesKey={frames_key!r})"
        )

    # The video frame-aligned with frames.csv/hands.csv is the ultrawide video
    # carrying the same episode-part tag as framesKey (e.g. frames_ep001.csv →
    # ultrawide_video_ep001.mp4), NOT the trimmed final_video.mp4.
    m = re.search(r"_ep(\d+)\.csv$", Path(frames_key).name)
    if not m:
        raise RuntimeError(f"{episode_id}: cannot derive ep tag from framesKey {frames_key!r}")
    eptag = f"ep{m.group(1)}"
    video_key = f"{s3_base}/ultrawide_video_{eptag}.mp4"

    # --- 2. Download raw files from the Mecka R2 'data' bucket ---------------
    bucket = os.environ.get("BUCKET", "data")
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ.get("R2_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL_S3"),
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )

    work = Path(tempfile.mkdtemp(prefix=f"mecka_{episode_id}_"))
    downloads = {
        "hands.csv": f"{s3_base}/hands.csv",
        "egomotion.txt": f"{s3_base}/egomotion.txt",
        "frames.csv": frames_key,
        "video.mp4": video_key,
    }
    for local_name, key in downloads.items():
        s3.download_file(bucket, key, str(work / local_name))

    # --- 3. THE SWAP: annotations.csv from the JSON preannotations ----------
    # MeckaDatasetConverter.extract_episode reads columns label/start_time/end_time.
    preanns = json.loads(annotations_json)
    pd.DataFrame(
        [
            {
                "label": a["label"],
                "start_time": a["start_seconds"],
                "end_time": a["end_seconds"],
            }
            for a in preanns
        ],
        columns=["label", "start_time", "end_time"],
    ).to_csv(work / "annotations.csv", index=False)

    # --- 4. Episode metadata JSON for the converter -------------------------
    episode_meta = {
        "id": episode_id,
        "user_id": doc.get("userId"),
        "duration": doc.get("duration"),
        "environment_id": doc.get("environment_id"),
        "scene_id": doc.get("scene_id"),
        "scene_desc": doc.get("scene_desc"),
        "objects": doc.get("objects", []),
    }
    episode_json_path = work / "episode.json"
    episode_json_path.write_text(json.dumps(episode_meta))

    # --- 5. Convert (writes <id>.zarr + <id>.mp4 preview into out_dir) ------
    out_dir = Path(tempfile.mkdtemp(prefix=f"out_{episode_id}_"))
    converter = MeckaDatasetConverter(
        episode_json_path=str(episode_json_path),
        output_dir=str(out_dir),
        repo_id="mecka/preannotations",
        arm="both",
        local_data_dir=work,
        task_description=doc.get("task_desc", "") or "",
    )
    converter.extract_episode()

    # --- 6. Move only the .zarr store onto the volume -----------------------
    produced = out_dir / f"{episode_id}.zarr"
    if not produced.exists():
        raise RuntimeError(f"{episode_id}: converter did not produce {produced}")
    shutil.move(str(produced), str(dest_zarr))
    zarr_volume.commit()

    n_frames = len(preanns)
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    return {
        "episode_id": episode_id,
        "status": "ok",
        "n_annotations": n_frames,
        "zarr": str(dest_zarr),
    }


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def ingest(
    json_path: str = "~/Downloads/preannotations-trusted-labelers.json",
    limit: int = 200,
) -> None:
    records = json.loads(Path(json_path).expanduser().read_text())
    if limit > 0:
        records = records[:limit]

    episode_ids = [r["episode_id"] for r in records]
    ann_jsons = [json.dumps(r["preannotations"]) for r in records]
    n = len(episode_ids)
    print(f"Converting {n} episode(s) from {json_path}")

    git_remote, git_commit, _ = _resolve_git_state()
    hf_token = _local_hf_token()

    succeeded, skipped, failed = [], [], []
    for result in _convert_one.map(
        episode_ids,
        ann_jsons,
        [git_remote] * n,
        [git_commit] * n,
        [hf_token] * n,
        return_exceptions=True,
    ):
        if isinstance(result, Exception):
            print(f"  ✗  {type(result).__name__}: {result}")
            failed.append(str(result))
        elif result.get("status", "").startswith("skipped"):
            print(f"  ⤼  {result['episode_id']} — {result['status']}")
            skipped.append(result["episode_id"])
        else:
            print(f"  ✓  {result['episode_id']} → {result['zarr']} ({result['n_annotations']} anns)")
            succeeded.append(result["episode_id"])

    print(
        f"\nDone: {len(succeeded)} converted, {len(skipped)} skipped, "
        f"{len(failed)} failed out of {n}."
    )
