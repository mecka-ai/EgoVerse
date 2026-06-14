"""
Local batch download + conversion of Mecka episodes to Zarr.

For each episode hash this:
  1. resolves the MongoDB doc + presigned R2 asset URLs,
  2. fetches VLM annotations from `episode_vlm_segments`,
  3. runs the base `MeckaDatasetConverter` (egomimic/.../mecka_to_zarr.py)
     to produce `<hash>.zarr` (+ preview `<hash>.mp4`),
  4. copies the result into `--output-dir`.

It reuses the proven resolve/convert path (`_prepare_episode`) from
`modal_mecka_to_zarr.py` but runs entirely locally — no Modal account, no
Modal Volume. Credentials come from the environment (MONGODB_URI, R2_*).

Usage:
    set -a; . /root/.egoverse_env; set +a
    python download_episodes_local.py \
        --ids-file /workspace/episode_hashes.txt \
        --output-dir /workspace/zarr_episodes
"""

import argparse
import importlib.util
import logging
import os
import shutil
import tempfile
import traceback
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("download_episodes_local")

_MODAL_SCRIPT = Path(__file__).resolve().parent / "modal_mecka_to_zarr.py"


def _load_modal_helpers():
    """Import the resolve/convert helpers from modal_mecka_to_zarr.py.

    The module imports cleanly without Modal auth — `modal.Volume.from_name`
    is lazy and never hits the network at import time.
    """
    spec = importlib.util.spec_from_file_location("mecka_modal_helpers", _MODAL_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _check_env():
    missing = []
    if not os.environ.get("MONGODB_URI"):
        missing.append("MONGODB_URI")
    if not (os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("R2_ACCESS_KEY")):
        missing.append("R2_ACCESS_KEY_ID")
    if not (os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get("R2_SECRET_KEY")):
        missing.append("R2_SECRET_ACCESS_KEY")
    if not (os.environ.get("R2_BUCKET") or os.environ.get("BUCKET")):
        missing.append("R2_BUCKET (or BUCKET)")
    if not (
        os.environ.get("AWS_ENDPOINT_URL_S3")
        or os.environ.get("R2_ENDPOINT_URL")
        or os.environ.get("R2_ENDPOINT")
        or os.environ.get("R2_ACCOUNT_ID")
    ):
        missing.append("AWS_ENDPOINT_URL_S3 (or R2_ACCOUNT_ID)")
    if missing:
        raise SystemExit(
            "Missing required credentials in environment: "
            + ", ".join(missing)
            + "\nLoad them first, e.g.:  set -a; . /root/.egoverse_env; set +a"
        )


def _read_ids(path: str) -> list[str]:
    ids = []
    for line in Path(path).read_text().splitlines():
        h = line.strip().strip(",").strip('"')
        if h and not h.startswith("#"):
            ids.append(h)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Mecka -> Zarr batch download")
    parser.add_argument("--ids-file", default="/workspace/episode_hashes.txt")
    parser.add_argument("--output-dir", default="/workspace/zarr_episodes")
    parser.add_argument(
        "--force-task-type",
        choices=["flagship", "freeform"],
        default="",
        help="Skip MongoDB classification and force this task type.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-convert episodes even if <hash>.zarr already exists in output-dir.",
    )
    args = parser.parse_args()

    _check_env()
    # _parse_storage_key reads R2_BUCKET; mirror BUCKET into it for convenience.
    if not os.environ.get("R2_BUCKET") and os.environ.get("BUCKET"):
        os.environ["R2_BUCKET"] = os.environ["BUCKET"]

    mm = _load_modal_helpers()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = _read_ids(args.ids_file)
    logger.info("Loaded %d episode hashes from %s", len(ids), args.ids_file)
    logger.info("Output dir: %s", out_dir)

    db = mm._get_mongo_db()

    ok, skipped, failed = [], [], []
    for i, h in enumerate(ids, 1):
        dest_zarr = out_dir / f"{h}.zarr"
        if dest_zarr.exists() and not args.overwrite:
            logger.info("[%d/%d] %s — already present, skipping", i, len(ids), h)
            skipped.append(h)
            continue

        if args.force_task_type:
            task_type = args.force_task_type
        else:
            try:
                task_type = "freeform" if mm._is_freeform(db, h) else "flagship"
            except Exception as e:
                logger.error(
                    "[%d/%d] %s — classification failed: %s", i, len(ids), h, e
                )
                failed.append((h, f"classify: {e}"))
                continue

        logger.info("[%d/%d] %s — converting (type=%s)", i, len(ids), h, task_type)
        tmp_dir = tempfile.mkdtemp(prefix=f"zarr_local_{h}_")
        try:
            result = mm._prepare_episode(h, task_type, tmp_dir)

            if dest_zarr.exists():
                shutil.rmtree(dest_zarr)
            shutil.copytree(result["zarr_dir"], dest_zarr)

            if result.get("mp4_path") and os.path.exists(result["mp4_path"]):
                shutil.copy2(result["mp4_path"], out_dir / f"{h}.mp4")

            logger.info(
                "[%d/%d] %s — done (%s annotations) -> %s",
                i,
                len(ids),
                h,
                result.get("num_annotations"),
                dest_zarr,
            )
            ok.append(h)
        except Exception as e:
            logger.error("[%d/%d] %s — FAILED: %s", i, len(ids), h, e)
            logger.debug(traceback.format_exc())
            failed.append((h, str(e)))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info("=" * 60)
    logger.info(
        "DONE  ok=%d  skipped=%d  failed=%d", len(ok), len(skipped), len(failed)
    )
    if failed:
        logger.info("Failures:")
        for h, err in failed:
            logger.info("  %s: %s", h, err)
        # Write a retry list of just the failed hashes.
        retry = out_dir / "failed_hashes.txt"
        retry.write_text("\n".join(h for h, _ in failed) + "\n")
        logger.info("Wrote failed hashes -> %s", retry)


if __name__ == "__main__":
    main()
