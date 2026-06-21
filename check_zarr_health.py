#!/usr/bin/env python
"""Validate converted mecka_random_250h Zarr episodes.

Catches the failure modes that batch downloads produce:
  * partial <hash>.zarr left by a process killed mid-copytree (missing arrays
    or missing trailing chunk files) — these are silently skipped as "done" by
    the idempotent downloader, so they must be found explicitly;
  * unreadable / corrupt stores;
  * (optional) missing episodes from the expected id-list.

For each <hash>.zarr it: opens the store, reads the `features` attr (the list of
arrays the converter promised), checks every one of those arrays exists, that
all time-series arrays share one frame count (> 0), and that the first AND last
element of each array actually read back (forces a chunk read → catches a
truncated copy). This is a fast structural check, not a full byte scan.

Usage (run with the project venv):
    emimic/bin/python check_zarr_health.py \
        --dir /workspace/mecka_random_250h_zarr \
        --ids-file mecka_random_250h.json \
        --workers 16
    # add --delete to remove bad/partial <hash>.zarr (+ .mp4) so the next
    # download run re-creates them.
"""

import argparse
import json
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore")  # zarr v3 UnstableSpecificationWarning etc.


def _check_one(zpath: Path):
    """Return (hash, status, detail). status in {ok, bad}."""
    h = zpath.name[: -len(".zarr")]
    try:
        import numpy as np  # noqa: F401
        import zarr

        g = zarr.open(str(zpath), mode="r")
        feats = dict(g.attrs).get("features")
        if not feats:
            return h, "bad", "no 'features' attr (incomplete root metadata)"
        frame_lengths = set()
        for key in feats:
            try:
                arr = g[key]
            except Exception:
                return h, "bad", f"missing array '{key}'"
            n = arr.shape[0]
            # `annotations` (dtype 'json') is indexed by annotation segments, not
            # frames, and may legitimately be length 0 — so it is exempt from the
            # frame-count consistency + non-empty checks. Only the frame-indexed
            # arrays (images/poses) must agree on length and be non-empty.
            is_frame_array = (feats.get(key) or {}).get("dtype") != "json"
            if is_frame_array:
                if n == 0:
                    return h, "bad", f"frame array '{key}' is empty"
                frame_lengths.add(n)
            # force first + last chunk reads -> catches a truncated copytree
            if n > 0:
                try:
                    _ = arr[0]
                    _ = arr[n - 1]
                except Exception as e:
                    return h, "bad", f"array '{key}' unreadable at edge: {e}"
        if len(frame_lengths) > 1:
            return (
                h,
                "bad",
                f"inconsistent frame counts across arrays: {sorted(frame_lengths)}",
            )
        if not frame_lengths:
            return h, "bad", "no frame-indexed arrays found"
        return h, "ok", None
    except Exception as e:
        return h, "bad", f"open failed: {e}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate mecka Zarr episodes")
    ap.add_argument("--dir", default="/workspace/mecka_random_250h_zarr")
    ap.add_argument(
        "--ids-file",
        default="",
        help="Optional JSON/txt id-list; also reports MISSING episodes.",
    )
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--delete",
        action="store_true",
        help="Delete bad/partial <hash>.zarr (+ .mp4) so a rerun recreates them.",
    )
    args = ap.parse_args()

    out = Path(args.dir)
    zarrs = sorted(out.glob("*.zarr"))
    print(f"Scanning {len(zarrs)} .zarr dirs in {out} with {args.workers} threads...")

    bad = []
    ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_check_one, z): z for z in zarrs}
        for i, fut in enumerate(as_completed(futs), 1):
            h, status, detail = fut.result()
            if status == "ok":
                ok += 1
            else:
                bad.append((h, detail))
                print(f"  BAD  {h}: {detail}")
            if i % 500 == 0:
                print(f"  ...{i}/{len(zarrs)} checked, {len(bad)} bad so far")

    print("=" * 60)
    print(f"OK: {ok}   BAD/partial: {len(bad)}   total present: {len(zarrs)}")

    if args.ids_file:
        text = Path(args.ids_file).read_text()
        if args.ids_file.endswith(".json"):
            ids = [str(x).strip() for x in json.loads(text) if str(x).strip()]
        else:
            ids = [
                line.strip().strip(",").strip('"')
                for line in text.splitlines()
                if line.strip() and not line.startswith("#")
            ]
        ids = list(dict.fromkeys(ids))  # dedup, keep order
        present = {z.name[: -len(".zarr")] for z in zarrs}
        good = present - {h for h, _ in bad}
        missing = [h for h in ids if h not in good]  # absent OR present-but-bad
        print(
            f"Expected {len(ids)} | complete {len(good)} | "
            f"to (re)download {len(missing)}"
        )
        Path(out / "missing_or_bad_hashes.txt").write_text("\n".join(missing) + "\n")
        print(f"Wrote re-download list -> {out / 'missing_or_bad_hashes.txt'}")

    if args.delete and bad:
        for h, _ in bad:
            for suffix in (".zarr", ".mp4"):
                p = out / f"{h}{suffix}"
                if p.is_dir():
                    import shutil

                    shutil.rmtree(p, ignore_errors=True)
                elif p.exists():
                    p.unlink()
        print(f"Deleted {len(bad)} bad .zarr (+ .mp4); rerun the download to recreate.")

    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
