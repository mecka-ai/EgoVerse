"""Plot a histogram of action magnitudes across a zarr dataset.

For each episode under ``--root``, opens the array at ``--key`` and reduces
each frame to a scalar magnitude. By default ("delta" mode) we plot the L2
norm of the frame-to-frame difference — i.e. how much the controlled
quantity moves between consecutive samples. That's the useful distribution
for tuning ``pause_removal_epsilon`` in the pause filter, picking action-
range normalization bounds, or sanity-checking that a new mecka batch has
sensible motion statistics.

Two other modes are available:
  --mode raw           : L2 norm of the action vector itself (full array)
  --mode delta-position: L2 norm of frame-to-frame *position* delta only,
                         assuming the array's last 3 components are xyz
                         (matches mecka's 7-dim ee_pose = quat[4]+xyz[3]).

Examples
--------
Mecka ee_pose deltas (epsilon-tuning view) across a downloaded sample:

    python -m egomimic.scripts.data_analysis.action_magnitude_histogram \\
        --root /mnt/zarr-data \\
        --key left.obs_ee_pose \\
        --mode delta-position \\
        --out ./scratch/left_ee_delta_hist.png

Raw action magnitudes for the right arm:

    python -m egomimic.scripts.data_analysis.action_magnitude_histogram \\
        --root /mnt/zarr-data --key right.obs_ee_pose --mode raw \\
        --out ./scratch/right_ee_raw_hist.png

The script doesn't need Modal or any egomimic install — only zarr +
numpy + matplotlib. Safe to point at a single .zarr group or a folder of
many episodes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr


# ---------------------------------------------------------------------------
# Magnitude reductions
# ---------------------------------------------------------------------------


def _raw_magnitudes(arr: np.ndarray) -> np.ndarray:
    """L2 norm of each frame's vector → shape (T,)."""
    return np.linalg.norm(arr, axis=-1)


def _delta_magnitudes(arr: np.ndarray) -> np.ndarray:
    """L2 norm of the frame-to-frame delta → shape (T-1,)."""
    if len(arr) < 2:
        return np.zeros(0)
    return np.linalg.norm(np.diff(arr, axis=0), axis=-1)


def _delta_position_magnitudes(arr: np.ndarray) -> np.ndarray:
    """L2 norm of the *position* component of the frame-to-frame delta.

    Assumes the trailing 3 dimensions are xyz. Useful for mecka's 7-dim
    obs_ee_pose where the first 4 components are a unit quaternion (whose
    delta is meaningless to L2-norm directly).
    """
    if arr.shape[-1] < 3 or len(arr) < 2:
        return np.zeros(0)
    pos = arr[..., -3:]
    return np.linalg.norm(np.diff(pos, axis=0), axis=-1)


_MODES = {
    "raw": _raw_magnitudes,
    "delta": _delta_magnitudes,
    "delta-position": _delta_position_magnitudes,
}


# ---------------------------------------------------------------------------
# Episode iteration
# ---------------------------------------------------------------------------


def _iter_episode_dirs(root: Path):
    """Yield .zarr episode directories under root.

    Accepts either a single .zarr/ group or a folder containing many of them
    (with or without the .zarr suffix). Skips hidden entries and anything
    that doesn't look like a zarr group.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"--root does not exist or is not a directory: {root}")
    # Single-episode case: root itself is a .zarr group.
    if (root / ".zattrs").is_file() or (root / "zarr.json").is_file():
        yield root
        return
    for child in sorted(root.iterdir()):
        if child.name.startswith("."):
            continue
        if not child.is_dir():
            continue
        if (
            (child / ".zattrs").is_file()
            or (child / "zarr.json").is_file()
            or child.name.endswith(".zarr")
        ):
            yield child


def _read_array(episode_dir: Path, key: str) -> np.ndarray | None:
    """Open the zarr group and return the requested key's data; None if missing."""
    try:
        store = zarr.open_group(str(episode_dir), mode="r")
    except Exception as e:
        print(f"  [skip] {episode_dir.name}: open failed ({e})", file=sys.stderr)
        return None
    try:
        return np.asarray(store[key][:])
    except KeyError:
        return None
    except Exception as e:
        print(f"  [skip] {episode_dir.name}: read {key!r} failed ({e})", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Path to a zarr dataset root (folder of .zarr/ episodes) "
        "OR a single .zarr/ group.",
    )
    p.add_argument(
        "--key",
        default="left.obs_ee_pose",
        help="Zarr array key to read from each episode (default: left.obs_ee_pose).",
    )
    p.add_argument(
        "--mode",
        choices=sorted(_MODES.keys()),
        default="delta-position",
        help="Magnitude reduction (default: delta-position).",
    )
    p.add_argument("--bins", type=int, default=100, help="Histogram bin count.")
    p.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Cap on how many episodes to read (default: all).",
    )
    p.add_argument(
        "--log-y",
        action="store_true",
        help="Use a log-scale y-axis (good when most mass sits near zero).",
    )
    p.add_argument(
        "--clip-percentile",
        type=float,
        default=99.5,
        help="Drop values above this percentile before plotting "
        "(default 99.5 — keeps the long tail from squashing the plot).",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output PNG path. Parent dirs are created if missing.",
    )
    return p


def main() -> int:
    args = build_argparser().parse_args()

    reducer = _MODES[args.mode]
    episodes_seen = 0
    episodes_skipped = 0
    all_magnitudes: list[np.ndarray] = []

    print(
        f"Scanning {args.root} for episodes (key={args.key}, mode={args.mode})..."
    )
    for ep in _iter_episode_dirs(args.root):
        if args.max_episodes is not None and episodes_seen >= args.max_episodes:
            break
        arr = _read_array(ep, args.key)
        if arr is None:
            episodes_skipped += 1
            continue
        mags = reducer(arr)
        if mags.size == 0:
            episodes_skipped += 1
            continue
        all_magnitudes.append(mags)
        episodes_seen += 1
        if episodes_seen % 50 == 0:
            print(f"  ...read {episodes_seen} episodes")

    if not all_magnitudes:
        print(
            f"No usable episodes found at {args.root} with key {args.key!r}.",
            file=sys.stderr,
        )
        return 1

    data = np.concatenate(all_magnitudes)
    p_clip = float(np.percentile(data, args.clip_percentile))
    clipped = data[data <= p_clip]
    n_clipped = data.size - clipped.size

    # Report a few useful summary stats.
    print(
        f"Read {episodes_seen} episodes ({episodes_skipped} skipped). "
        f"{data.size:,} magnitude samples."
    )
    print(
        f"  min={data.min():.6f}  median={np.median(data):.6f}  "
        f"mean={data.mean():.6f}  max={data.max():.6f}"
    )
    print(
        f"  p50={np.percentile(data, 50):.6f}  p90={np.percentile(data, 90):.6f}  "
        f"p99={np.percentile(data, 99):.6f}  p99.9={np.percentile(data, 99.9):.6f}"
    )
    if n_clipped:
        print(
            f"  clipped {n_clipped:,} samples above the {args.clip_percentile}th "
            f"percentile (={p_clip:.6f}) for the histogram plot"
        )

    # Plot.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(clipped, bins=args.bins, color="#1f77b4", alpha=0.85, edgecolor="white")
    if args.log_y:
        ax.set_yscale("log")
    ax.set_xlabel(f"Magnitude ({args.mode})")
    ax.set_ylabel("Frame count")
    ax.set_title(
        f"Action magnitude histogram\n"
        f"key={args.key}, mode={args.mode}, "
        f"episodes={episodes_seen}, samples={data.size:,}"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"Wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
