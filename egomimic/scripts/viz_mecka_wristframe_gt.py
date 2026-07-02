"""Visualize a mecka episode's GROUND-TRUTH actions under the wristframe-6D transform.

Pipeline exercised end to end, exactly as in training + eval viz:

  raw zarr (xyz+quat, world) --Mecka.get_transform_list("cartesian_wristframe_6d")-->
  wrist-frame 6D action chunks + head-frame 6D proprio
  --_build_human_cartesian_revert_eef_frame_transform_list(rot_repr="6d")-->
  head(camera)-frame ypr --project with INTRINSICS["mecka"]--> dots on the image.

The GT chunk is drawn twice: as the batch ground truth (green) and again as a fake
"prediction" (red) through the deep-copied prediction revert path, so the two
overlays must coincide pixel-for-pixel. The script also cross-checks the reverted
wrist-frame chunk numerically against an independent plain-"cartesian" (never
enters the wrist frame) pipeline on the same frames and fails loudly above 1e-5.

Usage (repo root, emimic venv):
    python egomimic/scripts/viz_mecka_wristframe_gt.py \
        --episode /workspace/mecka_random_250h_zarr/<hash>.zarr \
        --out /tmp/mecka_wf6d_viz [--frames 6]
"""

import argparse
import os
import sys

# Anchor imports to the repo this script lives in. Invoked as
# `python egomimic/scripts/viz_mecka_wristframe_gt.py`, sys.path[0] is the
# scripts/ dir, so `import egomimic` would otherwise resolve through the venv's
# editable install — possibly a DIFFERENT checkout/worktree missing the
# wristframe-6d code.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import imageio.v2 as imageio
import numpy as np
import torch
import zarr
from scipy.spatial.transform import Rotation as SciR

from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.human import (
    Mecka,
    _build_human_cartesian_revert_eef_frame_transform_list,
)
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset


def _make_dataset(episode, mode):
    transform_list = Mecka.get_transform_list(mode=mode)
    if transform_list is None:
        raise SystemExit(
            f"Mecka.get_transform_list returned None for mode={mode!r} — the "
            f"egomimic you imported ({os.path.dirname(sys.modules['egomimic'].__file__)}) "
            "does not support this mode. Run from a checkout of the "
            "aidan/wristframe-6d branch (or cherry-pick its mecka commit)."
        )
    return ZarrDataset(
        episode,
        key_map=Mecka.get_keymap(mode="cartesian"),
        transform_list=transform_list,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", required=True, help="path to one mecka .zarr episode")
    ap.add_argument("--out", required=True, help="output dir for overlay PNGs")
    ap.add_argument("--frames", type=int, default=6, help="number of anchor frames")
    args = ap.parse_args()

    ds_wrist = _make_dataset(args.episode, "cartesian_wristframe_6d")
    ds_cart = _make_dataset(args.episode, "cartesian")  # independent reference (ypr)
    n = len(ds_wrist)

    # Mecka episodes end with all-zero SENTINEL pose rows (invalid frames the
    # training-time pose-validity / bounds checks reject). A zero quaternion is
    # not a rotation, so anchor only where the whole 30-frame chunk has valid
    # quats on both arms and the head.
    g = zarr.open_group(args.episode, mode="r")
    valid = np.ones(n, dtype=bool)
    for k in ("left.obs_ee_pose", "right.obs_ee_pose", "obs_head_pose"):
        valid &= np.linalg.norm(np.asarray(g[k])[:n, 3:], axis=1) > 1e-3
    horizon = 30
    chunk_ok = np.array(
        [valid[i : min(i + horizon, n)].all() for i in range(n)], dtype=bool
    )
    ok_idx = np.flatnonzero(chunk_ok)
    if len(ok_idx) == 0:
        raise SystemExit("no frame with a fully-valid 30-frame chunk in this episode")
    anchors = ok_idx[np.linspace(0, len(ok_idx) - 1, args.frames, dtype=int)]
    print(
        f"episode: {args.episode} ({n} frames, {n - int(valid.sum())} sentinel); "
        f"anchors: {anchors.tolist()}"
    )

    revert = _build_human_cartesian_revert_eef_frame_transform_list(rot_repr="6d")
    emb_id = get_embodiment_id("mecka_bimanual")
    os.makedirs(args.out, exist_ok=True)

    max_err = 0.0
    for a in anchors:
        s = ds_wrist[int(a)]
        ac = s["actions_cartesian"]  # (100, 18) wrist-frame 6D
        assert tuple(ac.shape) == (100, 18), ac.shape
        obs = s["observations.state.ee_pose"]  # (18,) head-frame 6D
        assert tuple(obs.shape) == (18,), obs.shape

        # Numeric cross-check: revert(wrist 6D) must equal the independent
        # plain-cartesian pipeline (head-frame ypr, never entered wrist frame).
        d = {
            "actions_cartesian": np.asarray(ac, dtype=np.float64),
            "observations.state.ee_pose": np.asarray(obs, dtype=np.float64),
        }
        for t in revert:
            d = t.transform(d)
        reverted = np.asarray(d["actions_cartesian"])  # (100, 12) head-frame ypr
        cart = np.asarray(ds_cart[int(a)]["actions_cartesian"], dtype=np.float64)
        # Positions and the right rotation must match directly. The LEFT hand
        # carries the deliberate local Rz(pi) frame unification
        # (unify_hand_frames in the wristframe mode; plain "cartesian" keeps
        # the raw mirrored frame), so its rotation is compared modulo E.
        E3 = np.diag([-1.0, -1.0, 1.0])
        pos_err = max(
            np.abs(reverted[:, 0:3] - cart[:, 0:3]).max(),
            np.abs(reverted[:, 6:9] - cart[:, 6:9]).max(),
        )
        rl_rev = SciR.from_euler("ZYX", reverted[:, 3:6]).as_matrix()
        rl_cart = SciR.from_euler("ZYX", cart[:, 3:6]).as_matrix()
        rr_rev = SciR.from_euler("ZYX", reverted[:, 9:12]).as_matrix()
        rr_cart = SciR.from_euler("ZYX", cart[:, 9:12]).as_matrix()
        rot_err = max(
            np.abs(rl_rev - rl_cart @ E3).max(),
            np.abs(rr_rev - rr_cart).max(),
        )
        err = max(pos_err, rot_err)
        max_err = max(max_err, err)
        assert err < 1e-5, f"revert mismatch at frame {a}: {err}"

        # Render: GT (green) + the same GT chunk as a "prediction" (red) through
        # viz_gt_preds' prediction revert path -- overlays must coincide.
        batch = {
            "actions_cartesian": ac[None],
            "observations.state.ee_pose": obs[None],
            Mecka.VIZ_IMAGE_KEY: s[Mecka.VIZ_IMAGE_KEY][None],
            "embodiment": torch.tensor([emb_id]),
        }
        preds = {"mecka_bimanual_actions_cartesian": ac[None].clone()}
        ims = Mecka.viz_gt_preds(
            preds,
            batch,
            image_key=Mecka.VIZ_IMAGE_KEY,
            action_key="actions_cartesian",
            transform_list=revert,
            mode="traj",
        )
        # Overlay the wrist orientation as positive x/y/z axis arrows (mode
        # "axes": rot column * axis_len from each arm's anchor pose, x=red
        # y=green z=blue legend) using the reverted head-frame GT chunk.
        frame = Mecka.viz(ims[0], reverted, mode="axes")
        out_path = os.path.join(args.out, f"frame_{int(a):05d}.png")
        imageio.imwrite(out_path, frame.astype(np.uint8))
        print(f"  frame {a}: revert err {err:.2e} -> {out_path}")

    print(
        f"\nOK: wrist->cam revert matches plain-cartesian pipeline "
        f"(max err {max_err:.2e}) on {len(anchors)} frames; overlays in {args.out}"
    )


if __name__ == "__main__":
    main()
