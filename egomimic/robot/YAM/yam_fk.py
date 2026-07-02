"""URDF forward kinematics for the YAM arm (eef_link frame).

WHY THIS EXISTS
  YAMInterface.get_pose (and therefore the `eepose` observation) computes FK from
  the MuJoCo model via mink at the `grasp_site` frame. That MuJoCo model
  (`yam.xml`) disagrees with the URDF (`yam.urdf`) at the JOINT level, not just by
  a constant frame offset: on real data the grasp_site FK tracks the physical
  AprilTag at only corr 0.62 / ~10 deg, while the URDF `eef_link` FK tracks it at
  corr 0.98 / ~2.8 deg. So the URDF FK is the correct one; the mink/grasp_site FK
  is wrong for anything that needs true EE orientation (hand-eye calibration, and
  arguably the policy observation).

  This module computes the correct EE pose from `joint_positions` using the
  vendored URDF (models/yam.urdf) via pytorch_kinematics. Pinocchio would also
  work (that's what the reference used) but pin>=4 forces numpy>=2, which is
  incompatible with the emimic env; pytorch_kinematics is already a dependency and
  produces an identical result on this URDF.

USAGE
    fk = YamFK()
    T = fk.fk(joint_positions[t][0:6])      # 4x4 base->eef_link for the LEFT arm
    T = fk.fk(joint_positions[t][7:13])     # RIGHT arm (joints 7..12)
"""
import os

import numpy as np
import torch
import pytorch_kinematics as pk

_URDF_PATH = os.path.join(os.path.dirname(__file__), "models", "yam.urdf")
_EE_LINK = "eef_link"
N_ARM_JOINTS = 6
# joint_positions / eepose layout is 7 slots per arm (6 arm joints + gripper).
ARM_JOINT_OFFSET = {"left": 0, "right": 7}


class YamFK:
    """Forward kinematics to the URDF eef_link, from the 6 arm joint angles."""

    def __init__(self, urdf_path=_URDF_PATH, ee_link=_EE_LINK):
        with open(urdf_path, "rb") as f:
            self.chain = pk.build_serial_chain_from_urdf(f.read(), ee_link)

    def fk(self, q):
        """q: (>=6,) arm joint angles (rad). Returns 4x4 base->ee SE(3)."""
        q = np.asarray(q, dtype=np.float32).reshape(-1)[:N_ARM_JOINTS]
        M = self.chain.forward_kinematics(torch.from_numpy(q).unsqueeze(0)).get_matrix()
        return M[0].cpu().numpy().astype(np.float64)

    def fk_arm(self, joint_positions_row, arm):
        """joint_positions_row: (14,) full row; arm: 'left'|'right' -> 4x4 SE(3)."""
        off = ARM_JOINT_OFFSET[arm]
        return self.fk(np.asarray(joint_positions_row)[off:off + N_ARM_JOINTS])

    def fk_batch(self, qs):
        """qs: (N, >=6) arm joint angles. Returns (N, 4, 4) base->ee SE(3).

        One vectorized call -- prefer this over per-frame fk() in a tight loop,
        and keep it OUT of any loop that also calls a C-extension like the
        AprilTag detector (interleaving torch with it can segfault)."""
        qs = np.asarray(qs, dtype=np.float32)[:, :N_ARM_JOINTS]
        M = self.chain.forward_kinematics(torch.from_numpy(qs)).get_matrix()
        return M.cpu().numpy().astype(np.float64)
