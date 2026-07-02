import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def xyzw_to_wxyz(xyzw):
    return np.concatenate([xyzw[..., 3:4], xyzw[..., :3]], axis=-1)


def wxyz_to_xyzw(wxyz):
    return np.concatenate([wxyz[..., 1:4], wxyz[..., 0:1]], axis=-1)


def _interpolate_euler(seq: np.ndarray, chunk_length: int) -> np.ndarray:
    """Euler-aware interpolation for a single (T, 6) or (T, 7) sequence."""
    T, D = seq.shape
    assert D in (6, 7), f"Expected 6 or 7 dims, got {D}"

    if np.any(seq >= 1e8):
        return np.full((chunk_length, D), 1e9)

    old_time = np.linspace(0, 1, T)
    new_time = np.linspace(0, 1, chunk_length)

    trans_interp = interp1d(old_time, seq[:, :3], axis=0, kind="linear")(new_time)

    rot_unwrapped = np.unwrap(seq[:, 3:6], axis=0)
    rot_interp = interp1d(old_time, rot_unwrapped, axis=0, kind="linear")(new_time)
    rot_interp = (rot_interp + np.pi) % (2 * np.pi) - np.pi

    if D == 6:
        return np.concatenate([trans_interp, rot_interp], axis=-1)

    grip_interp = interp1d(old_time, seq[:, 6:7], axis=0, kind="linear")(new_time)
    return np.concatenate([trans_interp, rot_interp, grip_interp], axis=-1)


def _interpolate_linear(seq: np.ndarray, chunk_length: int) -> np.ndarray:
    """Simple linear interpolation for arbitrary (T, D) arrays."""
    T, _ = seq.shape
    old_time = np.linspace(0, 1, T)
    new_time = np.linspace(0, 1, chunk_length)
    return interp1d(old_time, seq, axis=0, kind="linear")(new_time)


def _interpolate_linear_batch(seqs: np.ndarray, chunk_length: int) -> np.ndarray:
    """
    Batched linear interpolation: (B, T, D) → (B, chunk_length, D).

    All B sequences share the same uniform time grid, so we reshape to
    (T, B*D), call interp1d once, then reshape back.
    """
    B, T, D = seqs.shape
    if T == 1:
        return np.repeat(seqs, chunk_length, axis=1)
    old_time = np.linspace(0, 1, T)
    new_time = np.linspace(0, 1, chunk_length)
    flat = seqs.transpose(1, 0, 2).reshape(T, B * D)  # (T, B*D)
    result = interp1d(old_time, flat, axis=0, kind="linear")(new_time)  # (chunk_length, B*D)
    return result.reshape(chunk_length, B, D).transpose(1, 0, 2)  # (B, chunk_length, D)


def _interpolate_quat_wxyz_batch(seqs: np.ndarray, chunk_length: int) -> np.ndarray:
    """
    Batched quaternion-aware interpolation: (B, T, 7) → (B, chunk_length, 7).

    All B sequences are interpolated simultaneously using numpy broadcasting.
    Sign-continuity is enforced with a T-step loop vectorised over B — this
    loop is over the *horizon* (≤30), not over episodes, so it is negligible.
    Falls back to 1e9 sentinel if any input contains a sentinel value.
    """
    B, T, D = seqs.shape
    if D != 7:
        raise ValueError(f"_interpolate_quat_wxyz_batch expects last dim 7, got {D}")

    if np.any(seqs >= 1e8):
        return np.full((B, chunk_length, 7), 1e9, dtype=seqs.dtype)

    old_time = np.linspace(0, 1, T)
    new_time = np.linspace(0, 1, chunk_length)

    xyz = seqs[:, :, :3].astype(np.float64)   # (B, T, 3)
    quat_wxyz = seqs[:, :, 3:7].astype(np.float64)  # (B, T, 4)
    quat_xyzw = quat_wxyz[:, :, [1, 2, 3, 0]]  # (B, T, 4)

    norms = np.linalg.norm(quat_xyzw, axis=-1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Zero-norm quaternion found in _interpolate_quat_wxyz_batch input")
    quat_xyzw = quat_xyzw / norms

    # Sign-continuity: loop over T-1 horizon steps, vectorised over B.
    q = quat_xyzw.copy()
    for i in range(1, T):
        dots = (q[:, i - 1, :] * q[:, i, :]).sum(-1)  # (B,)
        flip = dots < 0                                  # (B,)
        q[:, i, :] = np.where(flip[:, None], -q[:, i, :], q[:, i, :])

    # Batched linear interpolation of xyz.
    xyz_interp = _interpolate_linear_batch(xyz, chunk_length)  # (B, chunk_length, 3)

    if T == 1:
        quat_interp_wxyz = np.repeat(quat_wxyz, chunk_length, axis=1)  # (B, chunk_length, 4)
    else:
        # Lookup surrounding keyframe indices for each output time point.
        idx = np.clip(np.searchsorted(old_time, new_time, side="right") - 1, 0, T - 2)
        idx1 = idx + 1
        t0 = old_time[idx]
        t1 = old_time[idx1]
        alpha = np.where(t1 > t0, (new_time - t0) / (t1 - t0), 0.5)  # (chunk_length,)

        q0 = q[:, idx, :]   # (B, chunk_length, 4)
        q1 = q[:, idx1, :]  # (B, chunk_length, 4)

        dot = np.clip((q0 * q1).sum(-1, keepdims=True), -1.0, 1.0)  # (B, chunk_length, 1)
        # After sign-continuity pass, consecutive dots are ≥ 0, but keep the
        # sign-flip guard for SLERP between non-consecutive frames.
        q1_adj = np.where(dot < 0, -q1, q1)
        dot = np.clip((q0 * q1_adj).sum(-1, keepdims=True), -1.0, 1.0)

        theta = np.arccos(dot)            # (B, chunk_length, 1)
        sin_theta = np.sin(theta)

        a = alpha[None, :, None]          # (1, chunk_length, 1) for broadcasting
        w0 = np.where(sin_theta > 1e-6,
                      np.sin((1.0 - a) * theta) / (sin_theta + 1e-10), 1.0 - a)
        w1 = np.where(sin_theta > 1e-6,
                      np.sin(a * theta) / (sin_theta + 1e-10), a)

        quat_interp = w0 * q0 + w1 * q1_adj   # (B, chunk_length, 4)
        norms_out = np.linalg.norm(quat_interp, axis=-1, keepdims=True)
        quat_interp /= np.where(norms_out > 1e-10, norms_out, 1.0)
        quat_interp_wxyz = quat_interp[:, :, [3, 0, 1, 2]]  # (B, chunk_length, 4) xyzw→wxyz

    dtype = seqs.dtype if np.issubdtype(seqs.dtype, np.floating) else np.float64
    return np.concatenate([xyz_interp, quat_interp_wxyz], axis=-1).astype(dtype, copy=False)


def _interpolate_quat_wxyz(seq: np.ndarray, chunk_length: int) -> np.ndarray:
    """Quaternion-aware interpolation for a single (T, 7) sequence."""
    T, D = seq.shape
    if D != 7:
        raise ValueError(f"Expected 7 dims for xyz+quat(wxyz), got {D}")

    if np.any(seq >= 1e8):
        return np.full((chunk_length, D), 1e9)

    old_time = np.linspace(0, 1, T)
    new_time = np.linspace(0, 1, chunk_length)

    trans_interp = interp1d(old_time, seq[:, :3], axis=0, kind="linear")(new_time)
    quat_wxyz = np.asarray(seq[:, 3:7], dtype=np.float64)
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]

    norms = np.linalg.norm(quat_xyzw, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Found zero-norm quaternion in input sequence.")
    quat_xyzw = quat_xyzw / norms

    # Enforce sign continuity to avoid long-path interpolation.
    quat_contiguous = quat_xyzw.copy()
    for i in range(1, T):
        if np.dot(quat_contiguous[i - 1], quat_contiguous[i]) < 0:
            quat_contiguous[i] = -quat_contiguous[i]

    if T == 1:
        quat_interp_xyzw = np.repeat(quat_contiguous[:1], chunk_length, axis=0)
    else:
        slerp = Slerp(old_time, R.from_quat(quat_contiguous))
        quat_interp_xyzw = slerp(new_time).as_quat()

    quat_interp_wxyz = quat_interp_xyzw[:, [3, 0, 1, 2]]
    dtype = seq.dtype if np.issubdtype(seq.dtype, np.floating) else np.float64
    return np.concatenate([trans_interp, quat_interp_wxyz], axis=-1).astype(
        dtype, copy=False
    )


def _interpolate_xyz(seq: np.ndarray, chunk_length: int) -> np.ndarray:
    """Linear interpolation for arbitrary (T, 3) arrays or (T, K, 3) arrays."""
    T = seq.shape[0]
    old_time = np.linspace(0, 1, T)
    new_time = np.linspace(0, 1, chunk_length)
    return interp1d(old_time, seq, axis=0, kind="linear")(new_time)


def _matrix_to_xyzypr(mats: np.ndarray) -> np.ndarray:
    """
    args:
        mats: (B, 4, 4) array of SE3 transformation matrices
    returns:
        (B, 6) np.array of [[x, y, z, yaw, pitch, roll]]
    """
    if mats.ndim != 3 or mats.shape[-2:] != (4, 4):
        raise ValueError(f"Expected (B, 4, 4) array, got shape {mats.shape}")

    mats = np.asarray(mats)
    dtype = mats.dtype if np.issubdtype(mats.dtype, np.floating) else np.float64

    xyz = mats[:, :3, 3]
    ypr = R.from_matrix(mats[:, :3, :3]).as_euler("ZYX", degrees=False)

    return np.concatenate([xyz, ypr], axis=-1).astype(dtype, copy=False)


def _xyzypr_to_matrix(xyzypr: np.ndarray) -> np.ndarray:
    """
    args:
        xyzypr: (B, 6) np.array of [[x, y, z, yaw, pitch, roll]]
    returns:
        (B, 4, 4) array of SE3 transformation matrices
    """
    if xyzypr.ndim != 2 or xyzypr.shape[-1] != 6:
        raise ValueError(f"Expected (B, 6) array, got shape {xyzypr.shape}")
    B = xyzypr.shape[0]
    dtype = xyzypr.dtype if np.issubdtype(xyzypr.dtype, np.floating) else np.float64

    mats = np.broadcast_to(np.eye(4, dtype=dtype), (B, 4, 4)).copy()
    mats[:, :3, :3] = R.from_euler("ZYX", xyzypr[:, 3:6], degrees=False).as_matrix()
    mats[:, :3, 3] = xyzypr[:, :3]
    return mats


def _matrix_to_xyzwxyz(mats: np.ndarray) -> np.ndarray:
    """
    args:
        mats: (B, 4, 4) array of SE3 transformation matrices
    returns:
        (B, 7) np.array of [[x, y, z, qw, qx, qy, qz]]
    """
    if mats.ndim != 3 or mats.shape[-2:] != (4, 4):
        raise ValueError(f"Expected (B, 4, 4) array, got shape {mats.shape}")

    mats = np.asarray(mats)
    dtype = mats.dtype if np.issubdtype(mats.dtype, np.floating) else np.float64

    xyz = mats[:, :3, 3]
    quat_xyzw = R.from_matrix(mats[:, :3, :3]).as_quat()
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]

    return np.concatenate([xyz, quat_wxyz], axis=-1).astype(dtype, copy=False)


def _xyzwxyz_to_matrix(xyzwxyz: np.ndarray) -> np.ndarray:
    """
    args:
        xyzwxyz: (B, 7) np.array of [[x, y, z, qw, qx, qy, qz]]
    returns:
        (B, 4, 4) array of SE3 transformation matrices
    """
    if xyzwxyz.ndim != 2 or xyzwxyz.shape[-1] != 7:
        raise ValueError(f"Expected (B, 7) array, got shape {xyzwxyz.shape}")

    B = xyzwxyz.shape[0]
    dtype = xyzwxyz.dtype if np.issubdtype(xyzwxyz.dtype, np.floating) else np.float64

    mats = np.broadcast_to(np.eye(4, dtype=dtype), (B, 4, 4)).copy()
    quat_xyzw = xyzwxyz[:, [4, 5, 6, 3]]

    mats[:, :3, :3] = R.from_quat(quat_xyzw).as_matrix()
    mats[:, :3, 3] = xyzwxyz[:, :3]

    return mats


def T_rot_orientation(T: np.ndarray, rot_orientation: np.ndarray) -> np.ndarray:
    """
    Permute the rotation matrix of a SE(3) transformation.
    """
    rot = T[:3, :3]
    rot = rot @ rot_orientation
    T[:3, :3] = rot
    return T


def _xyz_to_matrix(xyz: np.ndarray) -> np.ndarray:
    """
    args:
        xyz: (B, 3) np.array of [[x, y, z]]
    returns:
        (B, 4, 4) array of SE3 transformation matrices
    """
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError(f"Expected (B, 3) array, got shape {xyz.shape}")
    B = xyz.shape[0]
    dtype = xyz.dtype if np.issubdtype(xyz.dtype, np.floating) else np.float64
    mats = np.broadcast_to(np.eye(4, dtype=dtype), (B, 4, 4)).copy()
    mats[:, :3, 3] = xyz
    return mats


def _matrix_to_xyz(mats: np.ndarray) -> np.ndarray:
    """
    args:
        mats: (B, 4, 4) array of SE3 transformation matrices
    returns:
        (B, 3) np.array of [[x, y, z]]
    """
    if mats.ndim != 3 or mats.shape[-2:] != (4, 4):
        raise ValueError(f"Expected (B, 4, 4) array, got shape {mats.shape}")
    mats = np.asarray(mats)
    dtype = mats.dtype if np.issubdtype(mats.dtype, np.floating) else np.float64
    return mats[:, :3, 3].astype(dtype, copy=False)


# ---------------------------------------------------------------------------
# Continuous 6D rotation representation (Zhou et al. / 6DRepNet).
# Ported from the pi-6d branch so train / eval / deploy agree bit-for-bit.
# ---------------------------------------------------------------------------


def _matrix_to_xyz6d(mats: np.ndarray) -> np.ndarray:
    """Continuous 6D rotation representation (Zhou et al. / 6DRepNet).

    Takes the first two columns of each rotation matrix and prepends the
    translation:

    args:
        mats: (B, 4, 4) array of SE3 transformation matrices
    returns:
        (B, 9) np.array of [[x, y, z, c1x, c1y, c1z, c2x, c2y, c2z]] where
        c1 / c2 are the first / second columns of the rotation matrix.
    """
    if mats.ndim != 3 or mats.shape[-2:] != (4, 4):
        raise ValueError(f"Expected (B, 4, 4) array, got shape {mats.shape}")

    mats = np.asarray(mats)
    dtype = mats.dtype if np.issubdtype(mats.dtype, np.floating) else np.float64

    xyz = mats[:, :3, 3]
    c1 = mats[:, :3, 0]
    c2 = mats[:, :3, 1]

    return np.concatenate([xyz, c1, c2], axis=-1).astype(dtype, copy=False)


def _xyz6d_to_matrix(xyz6d: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_matrix_to_xyz6d`.

    Reconstructs a proper rotation matrix from the first two (possibly
    non-orthonormal) columns via Gram-Schmidt (``eps = 1e-8`` floor, column
    order, ``c3 = c1 x c2``).

    args:
        xyz6d: (B, 9) np.array of [[x, y, z, c1(3), c2(3)]]
    returns:
        (B, 4, 4) array of SE3 transformation matrices
    """
    if xyz6d.ndim != 2 or xyz6d.shape[-1] != 9:
        raise ValueError(f"Expected (B, 9) array, got shape {xyz6d.shape}")

    B = xyz6d.shape[0]
    dtype = xyz6d.dtype if np.issubdtype(xyz6d.dtype, np.floating) else np.float64
    xyz6d = xyz6d.astype(dtype, copy=False)

    eps = 1e-8
    xyz = xyz6d[:, :3]
    c1 = xyz6d[:, 3:6]
    c2 = xyz6d[:, 6:9]

    # Gram-Schmidt (matches torch _reconstruct_R_from_cols).
    c1n = c1 / np.clip(np.linalg.norm(c1, axis=-1, keepdims=True), eps, None)
    proj = np.sum(c2 * c1n, axis=-1, keepdims=True) * c1n
    c2o = c2 - proj
    c2n = c2o / np.clip(np.linalg.norm(c2o, axis=-1, keepdims=True), eps, None)
    c3n = np.cross(c1n, c2n)

    mats = np.broadcast_to(np.eye(4, dtype=dtype), (B, 4, 4)).copy()
    mats[:, :3, 0] = c1n
    mats[:, :3, 1] = c2n
    mats[:, :3, 2] = c3n
    mats[:, :3, 3] = xyz
    return mats


def _split_action_pose(actions):
    # 14D layout: [L xyz ypr g, R xyz ypr g]
    # 12D layout: [L xyz ypr, R xyz ypr]
    if actions.shape[-1] == 14:
        left_xyz = actions[..., :3]
        left_ypr = actions[..., 3:6]
        right_xyz = actions[..., 7:10]
        right_ypr = actions[..., 10:13]
    elif actions.shape[-1] == 12:
        left_xyz = actions[..., :3]
        left_ypr = actions[..., 3:6]
        right_xyz = actions[..., 6:9]
        right_ypr = actions[..., 9:12]
    else:
        raise ValueError(f"Unsupported action dim {actions.shape[-1]}")
    return left_xyz, left_ypr, right_xyz, right_ypr


def _split_keypoints(keypoints, wrist_in_data: bool = False, is_quat: bool = True):
    if wrist_in_data:
        xyz_size = 3
        if is_quat:
            angle_size = 4
        else:
            angle_size = 3
        left_xyz_index = xyz_size
        left_angle_index = left_xyz_index + angle_size
        left_keypoints_index = left_angle_index + 21 * 3
        right_xyz_index = left_keypoints_index + xyz_size
        right_angle_index = right_xyz_index + angle_size
        right_keypoints_index = right_angle_index + 21 * 3
        return (
            keypoints[..., :left_xyz_index],
            keypoints[..., left_xyz_index:left_angle_index],
            keypoints[..., left_angle_index:left_keypoints_index],
            keypoints[..., left_keypoints_index:right_xyz_index],
            keypoints[..., right_xyz_index:right_angle_index],
            keypoints[..., right_angle_index:right_keypoints_index],
        )
    else:
        xyz_size = 3
        left_keypoints = keypoints[..., :63]
        right_keypoints = keypoints[..., 63:]
        return left_keypoints, right_keypoints
