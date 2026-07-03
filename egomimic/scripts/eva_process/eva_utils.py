import datetime
from pathlib import Path

import h5py
import numpy as np
import torch

from egomimic.rldb.embodiment.embodiment import EMBODIMENT

GRIPPER_NORMALIZE_CUTOFF = datetime.datetime(2026, 4, 8, tzinfo=datetime.timezone.utc)

DATASET_KEY_MAPPINGS = {
    "joint_positions": "joint_positions",
    "front_img_1": "front_img_1",
    "right_wrist_img": "right_wrist_img",
    "left_wrist_img": "left_wrist_img",
}

# Keys in episode_feats whose zero xyzypr frames should be filled.
ACTION_KEYS = {"cmd_eepose", "obs_eepose", "cmd_joints", "obs_joints"}


def resolve_timestamp_ms(episode_path) -> int:
    """Epoch-ms timestamp of an episode: the integer filename stem (EVA
    convention), else the ``timestamp_ms`` hdf5 attr (YAM collector).

    Raises instead of guessing: a fabricated small integer would both mis-hash
    the episode and (if it parses to a pre-cutoff date) silently trigger the
    legacy EVA gripper renormalization below.
    """
    stem = Path(episode_path).stem
    try:
        return int(stem)
    except ValueError:
        pass
    with h5py.File(episode_path, "r") as f:
        ts = f.attrs.get("timestamp_ms")
    if ts is not None:
        return int(ts)
    raise ValueError(
        f"Cannot determine the epoch-ms timestamp for '{episode_path}': the "
        f"filename stem '{stem}' is not an integer and the file has no "
        f"'timestamp_ms' attr. Re-collect with the current collect_yam_demo.py "
        f"(which stamps both), or add the attr / rename to the TRUE epoch-ms of "
        f"the recording — do NOT invent a number."
    )


class EvaHD5Extractor:
    @staticmethod
    def process_episode(episode_path, arm, robot="eva"):
        """
        Extracts all feature keys from a given episode and returns as a dictionary
        Parameters
        ----------
        episode_path : str or Path
            Path to the HDF5 file containing the episode data.
        arm : str
            String for which arm to add data for
        robot : str
            Which robot recorded the episode ("eva" or "yam"). Selects the
            metadata.embodiment id and gates the legacy EVA-only gripper
            renormalization.
        Returns
        -------
        episode_feats : dict
            dictionary mapping keys in the episode to episode features
            {
                {action_key} :
                observations :
                    images.{camera_key} :
                    state.{state_key} :
            }

            #TODO: Add metadata to be a nested dict

        """
        episode_feats = dict()

        timestamp_ms = resolve_timestamp_ms(episode_path)
        with h5py.File(episode_path, "r") as episode:
            for camera in EvaHD5Extractor.get_cameras(episode):
                images = (
                    torch.from_numpy(episode["observations"]["images"][camera][:])
                    .permute(0, 3, 1, 2)
                    .float()
                )

                images = images.byte().numpy()

                mapped_key = DATASET_KEY_MAPPINGS.get(camera, camera)
                episode_feats[f"images.{mapped_key}"] = images

            for state in EvaHD5Extractor.get_obs_state(episode):
                mapped_key = DATASET_KEY_MAPPINGS.get(state, state)
                episode_feats[f"obs_{mapped_key}"] = episode["observations"][state][:]

            for state in EvaHD5Extractor.get_cmd_state(episode):
                mapped_key = DATASET_KEY_MAPPINGS.get(state, state)
                episode_feats[f"cmd_{mapped_key}"] = episode["actions"][state][:]

            episode_dt = datetime.datetime.fromtimestamp(
                timestamp_ms / 1000.0, tz=datetime.timezone.utc
            )
            for key in ACTION_KEYS:
                if key in episode_feats:
                    episode_feats[key] = EvaHD5Extractor.clean_zero_data(
                        episode_feats[key]
                    )
                    # Legacy fix for early EVA recordings whose grippers were
                    # not yet [0,1]-normalized. EVA-only: YAM grippers have
                    # always been absolute [0,1], and a per-episode min-max
                    # would silently rescale them inconsistently.
                    if robot == "eva" and episode_dt < GRIPPER_NORMALIZE_CUTOFF:
                        episode_feats[key] = EvaHD5Extractor.normalize_grippers(
                            episode_feats[key]
                        )

            num_timesteps = episode_feats["obs_eepose"].shape[0]
            arm_suffix = {"right": "RIGHT_ARM", "left": "LEFT_ARM"}.get(
                arm, "BIMANUAL"
            )
            value = EMBODIMENT[f"{robot.upper()}_{arm_suffix}"].value

            episode_feats["metadata.embodiment"] = np.full(
                (num_timesteps, 1), value, dtype=np.int32
            )

        return episode_feats

    @staticmethod
    def get_cameras(hdf5_data: h5py.File):
        """
        Extracts the list of RGB camera keys from the given HDF5 data.
        Parameters
        ----------
        hdf5_data : h5py.File
            The HDF5 file object containing the dataset.
        Returns
        -------
        list of str
            A list of keys corresponding to RGB cameras in the dataset.
        """

        rgb_cameras = [
            key for key in hdf5_data["/observations/images"] if "depth" not in key
        ]
        return rgb_cameras

    @staticmethod
    def get_obs_state(hdf5_data: h5py.File):
        """
        Extracts the list of RGB camera keys from the given HDF5 data.
        Parameters
        ----------
        hdf5_data : h5py.File
            The HDF5 file object containing the dataset.
        Returns
        -------
        states : list of str
            A list of keys corresponding to states in the dataset.
        """

        states = [key for key in hdf5_data["/observations"] if "images" not in key]
        return states

    @staticmethod
    def get_cmd_state(hdf5_data: h5py.File):
        """
        Extracts the list of command state keys from the given HDF5 data.
        Parameters
        ----------
        hdf5_data : h5py.File
            The HDF5 file object containing the dataset.
        Returns
        -------
        cmd_states : list of str
        """
        states = [key for key in hdf5_data["/actions"]]
        return states

    @staticmethod
    def clean_zero_data(data: np.ndarray) -> np.ndarray:
        """
        Fill zero xyzypr frames in a (N, 14) action array per arm.

        Layout:
            [0:6]  left  xyzypr,  [6]  left  gripper
            [7:13] right xyzypr,  [13] right gripper

        For each arm independently: if all 6 xyzypr values at timestep t are
        zero, replace them with the latest preceding non-zero xyzypr. If there
        is no preceding non-zero value (start of episode), use the earliest
        following non-zero value instead.
        """
        data = data.copy()

        arm_pose_slices = [slice(0, 6), slice(7, 13)]  # left, right

        for pose_slice in arm_pose_slices:
            zero_mask = np.all(data[:, pose_slice] == 0, axis=1)  # (N,)

            if not np.any(zero_mask):
                continue

            nonzero_indices = np.where(~zero_mask)[0]
            if len(nonzero_indices) == 0:
                continue  # entire arm is zero, nothing to fill from

            for t in np.where(zero_mask)[0]:
                before = nonzero_indices[nonzero_indices < t]
                if len(before) > 0:
                    src = before[-1]
                else:
                    after = nonzero_indices[nonzero_indices > t]
                    src = after[0]  # guaranteed: nonzero_indices is non-empty
                data[t, pose_slice] = data[src, pose_slice]

        return data

    @staticmethod
    def normalize_grippers(data: np.ndarray) -> np.ndarray:
        """
        Normalize the gripper data to be between 0 and 1.
        """
        left_gripper = data[:, 6]
        right_gripper = data[:, 13]
        if left_gripper.max() - left_gripper.min() < 0.5:
            # then the gripper is not normalized yet if its in this range
            left_gripper = (left_gripper - left_gripper.min()) / (
                left_gripper.max() - left_gripper.min()
            )
        if right_gripper.max() - right_gripper.min() < 0.5:
            # then the gripper is not normalized yet if its in this range
            right_gripper = (right_gripper - right_gripper.min()) / (
                right_gripper.max() - right_gripper.min()
            )
        data[:, 6] = left_gripper
        data[:, 13] = right_gripper
        return data
