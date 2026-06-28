#!/usr/bin/env python3
"""Collect bimanual YAM demonstrations via leader-follower teleoperation.

This replaces the VR/Oculus-based ``collect_demo.py`` flow. Teleop is driven by
YAM teaching-handle *leader* arms; the *follower* arms (wrapped by the new
``YAMInterface``) mirror the leaders and are what we record. RealSense cameras
are captured in background threads. Episodes are saved as EgoVerse-format HDF5.

Workflow (per the teaching-handle sync button):
    * Press-and-hold the sync button(s) once  -> engage teleop + start recording
      a fresh episode. The follower slow-moves to the leader pose, then tracks it.
    * Press the sync button(s) again           -> disengage + save the episode
      (auto-increments the episode id). An empty or discarded episode is dropped.
    * If ``--episode-length`` is set, recording auto-stops/saves at that many steps.

Keyboard (type the letter + Enter in the launching terminal):
    d  discard the in-progress episode (won't be saved on disengage)
    h  send followers home (only while disengaged)
    q  quit

Example (bimanual, teaching-handle leaders + linear_4310 followers):
    python collect_yam_demo.py --arms both \
        --left-leader-can can_leader_l   --right-leader-can can_leader_r \
        --left-follower-can can_follower_l --right-follower-can can_follower_r \
        --camera-name 420222073106=front_img_1 \
        --camera-name 353322270967=left_wrist_img \
        --camera-name 323622270294=right_wrist_img \
        --demo-dir ./demos --episode-length 600
"""

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

# YAMInterface lives next to this file; importing it also puts the vendored
# i2rt submodule on sys.path (see yam_interface.py).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yam_interface import YAMInterface  # noqa: E402

from i2rt.robots.get_robot import get_yam_robot  # noqa: E402
from i2rt.robots.motor_chain_robot import MotorChainRobot  # noqa: E402
from i2rt.robots.utils import ArmType, GripperType  # noqa: E402

# Camera capture is shared with the rollout obs path (yam_interface.get_obs).
# YamCameraRig = Atlas (rectified) front_img_1 + RealSense D405 wrists.
from yam_cameras import (  # noqa: E402
    DEFAULT_CAMERA_NAMES,
    YamCameraRig,
    parse_camera_names,
)


# ------------------------- Configuration -------------------------

DEFAULT_FREQUENCY = 30.0  # Hz
DEFAULT_DEMO_DIR = "./demos"
SYNC_BUTTON_THRESHOLD = 0.5
SLOW_MOVE_STEPS = 100
SLOW_MOVE_DT = 0.01
STUCK_FRAME_THRESHOLD = 100  # consecutive identical frames => abort save

# Default CAN channels (match the pranav_code bimanual setup).
DEFAULT_LEADER_CHANNELS = {"left": "can_leader_l", "right": "can_leader_r"}
DEFAULT_FOLLOWER_CHANNELS = {"left": "can_follower_l", "right": "can_follower_r"}

# Per-machine camera serial->name aliases live in yam_cameras.DEFAULT_CAMERA_NAMES
# (imported above), shared with the rollout obs path.

_ARM_OFFSET = {"left": 0, "right": 7}

# Set during shutdown so the thread excepthook can distinguish a real error
# from harmless teardown noise.
_SHUTTING_DOWN = threading.Event()


def _quiet_thread_excepthook(args):
    """Swallow harmless CAN-socket teardown errors from SDK background threads.

    The i2rt DMChainCanInterface.close() sets running=False then immediately
    shuts the CAN socket WITHOUT joining its control thread, so an in-flight
    bus.send() can hit a closed socket (fd -1) and raise ValueError/OSError in
    that background thread. During shutdown that's expected; suppress it. Outside
    shutdown, defer to the default handler so real errors still surface.
    """
    if _SHUTTING_DOWN.is_set() and issubclass(args.exc_type, (ValueError, OSError)):
        return
    threading.__excepthook__(args)


# ------------------------- Leader (teaching handle) -------------------------


class YAMLeaderRobot:
    """Thin wrapper over a teaching-handle MotorChainRobot.

    Adapted from i2rt/examples/pranav_code. Exposes the arm joint positions
    plus the encoder-derived gripper command and the handle's button states.
    """

    def __init__(self, robot: MotorChainRobot):
        self._robot = robot
        self._motor_chain = robot.motor_chain

    @property
    def kp(self) -> np.ndarray:
        return self._robot._kp[:6].copy()

    @property
    def kd(self) -> np.ndarray:
        return self._robot._kd[:6].copy()

    def go_home(self, kp: np.ndarray, kd: np.ndarray, duration: float = 2.0) -> None:
        """Drive the (normally passive) teaching handle to the zero pose, then relax.

        Restores drivable PD gains, interpolates to zero, then returns the handle
        to zero-torque (back-drivable) mode. The operator should let go of the
        handle while this runs.
        """
        self.update_kp_kd(kp=np.asarray(kp, dtype=float), kd=np.asarray(kd, dtype=float))
        self._robot.move_joints(np.zeros(6), time_interval_s=duration)
        self._robot.zero_torque_mode()  # back to passive / back-drivable

    def get_info(self):
        """Return (qpos_with_gripper (7,), io_inputs).

        Gripper is normalized to [0, 1] (1 = open), matching the follower's
        gripper command convention.
        """
        qpos = self._robot.get_observations()["joint_pos"]
        encoder_obs = self._motor_chain.get_same_bus_device_states()
        time.sleep(0.01)  # per manufacturer spec for the encoder read
        gripper_cmd = 1.0 - encoder_obs[0].position
        qpos_with_gripper = np.concatenate([qpos, [gripper_cmd]])
        return qpos_with_gripper, encoder_obs[0].io_inputs

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        assert joint_pos.shape[0] == 6, "Leader command is arm-only (6 joints)"
        self._robot.command_joint_pos(joint_pos)

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        self._robot.update_kp_kd(kp, kd)

    def close(self) -> None:
        closer = getattr(self._robot, "close", None)
        if callable(closer):
            closer()


def make_leader(channel: str, gripper: GripperType, arm_type: ArmType) -> YAMLeaderRobot:
    robot = get_yam_robot(channel=channel, arm_type=arm_type, gripper_type=gripper)
    return YAMLeaderRobot(robot)


# ------------------------- Keyboard control -------------------------


class KeyboardController(threading.Thread):
    """Reads single-letter commands (letter + Enter) from stdin in the background."""

    def __init__(self):
        super().__init__(daemon=True)
        self.quit = False
        self.discard = threading.Event()
        self.home = threading.Event()

    def run(self):
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd == "q":
                self.quit = True
                print("[kbd] quit requested.")
            elif cmd == "d":
                self.discard.set()
                print("[kbd] discard current episode.")
            elif cmd == "h":
                self.home.set()
                print("[kbd] home requested.")


# ------------------------- Saving (EgoVerse HDF5 format) -------------------------


def fk_eepose14(robot_interface: YAMInterface, joints14: np.ndarray) -> np.ndarray:
    """FK a 14-dim joint vector into a 14-dim ee-pose (xyz + ZYX euler + gripper per arm)."""
    ee = np.zeros(14)
    # Only arms in robot_interface.arms are populated; absent arms stay zero.
    # Do NOT special-case all-zero joints — YAM FK(0) is NOT the origin
    # (fk(zeros) -> xyz ~= [0.113, 0.004, 0.172]), so a zero-guard would
    # fabricate a wrong ee_pose, inconsistent with get_obs / ARXInterface.
    for arm in robot_interface.arms:
        off = _ARM_OFFSET[arm]
        q = joints14[off : off + 6]
        T = robot_interface.kinematics[arm].fk(q)
        ypr = R.from_matrix(T[:3, :3]).as_euler("ZYX", degrees=False)
        ee[off : off + 6] = np.concatenate([T[:3, 3], ypr])
        ee[off + 6] = joints14[off + 6]
    return ee


def _has_stuck_frames(obs_list, cam_name, threshold=STUCK_FRAME_THRESHOLD) -> bool:
    consecutive, prev = 0, None
    for obs in obs_list:
        img = obs.get(cam_name)
        if img is None:
            prev, consecutive = None, 0
            continue
        checksum = int(img.sum())
        if prev is not None and checksum == prev:
            consecutive += 1
            if consecutive >= threshold:
                return True
        else:
            consecutive = 0
        prev = checksum
    return False


def save_demo(demo_data, demo_dir: Path, episode_id, camera_res, robot_interface,
              strict_cameras: bool = False) -> bool:
    """Write one episode to ``demo_<id>.hdf5`` in EgoVerse layout.

    demo_data keys (lists, one entry per step):
        obs                  -> {cam_name: BGR frame}
        robot_joint_actions  -> follower joints (14,)
        cmd_joint_actions    -> leader/commanded joints (14,)

    Args:
        strict_cameras: if True, abort (don't save) when any camera shows
            >STUCK_FRAME_THRESHOLD consecutive identical frames. If False
            (default), warn loudly but still save the episode.
    """
    n_steps = len(demo_data["robot_joint_actions"])
    if n_steps == 0:
        print("[save] No steps recorded; skipping.")
        return False

    stuck_cams = [c for c in camera_res if _has_stuck_frames(demo_data["obs"], c)]
    if stuck_cams:
        msg = (
            f">{STUCK_FRAME_THRESHOLD} consecutive identical frames on camera(s): "
            f"{', '.join(stuck_cams)} (likely a stalled/disconnected stream)."
        )
        if strict_cameras:
            print(f"[save] ABORT (--strict-cameras): {msg} Demo not saved.")
            return False
        print(f"[save] WARNING: {msg} Saving anyway.")

    filename = demo_dir / f"demo_{episode_id}.hdf5"
    print(f"[save] Writing {n_steps} steps to {filename}")

    robot_joints = np.asarray(demo_data["robot_joint_actions"])  # (T, 14)
    cmd_joints = np.asarray(demo_data["cmd_joint_actions"])      # (T, 14)
    obs_eepose = np.stack([fk_eepose14(robot_interface, j) for j in robot_joints])
    cmd_eepose = np.stack([fk_eepose14(robot_interface, j) for j in cmd_joints])

    t0 = time.time()
    # 32MB chunk cache: the front camera chunk alone is ~6.9MB (1920x1200x3),
    # so the default 2MB would thrash on every frame write.
    with h5py.File(str(filename), "w", rdcc_nbytes=1024**2 * 32) as root:
        root.attrs["sim"] = False
        obs = root.create_group("observations")
        images = obs.create_group("images")
        for cam_name, (H, W) in camera_res.items():
            images.create_dataset(
                cam_name, (n_steps, H, W, 3), dtype="uint8", chunks=(1, H, W, 3)
            )
        obs.create_dataset("joints", data=robot_joints, dtype="float64")
        obs.create_dataset("joint_positions", data=robot_joints, dtype="float64")
        obs.create_dataset("eepose", data=obs_eepose, dtype="float64")

        actions = root.create_group("actions")
        actions.create_dataset("joints", data=cmd_joints, dtype="float64")
        actions.create_dataset("eepose", data=cmd_eepose, dtype="float64")
        root.create_dataset("action", data=cmd_joints, dtype="float64")

        # Write images frame-by-frame, BGR -> RGB, to bound peak memory.
        for cam_name in camera_res:
            ds = images[cam_name]
            for idx, obs_step in enumerate(demo_data["obs"]):
                frame = obs_step.get(cam_name)
                if frame is not None:
                    ds[idx] = frame[..., ::-1]

    print(f"[save] Done in {time.time() - t0:.1f}s")
    return True


def reset_data(demo_data: dict):
    demo_data["obs"] = []
    demo_data["robot_joint_actions"] = []
    demo_data["cmd_joint_actions"] = []


# ------------------------- Teleop helpers -------------------------


def slow_move(robot_interface: YAMInterface, arm: str, target7: np.ndarray, current7: np.ndarray):
    """Linearly interpolate the follower from its current pose to the leader pose."""
    for i in range(SLOW_MOVE_STEPS + 1):
        alpha = i / SLOW_MOVE_STEPS
        cmd = alpha * target7 + (1 - alpha) * current7
        robot_interface.set_joints(cmd, arm)
        time.sleep(SLOW_MOVE_DT)


# ------------------------- Main collection loop -------------------------


def collect_yam_demo(
    arms,
    leader_channels,
    follower_channels,
    serial_to_name,
    frequency=DEFAULT_FREQUENCY,
    demo_dir=DEFAULT_DEMO_DIR,
    episode_id_start=0,
    episode_length=None,
    bilateral_kp=0.15,
    gripper=GripperType.LINEAR_4310,
    strict_cameras=False,
    debug_buttons=False,
):
    demo_dir = Path(demo_dir)
    demo_dir.mkdir(parents=True, exist_ok=True)
    dt = 1.0 / frequency

    # Quiet the harmless CAN-socket teardown errors raised by SDK background
    # threads during shutdown (see _quiet_thread_excepthook).
    threading.excepthook = _quiet_thread_excepthook

    # --- Followers (recorded arms) via the new interface --------------------
    print("[setup] Initializing follower arms ...")
    robot_interface = YAMInterface(
        arms=arms,
        channels={a: follower_channels[a] for a in arms},
        gripper_type=gripper,
        zero_gravity_mode=False,  # hold commanded poses during teleop
        camera_names=serial_to_name,  # interface opens & owns the cameras
    )

    # --- Leaders (teaching handles) -----------------------------------------
    print("[setup] Initializing leader teaching handles ...")
    leaders = {
        a: make_leader(leader_channels[a], GripperType.YAM_TEACHING_HANDLE, ArmType.YAM)
        for a in arms
    }
    leader_kp = {a: leaders[a].kp for a in arms}
    leader_kd = {a: leaders[a].kd for a in arms}

    # --- Cameras ------------------------------------------------------------
    # The YAMInterface already opened (and owns) the cameras above — it raised if
    # none were available. Wrap its recorders in a YamCameraRig VIEW for the live
    # FPS/stall monitor and per-step frame grab below; the view never closes them
    # (robot_interface.close() does).
    cameras = YamCameraRig(recorders=robot_interface.recorders)
    camera_res = cameras.camera_res

    # --- Keyboard + graceful SIGINT -----------------------------------------
    kbd = KeyboardController()
    kbd.start()

    # First Ctrl-C just asks the loop to stop so cleanup runs exactly once
    # (a raw KeyboardInterrupt mid-cleanup was crashing the realsense/CAN
    # C++ threads with a core dump).
    def _on_sigint(signum, frame):
        if not kbd.quit:
            print("\n[signal] Ctrl-C received; finishing up and shutting down ...")
        kbd.quit = True

    signal.signal(signal.SIGINT, _on_sigint)

    print(
        "\n[ready] Press the teaching-handle sync button(s) to engage teleop & "
        "record.\n        Keyboard: 'd' discard | 'h' home | 'q' quit.\n"
    )

    demo_data = {}
    reset_data(demo_data)
    episode_id = episode_id_start
    synchronized = False
    discard_episode = False
    printed_buttons = False
    last_cam_report = time.time()
    last_cam_counts = cameras.frame_counts() if cameras else {}

    try:
        while not kbd.quit:
            loop_start = time.time()

            # Read all leaders up front.
            leader_info = {a: leaders[a].get_info() for a in arms}
            leader_joints = {a: leader_info[a][0] for a in arms}  # (7,) incl gripper
            if debug_buttons and not printed_buttons:
                for a in arms:
                    print(f"[debug] {a} leader io_inputs = {list(leader_info[a][1])}")
                printed_buttons = True
            buttons_pressed = all(
                leader_info[a][1][0] > SYNC_BUTTON_THRESHOLD for a in arms
            )

            # --- Live camera FPS / stall monitor (every ~5s) -----------------
            if cameras is not None and time.time() - last_cam_report >= 5.0:
                now = time.time()
                counts = cameras.frame_counts()
                span = now - last_cam_report
                rates = {n: (counts[n] - last_cam_counts.get(n, 0)) / span for n in counts}
                stalled = [n for n, r in rates.items() if r < 1.0]
                summary = ", ".join(f"{n}:{rates[n]:.0f}fps" for n in rates)
                print(f"[cameras] {summary}")
                if stalled:
                    errs = cameras.error_info()
                    for n in stalled:
                        ec, le = errs.get(n, (0, None))
                        # error_count climbing => decode-path errors; flat with
                        # 0 fps => genuine USB/stream starvation.
                        detail = f"{ec} errors, last={le}" if ec else "no errors (USB/stream starvation)"
                        print(f"[cameras] WARNING: '{n}' stalled ({detail})")
                last_cam_report = now
                last_cam_counts = counts

            # Home request (only when disengaged).
            if kbd.home.is_set():
                kbd.home.clear()
                if not synchronized:
                    print("[home] Homing followers AND leaders — LET GO of the handles ...")
                    robot_interface.set_home()
                    for a in arms:
                        leaders[a].go_home(kp=leader_kp[a], kd=leader_kd[a])
                    print("[home] Done; leaders are passive (back-drivable) again.")
                else:
                    print("[home] Ignored: disengage teleop first.")

            # Discard request.
            if kbd.discard.is_set():
                kbd.discard.clear()
                discard_episode = True
                reset_data(demo_data)
                print("[episode] Marked for discard; buffer cleared.")

            # --- Sync button edge: toggle engagement --------------------------
            if buttons_pressed:
                if not synchronized:
                    # Engage: bilateral PD on leaders, slow-move followers to leaders.
                    print("[teleop] Engaging ...")
                    for a in arms:
                        leaders[a].update_kp_kd(
                            kp=leader_kp[a] * bilateral_kp, kd=np.zeros(6)
                        )
                        current7 = robot_interface.get_joints(a)
                        slow_move(robot_interface, a, leader_joints[a], current7)
                    synchronized = True
                    discard_episode = False
                    reset_data(demo_data)
                    print(f"[episode] Recording episode {episode_id} ...")
                else:
                    # Disengage: relax leaders, save (unless discarded).
                    print("[teleop] Disengaging ...")
                    for a in arms:
                        leaders[a].update_kp_kd(kp=np.zeros(6), kd=np.zeros(6))
                    synchronized = False
                    if not discard_episode:
                        if save_demo(demo_data, demo_dir, episode_id, camera_res,
                                     robot_interface, strict_cameras=strict_cameras):
                            episode_id += 1
                    reset_data(demo_data)

                # Debounce: wait for release.
                while all(
                    leaders[a].get_info()[1][0] > SYNC_BUTTON_THRESHOLD for a in arms
                ):
                    time.sleep(0.01)

            # --- Synchronized: follower tracks leader, record ----------------
            if synchronized:
                for a in arms:
                    robot_interface.set_joints(leader_joints[a], a)

                # Follower state for bilateral feedback + recording.
                follower_joints = {a: robot_interface.get_joints(a) for a in arms}
                for a in arms:
                    leaders[a].command_joint_pos(follower_joints[a][:6])

                robot_joint_action = np.zeros(14)
                cmd_joint_action = np.zeros(14)
                for a in arms:
                    off = _ARM_OFFSET[a]
                    robot_joint_action[off : off + 7] = follower_joints[a]
                    cmd_joint_action[off : off + 7] = leader_joints[a]

                obs_step = cameras.get_frames() if cameras else {}
                demo_data["obs"].append(obs_step)
                demo_data["robot_joint_actions"].append(robot_joint_action)
                demo_data["cmd_joint_actions"].append(cmd_joint_action)

                # Auto-stop at fixed episode length.
                if episode_length is not None and len(demo_data["obs"]) >= episode_length:
                    print(f"[episode] Reached {episode_length} steps.")
                    for a in arms:
                        leaders[a].update_kp_kd(kp=np.zeros(6), kd=np.zeros(6))
                    synchronized = False
                    if not discard_episode:
                        if save_demo(demo_data, demo_dir, episode_id, camera_res,
                                     robot_interface, strict_cameras=strict_cameras):
                            episode_id += 1
                    reset_data(demo_data)

            # Maintain loop rate.
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n[exit] KeyboardInterrupt; shutting down.")
    finally:
        # Best-effort, idempotent cleanup. Each step is guarded so a hang or a
        # second Ctrl-C in one component can't abort the rest (which previously
        # left C++ threads to core-dump).
        _SHUTTING_DOWN.set()
        print("[exit] Closing hardware ...")

        def _safe(label, fn):
            try:
                fn()
            except BaseException as e:
                print(f"[exit] {label} error: {e}")

        # Stop EVERY robot's control loops BEFORE any CAN socket is closed.
        # The SDK's close() shuts the socket without joining its control thread,
        # so an in-flight bus.send() on a torn-down socket raises ValueError
        # (fd -1) in a background thread. Draining first avoids that race.
        underlying = [robot_interface.controller[a] for a in arms]
        underlying += [leaders[a]._robot for a in arms]
        for r in underlying:
            try:
                r.motor_chain.running = False  # stop internal CAN control thread
                r._stop_event.set()            # stop MotorChainRobot server thread
            except BaseException:
                pass
        time.sleep(0.3)  # let those threads exit their loops before sockets close

        if cameras is not None:
            _safe("cameras", cameras.close)
        for a in arms:
            _safe(f"leader[{a}]", leaders[a].close)
        _safe("followers", robot_interface.close)
        print("[exit] Done.")


# ------------------------- CLI -------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Collect YAM demos via leader-follower teleoperation."
    )
    parser.add_argument(
        "--arms", choices=["left", "right", "both"], default="both",
        help="Which arm(s) to teleoperate and record.",
    )
    parser.add_argument("--left-leader-can", default=DEFAULT_LEADER_CHANNELS["left"])
    parser.add_argument("--right-leader-can", default=DEFAULT_LEADER_CHANNELS["right"])
    parser.add_argument("--left-follower-can", default=DEFAULT_FOLLOWER_CHANNELS["left"])
    parser.add_argument("--right-follower-can", default=DEFAULT_FOLLOWER_CHANNELS["right"])
    parser.add_argument(
        "--camera-name", action="append", metavar="SERIAL=NAME",
        help="Map a RealSense serial to a friendly name (repeatable). Overrides/"
             "extends the hardcoded DEFAULT_CAMERA_NAMES; normally you don't need "
             "this. e.g. --camera-name 420222073106=front_img_1",
    )
    parser.add_argument(
        "--strict-cameras", action="store_true",
        help="Abort (don't save) an episode if any camera shows >100 consecutive "
             "identical frames. Default: warn but save anyway.",
    )
    parser.add_argument(
        "--debug-buttons", action="store_true",
        help="Print each leader's raw io_inputs once, to identify the sync-button index.",
    )
    parser.add_argument("--frequency", type=float, default=DEFAULT_FREQUENCY)
    parser.add_argument("--demo-dir", default=DEFAULT_DEMO_DIR)
    parser.add_argument("--episode-id-start", type=int, default=0)
    parser.add_argument(
        "--episode-length", type=int, default=None,
        help="If set, auto-save and increment after this many recorded steps.",
    )
    parser.add_argument(
        "--bilateral-kp", type=float, default=0.15,
        help="Leader force-feedback gain (fraction of the leader's control kp).",
    )
    args = parser.parse_args()

    if args.arms == "both":
        arms = ["left", "right"]
    else:
        arms = [args.arms]

    leader_channels = {"left": args.left_leader_can, "right": args.right_leader_can}
    follower_channels = {"left": args.left_follower_can, "right": args.right_follower_can}
    # Start from the hardcoded per-machine aliases, then let CLI flags
    # override/extend them.
    serial_to_name = dict(DEFAULT_CAMERA_NAMES)
    serial_to_name.update(parse_camera_names(args.camera_name))

    collect_yam_demo(
        arms=arms,
        leader_channels=leader_channels,
        follower_channels=follower_channels,
        serial_to_name=serial_to_name,
        frequency=args.frequency,
        demo_dir=args.demo_dir,
        episode_id_start=args.episode_id_start,
        episode_length=args.episode_length,
        bilateral_kp=args.bilateral_kp,
        strict_cameras=args.strict_cameras,
        debug_buttons=args.debug_buttons,
    )


if __name__ == "__main__":
    main()
