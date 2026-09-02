#!/usr/bin/env python3
"""Collect bimanual YAM + Aero Hand + JQ-glove demonstrations via teleoperation.

EgoVerse's ``collect_yam_demo.py`` extended for the dexterous rig: the follower
arms carry TetherIA Aero Hands (USB serial, driven by the teaching-handle
trigger through a grasp synergy) and wear JQ Precision tactile gloves (162
taxels @ 200 Hz). The workflow and HDF5 layout are IDENTICAL to EgoVerse —
existing pipelines keep working — with extra datasets added per episode.

Workflow (unchanged from EgoVerse):
    * Press the teaching-handle SYNC button(s) -> engage teleop + start recording.
    * Press again                              -> disengage + save the episode.
    * Keyboard: 'd' discard | 'h' home | 'q' quit.
    * The trigger continuously drives the Aero Hand along the selected grasp
      synergy the whole time the leader is up (engaged or not).

HDF5 additions on top of the EgoVerse layout (T = steps; absent arms are zero):
    observations/tactile/{left,right}        (T, 256) uint8  raw glove array
    observations/tactile/{left,right}_quat   (T, 4)   float  glove IMU (w,x,y,z)
    observations/hand/{left,right}_actuations (T, 7)  float  measured actuator deg (NaN row when a read fails)
    actions/hand/{left,right}                 (T, 7)  float  commanded hand pose deg
    actions/hand/{left,right}_frac            (T,)    float  commanded grasp fraction 0..1
    The standard 14-dim joint vectors keep the EgoVerse convention, with the
    per-arm gripper slot carrying the commanded grasp fraction.

Run on the rig, inside the EgoVerse venv (needs h5py/scipy/pyrealsense2 there,
plus aero-open-sdk + pyserial):

    ~/EgoVerse/emimic/bin/python collect_dex_demo.py --arms both --demo-dir ./demos

NOTE: mutually exclusive with deploy.py — both want the arms, hands, and glove
serial ports. Stop the deployment before collecting.
"""

import argparse
import glob as globlib
import os
import signal
import sys
import threading
import time
from pathlib import Path

import h5py
import numpy as np

# Reuse EgoVerse's YAM collection machinery (this file lives next to it).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_yam_demo import (  # noqa: E402
    DEFAULT_FOLLOWER_CHANNELS,
    DEFAULT_LEADER_CHANNELS,
    KeyboardController,
    SYNC_BUTTON_THRESHOLD,
    YAMLeaderRobot,
    _quiet_thread_excepthook,
    _ARM_OFFSET,
    _SHUTTING_DOWN,
    fk_eepose14,
    make_leader,
)
from yam_interface import YAMInterface  # noqa: E402
from yam_cameras import DEFAULT_CAMERA_NAMES, YamCameraRig, parse_camera_names  # noqa: E402

from i2rt.robots.utils import ArmType, GripperType  # noqa: E402

import serial as pyserial  # noqa: E402
from serial.tools import list_ports  # noqa: E402
from jq_protocol import JqParser  # noqa: E402

from aero_open_sdk.aero_hand import AeroHand  # noqa: E402

# ------------------------- Rig configuration -------------------------

DEFAULT_FREQUENCY = 30.0
DEFAULT_DEMO_DIR = "./demos"

# Aero Hand serial ports per arm (by-id paths are stable across reboots).
DEFAULT_HAND_PORTS = {
    "right": "/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_1C:DB:D4:75:E7:64-if00",
    "left": "/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_1C:DB:D4:76:1B:5C-if00",
}

# JQ glove cables: CH343 bridges (VID 0x1A86). Auto-detected by default —
# NEVER match 0x303A here (that's the Aero Hands' ESP32s).
GLOVE_VID = 0x1A86
GLOVE_BAUD = 921_600

# Grasp synergies (deg), from the deploy stack; trigger interpolates open->close.
# Joint order: [thumb_abd, thumb_flex, thumb_curl, index, middle, ring, pinky].
GRASP_MODES = {
    "power": (np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
              np.array([100.0, 55.0, 30.0, 60.0, 60.0, 60.0, 60.0])),
    "pinch": (np.array([75.0, 10.0, 0.0, 0.0, 90.0, 90.0, 90.0]),
              np.array([75.0, 28.0, 32.0, 55.0, 90.0, 90.0, 90.0])),
    "tripod": (np.array([80.0, 10.0, 0.0, 0.0, 0.0, 90.0, 90.0]),
               np.array([80.0, 36.0, 28.0, 52.0, 52.0, 90.0, 90.0])),
    "hook": (np.array([0.0, 0.0, 0.0, 20.0, 20.0, 20.0, 20.0]),
             np.array([0.0, 0.0, 0.0, 90.0, 90.0, 90.0, 90.0])),
}

HAND_SEND_MIN_DT = 0.05     # rate-limit hand serial commands to <= 20 Hz
HAND_SEND_DEADBAND = 0.02   # only send when the fraction moved this much


# ------------------------- Aero Hand wrapper -------------------------


class DexHand:
    """One Aero Hand: trigger->synergy command path + state reads for recording."""

    def __init__(self, arm: str, port: str, grasp: str):
        self.arm = arm
        self.hand = AeroHand(port=port)
        self.g_open, self.g_close = GRASP_MODES[grasp]
        self.trig_rest = 1.0        # calibrated at startup (released reading)
        self.last_frac = -1.0
        self.last_cmd_pose = self.g_open.copy()
        self.last_send = 0.0
        # First read after connect is often a framing glitch — burn one.
        try:
            self.hand.get_actuations()
        except Exception:
            pass

    def calibrate_rest(self, rest_value: float) -> None:
        # Anchor so the released trigger maps to exactly 0 (hand open).
        self.trig_rest = max(0.2, float(rest_value))
        print(f"[hand:{self.arm}] trigger rest calibrated to {self.trig_rest:.3f}")

    def frac_from_trigger(self, gripper_cmd: float) -> float:
        # Handle encoder: released ~= rest anchor, decreases toward 0 when squeezed.
        return float(np.clip((self.trig_rest - gripper_cmd) / self.trig_rest, 0.0, 1.0))

    def drive(self, gripper_cmd: float, now: float) -> float:
        """Map trigger to the synergy and (rate-limited) command the hand."""
        frac = self.frac_from_trigger(gripper_cmd)
        if abs(frac - self.last_frac) > HAND_SEND_DEADBAND and now - self.last_send > HAND_SEND_MIN_DT:
            pose = self.g_open + frac * (self.g_close - self.g_open)
            try:
                self.hand.set_joint_positions(pose.tolist())
                self.last_frac = frac
                self.last_cmd_pose = pose
                self.last_send = now
            except Exception as e:
                print(f"[hand:{self.arm}] command error: {e}")
        return frac

    def read_actuations(self) -> np.ndarray:
        try:
            act = self.hand.get_actuations()
            if act is not None:
                return np.asarray(act, dtype=np.float64)
        except Exception:
            pass
        return np.full(7, np.nan)

    def open_and_close(self) -> None:
        try:
            self.hand.set_joint_positions(self.g_open.tolist())
            time.sleep(0.3)
            self.hand.close()
        except Exception:
            pass


# ------------------------- JQ glove reader -------------------------


class GloveBus:
    """Single reader per glove serial port; keeps the latest frame per hand."""

    def __init__(self, explicit_ports):
        self.latest = {}     # hand -> (pressure bytes(256), quat tuple)
        self.counts = {"left": 0, "right": 0}
        ports = explicit_ports or self._autodetect()
        if not ports:
            print("[glove] WARNING: no glove cables found (VID 0x1A86) — tactile "
                  "datasets will be zeros.")
        for device in ports:
            threading.Thread(target=self._reader, args=(device,), daemon=True).start()

    @staticmethod
    def _autodetect():
        found = [p.device for p in list_ports.comports() if p.vid == GLOVE_VID]
        # Prefer stable by-id paths when available.
        by_id = sorted(globlib.glob("/dev/serial/by-id/usb-1a86_*"))
        return by_id or sorted(found)

    def _reader(self, device: str) -> None:
        while not _SHUTTING_DOWN.is_set():
            try:
                port = pyserial.Serial(device, GLOVE_BAUD, timeout=0.05)
                port.reset_input_buffer()
            except Exception as e:
                print(f"[glove] open failed on {device}: {e} — retrying in 3 s")
                time.sleep(3)
                continue
            print(f"[glove] streaming from {device}")
            parser = JqParser()
            try:
                while not _SHUTTING_DOWN.is_set():
                    chunk = port.read(max(292, port.in_waiting))
                    for frame in parser.push(chunk):
                        self.latest[frame.hand] = (frame.pressure, frame.quat)
                        self.counts[frame.hand] += 1
            except Exception as e:
                print(f"[glove] read error on {device}: {e} — reopening")
            finally:
                try:
                    port.close()
                except Exception:
                    pass

    def snapshot(self, hand: str):
        """(pressure uint8 (256,), quat (4,)) — zeros if this hand never arrived."""
        entry = self.latest.get(hand)
        if entry is None:
            return np.zeros(256, dtype=np.uint8), np.zeros(4)
        pressure, quat = entry
        return np.frombuffer(pressure, dtype=np.uint8).copy(), np.asarray(quat)


# ------------------------- Saving (EgoVerse layout + dex extras) ---------------


def reset_data(demo_data: dict) -> None:
    demo_data["obs"] = []
    demo_data["robot_joint_actions"] = []
    demo_data["cmd_joint_actions"] = []
    demo_data["tactile"] = []          # {hand: (256,) uint8}
    demo_data["tactile_quat"] = []     # {hand: (4,)}
    demo_data["hand_actuations"] = []  # {arm: (7,) or NaN}
    demo_data["hand_cmd_pose"] = []    # {arm: (7,)}
    demo_data["hand_cmd_frac"] = []    # {arm: float}


def save_demo(demo_data, demo_dir: Path, episode_id, camera_res, robot_interface) -> bool:
    """EgoVerse HDF5 layout + tactile/hand groups."""
    n_steps = len(demo_data["robot_joint_actions"])
    if n_steps == 0:
        print("[save] No steps recorded; skipping.")
        return False

    filename = demo_dir / f"demo_{episode_id}.hdf5"
    print(f"[save] Writing {n_steps} steps to {filename}")
    robot_joints = np.asarray(demo_data["robot_joint_actions"])
    cmd_joints = np.asarray(demo_data["cmd_joint_actions"])
    obs_eepose = np.stack([fk_eepose14(robot_interface, j) for j in robot_joints])
    cmd_eepose = np.stack([fk_eepose14(robot_interface, j) for j in cmd_joints])

    t0 = time.time()
    with h5py.File(str(filename), "w", rdcc_nbytes=1024**2 * 32) as root:
        root.attrs["sim"] = False
        obs = root.create_group("observations")
        images = obs.create_group("images")
        for cam_name, (H, W) in camera_res.items():
            images.create_dataset(cam_name, (n_steps, H, W, 3), dtype="uint8",
                                  chunks=(1, H, W, 3))
        obs.create_dataset("joints", data=robot_joints, dtype="float64")
        obs.create_dataset("joint_positions", data=robot_joints, dtype="float64")
        obs.create_dataset("eepose", data=obs_eepose, dtype="float64")

        actions = root.create_group("actions")
        actions.create_dataset("joints", data=cmd_joints, dtype="float64")
        actions.create_dataset("eepose", data=cmd_eepose, dtype="float64")
        root.create_dataset("action", data=cmd_joints, dtype="float64")

        # ---- dex extras -----------------------------------------------------
        tact = obs.create_group("tactile")
        hand_obs = obs.create_group("hand")
        hand_act = actions.create_group("hand")
        for side in ("left", "right"):
            tact.create_dataset(side, data=np.stack(
                [step[side] for step in demo_data["tactile"]]), dtype="uint8")
            tact.create_dataset(f"{side}_quat", data=np.stack(
                [step[side] for step in demo_data["tactile_quat"]]), dtype="float64")
            hand_obs.create_dataset(f"{side}_actuations", data=np.stack(
                [step.get(side, np.full(7, np.nan)) for step in demo_data["hand_actuations"]]),
                dtype="float64")
            hand_act.create_dataset(side, data=np.stack(
                [step.get(side, np.zeros(7)) for step in demo_data["hand_cmd_pose"]]),
                dtype="float64")
            hand_act.create_dataset(f"{side}_frac", data=np.asarray(
                [step.get(side, 0.0) for step in demo_data["hand_cmd_frac"]]),
                dtype="float64")

        for cam_name in camera_res:
            ds = images[cam_name]
            for idx, obs_step in enumerate(demo_data["obs"]):
                frame = obs_step.get(cam_name)
                if frame is not None:
                    ds[idx] = frame[..., ::-1]

    print(f"[save] Done in {time.time() - t0:.1f}s")
    return True


# ------------------------- Teleop helpers -------------------------


def command_arm(robot_interface: YAMInterface, arm: str, cmd7: np.ndarray) -> None:
    """Send only the joints the follower actually has (no_gripper follower = 6)."""
    ctrl = robot_interface.controller[arm]
    ctrl.command_joint_pos(np.asarray(cmd7)[: ctrl.num_dofs()])


def slow_move(robot_interface, arm, target7, current7, steps=100, dt=0.01):
    for i in range(steps + 1):
        alpha = i / steps
        command_arm(robot_interface, arm, alpha * target7 + (1 - alpha) * current7)
        time.sleep(dt)


# ------------------------- Main collection loop -------------------------


def collect_dex_demo(arms, leader_channels, follower_channels, hand_ports, glove_ports,
                     serial_to_name, grasp="power", frequency=DEFAULT_FREQUENCY,
                     demo_dir=DEFAULT_DEMO_DIR, episode_id_start=0, episode_length=None,
                     bilateral_kp=0.15):
    demo_dir = Path(demo_dir)
    demo_dir.mkdir(parents=True, exist_ok=True)
    dt = 1.0 / frequency
    threading.excepthook = _quiet_thread_excepthook

    # --- Followers (no gripper motor: Aero Hands are mounted) ----------------
    print("[setup] Initializing follower arms (no_gripper — Aero Hands mounted) ...")
    robot_interface = YAMInterface(
        arms=arms,
        channels={a: follower_channels[a] for a in arms},
        gripper_type=GripperType.NO_GRIPPER,
        zero_gravity_mode=False,
        camera_names=serial_to_name,
    )

    # --- Leaders --------------------------------------------------------------
    print("[setup] Initializing leader teaching handles ...")
    leaders = {a: make_leader(leader_channels[a], GripperType.YAM_TEACHING_HANDLE, ArmType.YAM)
               for a in arms}
    leader_kp = {a: leaders[a].kp for a in arms}
    leader_kd = {a: leaders[a].kd for a in arms}

    # --- Aero Hands -----------------------------------------------------------
    print("[setup] Connecting Aero Hands ...")
    hands = {a: DexHand(a, hand_ports[a], grasp) for a in arms}
    # Calibrate each trigger's rest point from ~0.5 s of released readings.
    for a in arms:
        samples = [leaders[a].get_info()[0][6] for _ in range(10)]
        hands[a].calibrate_rest(float(np.median(samples)))

    # --- JQ gloves ------------------------------------------------------------
    gloves = GloveBus(glove_ports)

    # --- Cameras (opened & owned by YAMInterface) -----------------------------
    cameras = YamCameraRig(recorders=robot_interface.recorders)
    camera_res = cameras.camera_res

    kbd = KeyboardController()
    kbd.start()

    def _on_sigint(signum, frame):
        if not kbd.quit:
            print("\n[signal] Ctrl-C received; finishing up and shutting down ...")
        kbd.quit = True

    signal.signal(signal.SIGINT, _on_sigint)

    print("\n[ready] SYNC button(s): engage teleop & record / disengage & save."
          "\n        Trigger drives the Aero Hand (grasp mode: " + grasp + ")."
          "\n        Keyboard: 'd' discard | 'h' home | 'q' quit.\n")

    demo_data = {}
    reset_data(demo_data)
    episode_id = episode_id_start
    synchronized = False
    discard_episode = False

    try:
        while not kbd.quit:
            loop_start = time.time()
            leader_info = {a: leaders[a].get_info() for a in arms}
            leader_joints = {a: leader_info[a][0] for a in arms}
            buttons_pressed = all(leader_info[a][1][0] > SYNC_BUTTON_THRESHOLD for a in arms)

            # Trigger -> hand, every cycle, engaged or not (matches deploy teleop).
            now = time.time()
            hand_frac = {a: hands[a].drive(leader_joints[a][6], now) for a in arms}

            if kbd.home.is_set():
                kbd.home.clear()
                if not synchronized:
                    print("[home] Homing followers AND leaders — LET GO of the handles ...")
                    for a in arms:
                        slow_move(robot_interface, a, np.zeros(7),
                                  robot_interface.get_joints(a))
                        leaders[a].go_home(kp=leader_kp[a], kd=leader_kd[a])
                    print("[home] Done.")
                else:
                    print("[home] Ignored: disengage teleop first.")

            if kbd.discard.is_set():
                kbd.discard.clear()
                discard_episode = True
                reset_data(demo_data)
                print("[episode] Marked for discard; buffer cleared.")

            if buttons_pressed:
                if not synchronized:
                    print("[teleop] Engaging ...")
                    for a in arms:
                        leaders[a].update_kp_kd(kp=leader_kp[a] * bilateral_kp, kd=np.zeros(6))
                        slow_move(robot_interface, a, leader_joints[a],
                                  robot_interface.get_joints(a))
                    synchronized = True
                    discard_episode = False
                    reset_data(demo_data)
                    print(f"[episode] Recording episode {episode_id} ...")
                else:
                    print("[teleop] Disengaging ...")
                    for a in arms:
                        leaders[a].update_kp_kd(kp=np.zeros(6), kd=np.zeros(6))
                    synchronized = False
                    if not discard_episode:
                        if save_demo(demo_data, demo_dir, episode_id, camera_res,
                                     robot_interface):
                            episode_id += 1
                    reset_data(demo_data)
                while all(leaders[a].get_info()[1][0] > SYNC_BUTTON_THRESHOLD for a in arms):
                    time.sleep(0.01)

            if synchronized:
                for a in arms:
                    command_arm(robot_interface, a, leader_joints[a])
                follower_joints = {a: robot_interface.get_joints(a) for a in arms}
                for a in arms:
                    leaders[a].command_joint_pos(follower_joints[a][:6])

                robot_joint_action = np.zeros(14)
                cmd_joint_action = np.zeros(14)
                tact, tact_q, h_act, h_pose, h_frac = {}, {}, {}, {}, {}
                for side in ("left", "right"):
                    tact[side], tact_q[side] = gloves.snapshot(side)
                for a in arms:
                    off = _ARM_OFFSET[a]
                    robot_joint_action[off:off + 6] = follower_joints[a][:6]
                    cmd_joint_action[off:off + 6] = leader_joints[a][:6]
                    # Gripper slot: commanded grasp fraction (EgoVerse-compatible).
                    robot_joint_action[off + 6] = hand_frac[a]
                    cmd_joint_action[off + 6] = hand_frac[a]
                    h_act[a] = hands[a].read_actuations()
                    h_pose[a] = hands[a].last_cmd_pose.copy()
                    h_frac[a] = hand_frac[a]

                demo_data["obs"].append(cameras.get_frames() if cameras else {})
                demo_data["robot_joint_actions"].append(robot_joint_action)
                demo_data["cmd_joint_actions"].append(cmd_joint_action)
                demo_data["tactile"].append(tact)
                demo_data["tactile_quat"].append(tact_q)
                demo_data["hand_actuations"].append(h_act)
                demo_data["hand_cmd_pose"].append(h_pose)
                demo_data["hand_cmd_frac"].append(h_frac)

                if episode_length is not None and len(demo_data["obs"]) >= episode_length:
                    print(f"[episode] Reached {episode_length} steps.")
                    for a in arms:
                        leaders[a].update_kp_kd(kp=np.zeros(6), kd=np.zeros(6))
                    synchronized = False
                    if not discard_episode:
                        if save_demo(demo_data, demo_dir, episode_id, camera_res,
                                     robot_interface):
                            episode_id += 1
                    reset_data(demo_data)

            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n[exit] KeyboardInterrupt; shutting down.")
    finally:
        _SHUTTING_DOWN.set()
        print("[exit] Closing hardware ...")

        def _safe(label, fn):
            try:
                fn()
            except BaseException as e:
                print(f"[exit] {label} error: {e}")

        underlying = [robot_interface.controller[a] for a in arms]
        underlying += [leaders[a]._robot for a in arms]
        for r in underlying:
            try:
                r.motor_chain.running = False
                r._stop_event.set()
            except BaseException:
                pass
        time.sleep(0.3)
        for a in arms:
            _safe(f"hand[{a}]", hands[a].open_and_close)
        if cameras is not None:
            _safe("cameras", cameras.close)
        for a in arms:
            _safe(f"leader[{a}]", leaders[a].close)
        _safe("followers", robot_interface.close)
        print("[exit] Done.")


# ------------------------- CLI -------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Collect YAM + AeroHand + JQ-glove demos via teleoperation.")
    parser.add_argument("--arms", choices=["left", "right", "both"], default="both")
    parser.add_argument("--left-leader-can", default=DEFAULT_LEADER_CHANNELS["left"])
    parser.add_argument("--right-leader-can", default=DEFAULT_LEADER_CHANNELS["right"])
    parser.add_argument("--left-follower-can", default=DEFAULT_FOLLOWER_CHANNELS["left"])
    parser.add_argument("--right-follower-can", default=DEFAULT_FOLLOWER_CHANNELS["right"])
    parser.add_argument("--hand-port", action="append", metavar="ARM=PATH", default=[],
                        help="Override an Aero Hand serial port, e.g. left=/dev/serial/by-id/...")
    parser.add_argument("--glove-port", action="append", default=[],
                        help="Pin glove serial port(s); default auto-detects CH343 cables.")
    parser.add_argument("--grasp", choices=sorted(GRASP_MODES), default="power",
                        help="Grasp synergy the trigger drives (default: power).")
    parser.add_argument("--camera-name", action="append", metavar="SERIAL=NAME")
    parser.add_argument("--frequency", type=float, default=DEFAULT_FREQUENCY)
    parser.add_argument("--demo-dir", default=DEFAULT_DEMO_DIR)
    parser.add_argument("--episode-id-start", type=int, default=0)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--bilateral-kp", type=float, default=0.15)
    args = parser.parse_args()

    arms = ["left", "right"] if args.arms == "both" else [args.arms]
    hand_ports = dict(DEFAULT_HAND_PORTS)
    for pair in args.hand_port:
        arm, path = pair.split("=", 1)
        hand_ports[arm.strip()] = path.strip()

    serial_to_name = dict(DEFAULT_CAMERA_NAMES)
    serial_to_name.update(parse_camera_names(args.camera_name))

    collect_dex_demo(
        arms=arms,
        leader_channels={"left": args.left_leader_can, "right": args.right_leader_can},
        follower_channels={"left": args.left_follower_can, "right": args.right_follower_can},
        hand_ports=hand_ports,
        glove_ports=args.glove_port,
        serial_to_name=serial_to_name,
        grasp=args.grasp,
        frequency=args.frequency,
        demo_dir=args.demo_dir,
        episode_id_start=args.episode_id_start,
        episode_length=args.episode_length,
        bilateral_kp=args.bilateral_kp,
    )


if __name__ == "__main__":
    main()
