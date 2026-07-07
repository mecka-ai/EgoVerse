#!/usr/bin/env python3
"""Live DAgger rollout for the I2RT YAM bimanual arms.

Runs a trained policy exactly like ``yam_rollout.py``, but streams every
control tick's observation/action through the ``failure_dashboard`` online
heuristics pipeline (ee_error + action_variance per arm, robometer for the
whole task). The moment a heuristic latches ``failed``, the policy is frozen
and control hands to a human via the teaching-handle leader arms — the same
bilateral engage/disengage protocol ``collect_yam_demo.py`` uses for teleop.
The human's correction is recorded as a labeled HDF5 episode (tagged with
what triggered it), then the policy resumes automatically. This is the live
DAgger loop: the ``failure-dashboard`` repo's own roadmap calls wiring a real
robot into its heuristics pipeline "the only hardware-dependent work left" —
this file is that wiring.

Deliberately reuses rather than reimplements:
  * yam_rollout.py      — PolicyRollout, the robot-interface builder, the
                           keyboard-intervention plumbing.
  * collect_yam_demo.py — the teaching-handle leader arms, the bilateral
                           engage/disengage teleop protocol, the HDF5 save
                           format.

Requires the teaching-handle LEADER arms to be connected in addition to the
follower arms (unlike plain yam_rollout.py, which only needs followers).

Example:
    python DAgger.py --policy-path <ckpt> --arms both --cartesian \\
        --annotation "Fold the shirt" \\
        --left-follower-can can_follower_l --right-follower-can can_follower_r \\
        --left-leader-can can_leader_l     --right-leader-can can_leader_r \\
        --dagger-dir ./dagger_corrections
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

# collect_yam_demo.py/yam_interface.py already do this sys.path insert on
# import; done here too so DAgger.py behaves the same when launched from a
# different cwd or invoked as a module rather than a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h5py
import numpy as np

from collect_yam_demo import (
    DEFAULT_FOLLOWER_CHANNELS,
    DEFAULT_LEADER_CHANNELS,
    STUCK_FRAME_THRESHOLD,
    SYNC_BUTTON_THRESHOLD,
    YAMLeaderRobot,
    _ARM_OFFSET,
    _has_stuck_frames,
    fk_eepose14,
    make_leader,
    slow_move,
)
from i2rt.robots.utils import ArmType, GripperType
from models.registry import MODEL_REGISTRY
from yam_cameras import YamCameraRig
from yam_rollout import (
    DEFAULT_FREQUENCY,
    DEFAULT_RESAMPLE_LENGTH,
    QUERY_FREQUENCY,
    PolicyRollout,
    RateLoop,
    _build_robot_interface,
    _match_training_front_resolution,
    _quiet_thread_excepthook,
    _KeyPoll,
    _SHUTTING_DOWN,
    reset_rollout,
)

from failure_dashboard.frame import Frame
from failure_dashboard.heuristics.action_variance import ActionVariance
from failure_dashboard.heuristics.ee_error import EeError
from failure_dashboard.heuristics.robometer import Robometer
from failure_dashboard.pipeline import Pipeline

DEFAULT_DAGGER_DIR = "./dagger_corrections"
DEFAULT_HEURISTICS = ("ee_error", "action_variance", "robometer")


# ------------------------- Heuristic wiring -------------------------


def build_pipelines(arms_list, only):
    """ee_error/action_variance are per-arm (each needs that arm's own joints);
    robometer is task-level (it judges the whole episode from the front camera).
    Fresh heuristic instances per arm/task — NOT the discover()'d singletons —
    so latch/EWMA state never bleeds between arms.
    """
    names = set(only) if only else set(DEFAULT_HEURISTICS)

    per_arm = {}
    for arm in arms_list:
        heuristics = []
        if "ee_error" in names:
            heuristics.append(EeError())
        if "action_variance" in names:
            heuristics.append(ActionVariance())
        if heuristics:
            per_arm[arm] = Pipeline(heuristics)

    task_pipeline = None
    if "robometer" in names:
        robometer = Robometer()
        reason = robometer.skip_reason()
        if reason:
            print(f"[dagger] heuristic 'robometer' skipped: {reason}")
        else:
            task_pipeline = Pipeline([robometer])

    return per_arm, task_pipeline


def warmup_pipelines(*pipelines):
    for pipeline in pipelines:
        if pipeline is None:
            continue
        for h in pipeline.heuristics:
            warmer = getattr(h, "warmup", None)
            if callable(warmer):
                print(f"[dagger] loading '{h.name}' model ...")
                warmer()
                print(f"[dagger] '{h.name}' ready.")


def update_pipelines(per_arm_pipelines, task_pipeline, step_i, t, obs, commanded_joints):
    """Feed one control tick to every pipeline; return the (arm, heuristic_name,
    Outputs) triples for anything that just latched failed. arm is None for the
    task-level pipeline.
    """
    failures = []
    for arm, pipeline in per_arm_pipelines.items():
        joints = commanded_joints.get(arm)
        if joints is None:
            continue  # IK failed this tick — hold, don't feed a bogus action
        off = _ARM_OFFSET[arm]
        frame = Frame(
            index=step_i, time=t,
            columns={
                "joint_positions": np.asarray(obs["joint_positions"][off:off + 7], dtype=float),
                "actions": np.asarray(joints, dtype=float),
            },
        )
        for name, outputs in pipeline.update(frame).items():
            if outputs.failed:
                failures.append((arm, name, outputs))

    if task_pipeline is not None:
        image = obs.get("front_img_1")
        if image is not None:
            frame = Frame(index=step_i, time=t, columns={"image": image})
            for name, outputs in task_pipeline.update(frame).items():
                if outputs.failed:
                    failures.append((None, name, outputs))

    return failures


# ------------------------- Correction recording -------------------------


def save_dagger_episode(demo_data, dagger_dir, episode_id, camera_res, robot_interface,
                         trigger, policy_checkpoint, strict_cameras=False):
    """Write one human-correction episode — same EgoVerse HDF5 layout as
    collect_yam_demo.py::save_demo, plus attrs tying it back to what triggered it
    so these episodes are easy to filter out of / mix into training data later.
    """
    n_steps = len(demo_data["robot_joint_actions"])
    if n_steps == 0:
        print("[dagger] No steps recorded; skipping save.")
        return False

    stuck_cams = [c for c in camera_res if _has_stuck_frames(demo_data["obs"], c)]
    if stuck_cams:
        msg = (
            f">{STUCK_FRAME_THRESHOLD} consecutive identical frames on camera(s): "
            f"{', '.join(stuck_cams)} (likely a stalled/disconnected stream)."
        )
        if strict_cameras:
            print(f"[dagger] ABORT (--strict-cameras): {msg} Episode not saved.")
            return False
        print(f"[dagger] WARNING: {msg} Saving anyway.")

    dagger_dir = Path(dagger_dir)
    dagger_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ms = int(time.time() * 1000)
    filename = dagger_dir / f"{timestamp_ms}.hdf5"
    print(f"[dagger] Writing correction episode {episode_id} ({n_steps} steps) to {filename}")

    robot_joints = np.asarray(demo_data["robot_joint_actions"])  # (T, 14)
    cmd_joints = np.asarray(demo_data["cmd_joint_actions"])      # (T, 14)
    obs_eepose = np.stack([fk_eepose14(robot_interface, j) for j in robot_joints])
    cmd_eepose = np.stack([fk_eepose14(robot_interface, j) for j in cmd_joints])

    trigger_heuristics = [name for _, name, _ in trigger]
    trigger_scores = {name: outputs.scores for _, name, outputs in trigger}
    trigger_arms = [arm for arm, _, _ in trigger if arm is not None]

    with h5py.File(str(filename), "w", rdcc_nbytes=1024**2 * 32) as root:
        root.attrs["sim"] = False
        root.attrs["timestamp_ms"] = timestamp_ms
        root.attrs["trigger_heuristic"] = json.dumps(trigger_heuristics)
        root.attrs["trigger_scores_json"] = json.dumps(trigger_scores)
        root.attrs["trigger_arm"] = json.dumps(trigger_arms)
        root.attrs["policy_checkpoint"] = policy_checkpoint or ""

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

        for cam_name in camera_res:
            ds = images[cam_name]
            for idx, obs_step in enumerate(demo_data["obs"]):
                frame = obs_step.get(cam_name)
                if frame is not None:
                    ds[idx] = frame[..., ::-1]

    print("[dagger] Done.")
    return True


# ------------------------- Failure -> teleop handoff -------------------------


def handle_failure(ri, leaders, cameras, arms_list, kp, failures, policy, per_arm_pipelines,
                    task_pipeline, dagger_dir, episode_id, policy_checkpoint, bilateral_kp,
                    strict_cameras, frequency):
    """Freeze the policy, hand control to the teaching-handle leaders, record the
    human's correction, then hand control back. Returns (episode_id, quit_requested).
    """
    trigger_summary = ", ".join(
        f"{(arm or 'task')}:{name}={outputs.scores}" for arm, name, outputs in failures
    )
    print("\n--- DAGGER INTERVENTION: policy failure detected ---")
    print(f"  triggered by: {trigger_summary}")
    print("  Press the teaching-handle sync button to take over and demonstrate "
          "the correct continuation, or 'q'+Enter to quit.\n")

    leader_kp = {a: leaders[a].kp for a in arms_list}

    def _quit_requested():
        ch = kp.getch()
        return ch is not None and ch.lower() == "q"

    # Wait for the operator to engage (sync button) or bail (keyboard 'q').
    while True:
        if _quit_requested():
            print("[dagger] quit requested during intervention wait.")
            return episode_id, True
        leader_info = {a: leaders[a].get_info() for a in arms_list}
        if all(leader_info[a][1][0] > SYNC_BUTTON_THRESHOLD for a in arms_list):
            break
        time.sleep(0.01)

    print("[dagger] Engaging teleop ...")
    for a in arms_list:
        leaders[a].update_kp_kd(kp=leader_kp[a] * bilateral_kp, kd=np.zeros(6))
        current7 = ri.get_joints(a)
        slow_move(ri, a, leader_info[a][0], current7)

    # Debounce the engaging press.
    while all(leaders[a].get_info()[1][0] > SYNC_BUTTON_THRESHOLD for a in arms_list):
        time.sleep(0.01)

    print("[dagger] Recording correction — press the sync button again to finish.")
    demo_data = {"obs": [], "robot_joint_actions": [], "cmd_joint_actions": []}
    dt = 1.0 / frequency
    while True:
        loop_start = time.time()
        if _quit_requested():
            print("[dagger] quit requested mid-correction; discarding & exiting.")
            for a in arms_list:
                leaders[a].update_kp_kd(kp=np.zeros(6), kd=np.zeros(6))
            return episode_id, True

        leader_info = {a: leaders[a].get_info() for a in arms_list}
        leader_joints = {a: leader_info[a][0] for a in arms_list}
        if all(leader_info[a][1][0] > SYNC_BUTTON_THRESHOLD for a in arms_list):
            break  # disengage

        for a in arms_list:
            ri.set_joints(leader_joints[a], a)
        follower_joints = {a: ri.get_joints(a) for a in arms_list}
        for a in arms_list:
            leaders[a].command_joint_pos(follower_joints[a][:6])

        robot_joint_action = np.zeros(14)
        cmd_joint_action = np.zeros(14)
        for a in arms_list:
            off = _ARM_OFFSET[a]
            robot_joint_action[off:off + 7] = follower_joints[a]
            cmd_joint_action[off:off + 7] = leader_joints[a]

        demo_data["obs"].append(cameras.get_frames())
        demo_data["robot_joint_actions"].append(robot_joint_action)
        demo_data["cmd_joint_actions"].append(cmd_joint_action)

        elapsed = time.time() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

    print("[dagger] Disengaging teleop ...")
    for a in arms_list:
        leaders[a].update_kp_kd(kp=np.zeros(6), kd=np.zeros(6))
    while all(leaders[a].get_info()[1][0] > SYNC_BUTTON_THRESHOLD for a in arms_list):
        time.sleep(0.01)

    if save_dagger_episode(demo_data, dagger_dir, episode_id, cameras.camera_res, ri,
                            trigger=failures, policy_checkpoint=policy_checkpoint,
                            strict_cameras=strict_cameras):
        episode_id += 1

    # Only the triggered pipeline(s) get their latch cleared — an untriggered
    # heuristic's sliding window is left alone.
    for arm, _name, _outputs in failures:
        pipeline = per_arm_pipelines.get(arm) if arm is not None else task_pipeline
        if pipeline is not None:
            pipeline.reset()

    if hasattr(policy, "actions"):
        policy.actions = None
    if hasattr(policy, "debug_actions"):
        policy.debug_actions = None

    print("[dagger] Correction saved; resuming automated rollout.\n")
    return episode_id, False


# ------------------------- Manual (keyboard) intervention -------------------------


def _enter_manual_intervention(kp, ri, policy):
    """Same c/h/a/q/r menu as yam_rollout.py's keyboard-triggered pause, so an
    operator can still step in manually between automated DAgger corrections.
    """
    import termios
    import tty

    termios.tcsetattr(kp.fd, termios.TCSADRAIN, kp.old)
    print("\n--- MANUAL INTERVENTION (rollout paused) ---")
    print("  c            : continue rollout")
    print("  h            : send arms to home (does not clear policy state)")
    print("  a <path>     : load new annotation file")
    print("  r            : restart rollout")
    print("  q            : quit")

    while True:
        try:
            cmd = input("> ").strip()
        except EOFError:
            tty.setcbreak(kp.fd)
            return "quit"

        if cmd == "c":
            print("Resuming rollout.")
            tty.setcbreak(kp.fd)
            return "continue"
        elif cmd == "q":
            tty.setcbreak(kp.fd)
            return "quit"
        elif cmd == "h":
            print("Sending arms to home...")
            ri.set_home()
            print("Arms at home. Still paused — c to continue, r to restart, q to quit.")
        elif cmd == "r":
            tty.setcbreak(kp.fd)
            return "restart"
        elif cmd.startswith("a "):
            ann_path = cmd[2:].strip()
            if not ann_path:
                print("Usage: a <annotation_path>")
                continue
            if policy.load_annotation(ann_path):
                os.environ["ROBOMETER_TASK"] = policy.annotation
        else:
            print(f"Unknown command: '{cmd}'. Use c / h / a <path> / r / q.")


# ------------------------- Main loop -------------------------


def dagger_rollout(
    arms,
    frequency,
    cartesian,
    query_frequency,
    policy_path,
    model_type,
    resampled_action_len,
    annotation_path,
    annotation_text,
    left_follower_can,
    right_follower_can,
    left_leader_can,
    right_leader_can,
    ee_convention,
    extrinsics_key,
    dagger_dir,
    episode_id_start,
    bilateral_kp,
    only,
    strict_cameras,
):
    threading.excepthook = _quiet_thread_excepthook

    if extrinsics_key is None:
        extrinsics_key = "yam"

    if arms == "both":
        arms_list = ["right", "left"]
    elif arms == "right":
        arms_list = ["right"]
    else:
        arms_list = ["left"]

    follower_channels = {"left": left_follower_can, "right": right_follower_can}
    leader_channels = {"left": left_leader_can, "right": right_leader_can}

    print("[dagger] Initializing follower arms + cameras ...")
    ri = _build_robot_interface(
        arms_list=arms_list,
        robot="yam",
        yam_channels={a: follower_channels[a] for a in arms_list},
        yam_ee_convention=ee_convention,
    )
    cameras = YamCameraRig(recorders=ri.recorders)  # non-owning view; ri.close() owns them

    print("[dagger] Initializing leader teaching handles ...")
    leaders = {
        a: make_leader(leader_channels[a], GripperType.YAM_TEACHING_HANDLE, ArmType.YAM)
        for a in arms_list
    }

    print(f"[dagger] Loading policy from {policy_path} (model_type={model_type})")
    policy = PolicyRollout(
        arm=arms,
        policy_path=policy_path,
        query_frequency=query_frequency,
        cartesian=cartesian,
        extrinsics_key=extrinsics_key,
        resampled_action_len=resampled_action_len,
        annotation_path=annotation_path,
        annotation_text=annotation_text,
        robot="yam",
        model_type=model_type,
    )
    if policy.annotation:
        os.environ["ROBOMETER_TASK"] = policy.annotation

    per_arm_pipelines, task_pipeline = build_pipelines(arms_list, only)
    warmup_pipelines(task_pipeline, *per_arm_pipelines.values())

    episode_id = episode_id_start

    try:
        with _KeyPoll() as kp:
            reset_rollout(ri, policy)
            result = _enter_manual_intervention(kp, ri, policy)
            if result == "quit":
                print("Quit requested.")
                return
            if result == "restart":
                reset_rollout(ri, policy)

            while True:  # restartable
                with RateLoop(frequency=frequency, verbose=True) as loop:
                    for step_i in loop:
                        ch = kp.getch()
                        if ch is not None:
                            result = _enter_manual_intervention(kp, ri, policy)
                            if result == "quit":
                                print("Quit requested.")
                                return
                            elif result == "restart":
                                print("Restart requested.")
                                reset_rollout(ri, policy)
                                result = _enter_manual_intervention(kp, ri, policy)
                                if result == "quit":
                                    return
                                if result == "restart":
                                    reset_rollout(ri, policy)
                                break
                            if hasattr(policy, "actions"):
                                policy.actions = None
                            break

                        obs = ri.get_obs()
                        _match_training_front_resolution(obs)
                        actions = policy.rollout_step(step_i, obs)

                        if actions is None:
                            print("Finish rollout.")
                            reset_rollout(ri, policy)
                            result = _enter_manual_intervention(kp, ri, policy)
                            if result == "quit":
                                return
                            if result == "restart":
                                reset_rollout(ri, policy)
                            break

                        commanded_joints = {}
                        for arm in arms_list:
                            arm_offset = 7 if (arm == "right" and arms == "both") else 0
                            arm_action = actions[arm_offset:arm_offset + 7]
                            if cartesian:
                                commanded_joints[arm] = ri.set_pose(arm_action, arm)
                            else:
                                ri.set_joints(arm_action, arm)
                                commanded_joints[arm] = arm_action

                        failures = update_pipelines(
                            per_arm_pipelines, task_pipeline, step_i,
                            step_i / frequency, obs, commanded_joints,
                        )
                        if failures:
                            episode_id, quit_requested = handle_failure(
                                ri, leaders, cameras, arms_list, kp, failures, policy,
                                per_arm_pipelines, task_pipeline, dagger_dir, episode_id,
                                policy_path, bilateral_kp, strict_cameras, frequency,
                            )
                            if quit_requested:
                                return
                            break  # re-enter the RateLoop fresh after a correction

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected, exiting.")
        return
    finally:
        _SHUTTING_DOWN.set()
        underlying = [ri.controller[a] for a in arms_list]
        underlying += [leaders[a]._robot for a in arms_list]
        for r in underlying:
            try:
                r.motor_chain.running = False
                r._stop_event.set()
            except BaseException:
                pass
        time.sleep(0.3)  # let control threads exit before CAN sockets close

        def _safe(label, fn):
            try:
                fn()
            except BaseException as e:
                print(f"[dagger] {label} error: {e}")

        for a in arms_list:
            _safe(f"leader[{a}]", leaders[a].close)
        _safe("followers", ri.close)
        print("[dagger] Done.")


def build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Live DAgger rollout for YAM: run the policy, detect "
        "failures online, hand off to the teaching-handle leaders for "
        "correction, resume."
    )
    parser.add_argument("--arms", choices=["left", "right", "both"], default="both")
    parser.add_argument("--frequency", type=float, default=DEFAULT_FREQUENCY)
    parser.add_argument("--query_frequency", type=int, default=QUERY_FREQUENCY)
    parser.add_argument("--policy-path", type=str, required=True, help="policy checkpoint path")
    parser.add_argument(
        "--model-type", type=str, default="pi05", choices=sorted(MODEL_REGISTRY),
        help="Policy backend to load --policy-path with (see models/registry.py).",
    )
    parser.add_argument("--cartesian", action="store_true")
    parser.add_argument("--resampled-action-len", type=int, default=DEFAULT_RESAMPLE_LENGTH)
    parser.add_argument("--annotation-path", type=str, help="path to the annotation file")
    parser.add_argument(
        "--annotation", type=str, default=None,
        help="inline language prompt, e.g. --annotation \"Fold the shirt\". "
        "Used only if --annotation-path is not provided; also feeds ROBOMETER_TASK.",
    )
    parser.add_argument("--left-follower-can", default=DEFAULT_FOLLOWER_CHANNELS["left"])
    parser.add_argument("--right-follower-can", default=DEFAULT_FOLLOWER_CHANNELS["right"])
    parser.add_argument("--left-leader-can", default=DEFAULT_LEADER_CHANNELS["left"])
    parser.add_argument("--right-leader-can", default=DEFAULT_LEADER_CHANNELS["right"])
    parser.add_argument(
        "--ee-convention", default="default", choices=["default", "libero"],
        help="YAM grasp_site frame convention. MUST match the convention the "
        "training demos were collected with.",
    )
    parser.add_argument("--extrinsics-key", type=str, default=None)
    parser.add_argument("--dagger-dir", default=DEFAULT_DAGGER_DIR,
                         help="Where correction episodes are saved (separate from regular demos).")
    parser.add_argument("--episode-id-start", type=int, default=0)
    parser.add_argument("--bilateral-kp", type=float, default=0.05)
    parser.add_argument(
        "--only", nargs="+", default=None, choices=list(DEFAULT_HEURISTICS),
        help="Restrict live failure detection to these heuristics (default: all three).",
    )
    parser.add_argument(
        "--strict-cameras", action="store_true",
        help="Abort (don't save) a correction if any camera stalls during it.",
    )
    return parser


def run_from_args(args):
    return dagger_rollout(
        arms=args.arms,
        frequency=args.frequency,
        cartesian=args.cartesian,
        query_frequency=args.query_frequency,
        policy_path=args.policy_path,
        model_type=args.model_type,
        resampled_action_len=args.resampled_action_len,
        annotation_path=args.annotation_path,
        annotation_text=args.annotation,
        left_follower_can=args.left_follower_can,
        right_follower_can=args.right_follower_can,
        left_leader_can=args.left_leader_can,
        right_leader_can=args.right_leader_can,
        ee_convention=args.ee_convention,
        extrinsics_key=args.extrinsics_key,
        dagger_dir=args.dagger_dir,
        episode_id_start=args.episode_id_start,
        bilateral_kp=args.bilateral_kp,
        only=args.only,
        strict_cameras=args.strict_cameras,
    )


if __name__ == "__main__":
    run_from_args(build_arg_parser().parse_args())
