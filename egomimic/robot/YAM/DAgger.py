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
import queue
import sys
import threading
import time
from collections import deque
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
    SLOW_MOVE_DT,
    SLOW_MOVE_STEPS,
    STUCK_FRAME_THRESHOLD,
    SYNC_BUTTON_THRESHOLD,
    YAMLeaderRobot,
    _ARM_OFFSET,
    _has_stuck_frames,
    fk_eepose14,
    make_leader,
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

    ee_error and action_variance are kept in SEPARATE per-arm Pipelines (each
    a single-heuristic Pipeline, not bundled together) so update_pipelines can
    gate action_variance on a MotionGate without also gating ee_error, which
    must keep running every tick even while an arm is deliberately held still.
    """
    names = set(only) if only else set(DEFAULT_HEURISTICS)

    per_arm_ee_error = {}
    per_arm_action_variance = {}
    for arm in arms_list:
        if "ee_error" in names:
            per_arm_ee_error[arm] = Pipeline([EeError()])
        if "action_variance" in names:
            per_arm_action_variance[arm] = Pipeline([ActionVariance()])

    task_pipeline = None
    if "robometer" in names:
        robometer = Robometer()
        reason = robometer.skip_reason()
        if reason:
            print(f"[dagger] heuristic 'robometer' skipped: {reason}")
        else:
            task_pipeline = Pipeline([robometer])

    return per_arm_ee_error, per_arm_action_variance, task_pipeline


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


class AsyncPipelineWorker:
    """Runs a Pipeline's update() on a background thread so a slow heuristic
    (robometer's VLM call over a UNIX socket measures ~96ms per call on this
    GPU — ~2.9x a 30Hz control tick's 33ms budget) never blocks the real-time
    control loop.

    Only the most recent submitted frame is kept: submit() is non-blocking and
    replaces any not-yet-processed frame rather than queueing behind it, so a
    busy worker never makes the backlog (and therefore the staleness of the
    result the control loop reads) grow over time. The tradeoff this makes
    real-time-safe is that failure detection here runs on the latest
    completed result, not necessarily the latest frame — acceptable since
    robometer's own signal is already smoothed/debounced over ~150 frames.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self._mailbox = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._latest_results = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame):
        try:
            self._mailbox.get_nowait()  # drop the stale not-yet-processed frame, if any
        except queue.Empty:
            pass
        try:
            self._mailbox.put_nowait(frame)
        except queue.Full:
            pass  # lost a race with _run draining the mailbox; harmless, resubmitted next tick

    def latest_results(self):
        with self._lock:
            return dict(self._latest_results)

    def reset(self):
        with self._lock:
            self._latest_results = {}
        self.pipeline.reset()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self):
        while not self._stop.is_set():
            try:
                frame = self._mailbox.get(timeout=0.1)
            except queue.Empty:
                continue
            results = self.pipeline.update(frame)
            with self._lock:
                self._latest_results = results


DEFAULT_ACTION_VARIANCE_MIN_MOTION_DEG = 0.5


class MotionGate:
    """Tracks each arm's recent commanded ARM-joint motion (radians, gripper
    excluded — different units/scale) to tell whether an arm is genuinely
    moving right now.

    Exists because action_variance's jitter score has a normalization flaw
    for a STATIONARY arm: jitter = sum(reversals * weight) / (sum(weight) +
    1e-9), where weight is meant to "ignore near-still joints" but only
    discriminates BETWEEN joints within one arm — it does nothing when the
    WHOLE arm is still. A held-still arm's commanded value is ~constant, so
    its tick-to-tick delta is pure noise (IK numerical noise, float
    precision) around zero; that noise flips sign ~randomly, giving a
    reversal rate near 1.0. Since weight and reversals end up roughly
    proportional across joints in that case, the 1e-9 epsilon (meant only to
    guard an exact 0/0) becomes negligible and the ratio converges toward
    ~1.0 — i.e. a genuinely idle arm can score HIGHER jitter than a smoothly
    moving one. Observed in practice: a stationary arm tripping
    action_variance on the very first action of a run (the 20-tick window
    fills starting at t=0, right after set_home(), so that first evaluation
    is pure hold-noise with no real motion yet to dilute it).

    update_pipelines uses is_moving() to skip feeding action_variance a
    stationary arm's noise-only sample entirely (not just discard the
    result) — action_variance's own StickyLatch is sticky, so a value that's
    allowed to latch True internally would stay True forever even after real
    motion resumes; the fix has to happen before that sample ever reaches it.
    """

    def __init__(self, min_motion_rad, window=20):
        self.min_motion_rad = min_motion_rad
        self.window = window
        self._prev = {}
        self._recent = {}

    def is_moving(self, arm, action):
        arm_joints = np.asarray(action, dtype=float)[:6]
        prev = self._prev.get(arm)
        self._prev[arm] = arm_joints
        if prev is None:
            return True  # no delta yet for this arm; nothing to gate on
        recent = self._recent.setdefault(arm, deque(maxlen=self.window))
        recent.append(np.abs(arm_joints - prev))
        avg_motion = float(np.stack(recent).mean(axis=0).sum())
        return avg_motion >= self.min_motion_rad


def update_pipelines(per_arm_ee_error, per_arm_action_variance, motion_gate, task_worker,
                      step_i, t, obs, commanded_joints):
    """Feed one control tick to every pipeline; return the (arm, heuristic_name,
    Outputs) triples for anything that just latched failed. arm is None for the
    task-level (robometer) worker.
    """
    failures = []
    for arm, joints in commanded_joints.items():
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

        ee_pipeline = per_arm_ee_error.get(arm)
        if ee_pipeline is not None:
            for name, outputs in ee_pipeline.update(frame).items():
                if outputs.failed:
                    failures.append((arm, name, outputs))

        av_pipeline = per_arm_action_variance.get(arm)
        if av_pipeline is not None and motion_gate.is_moving(arm, joints):
            for name, outputs in av_pipeline.update(frame).items():
                if outputs.failed:
                    failures.append((arm, name, outputs))
            # else: arm isn't meaningfully moving — skip the update entirely so
            # this noise-only sample never enters action_variance's window/latch.

    if task_worker is not None:
        image = obs.get("front_img_1")
        if image is not None:
            task_worker.submit(Frame(index=step_i, time=t, columns={"image": image}))
        for name, outputs in task_worker.latest_results().items():
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


def _slow_move_leader(leader, target6, current6, dt=SLOW_MOVE_DT):
    """Slow-interpolate the LEADER teaching handle to the FOLLOWER's current
    pose — the reverse of collect_yam_demo.py's slow_move(), which moves the
    follower to the leader. In DAgger the follower is mid-task when a failure
    triggers, while the passive leader could be resting anywhere; snapping the
    follower to the leader's arbitrary pose would yank it out of the task, so
    instead the (currently passive, back-drivable) handle repositions itself
    to match the robot's current state before the operator takes hold of it.

    ``dt`` (seconds/step, default matches collect_yam_demo.py's SLOW_MOVE_DT)
    is exposed separately here — via --leader-sync-dt — because a fast sync
    move here just needs to look/feel reasonable to a waiting operator,
    unlike collect_yam_demo.py's follower-catch-up move.
    """
    for i in range(SLOW_MOVE_STEPS + 1):
        alpha = i / SLOW_MOVE_STEPS
        cmd6 = alpha * target6 + (1 - alpha) * current6
        leader.command_joint_pos(cmd6)
        time.sleep(dt)


def _reset_triggered(failures, per_arm_ee_error, per_arm_action_variance, task_worker, policy):
    """Clear the latch state of whichever heuristic(s) fired, and drop the
    policy's stale action chunk so the next tick re-infers from scratch — used
    both after a real correction and after an operator override/ignore.
    """
    # Only the triggered pipeline(s)/worker get their latch cleared — an
    # untriggered heuristic's sliding window is left alone.
    for arm, name, _outputs in failures:
        if arm is None:
            target = task_worker
        elif name == "ee_error":
            target = per_arm_ee_error.get(arm)
        elif name == "action_variance":
            target = per_arm_action_variance.get(arm)
        else:
            target = None
        if target is not None:
            target.reset()

    if hasattr(policy, "actions"):
        policy.actions = None
    if hasattr(policy, "debug_actions"):
        policy.debug_actions = None


def handle_failure(ri, leaders, cameras, arms_list, kp, failures, policy, per_arm_ee_error,
                    per_arm_action_variance, task_worker, dagger_dir, episode_id,
                    policy_checkpoint, bilateral_kp, strict_cameras, frequency,
                    leader_sync_dt):
    """Freeze the policy, hand control to the teaching-handle leaders, record the
    human's correction, then hand control back. Returns (episode_id, quit_requested).

    An operator can also press 'i'+Enter instead of engaging the leader, to
    override a misclassified/false-positive detection: the triggered
    heuristic's latch is cleared and the rollout resumes immediately, with no
    teleop and no episode saved.
    """
    trigger_summary = ", ".join(
        f"{(arm or 'task')}:{name}={outputs.scores}" for arm, name, outputs in failures
    )
    print("\n--- DAGGER INTERVENTION: policy failure detected ---")
    print(f"  triggered by: {trigger_summary}")
    print("  Press the teaching-handle sync button to take over and demonstrate "
          "the correct continuation, 'i'+Enter to ignore this and keep rolling "
          "out (false positive), or 'q'+Enter to quit.\n")

    leader_kp = {a: leaders[a].kp for a in arms_list}

    def _quit_requested():
        ch = kp.getch()
        return ch is not None and ch.lower() == "q"

    # Wait for the operator to engage (sync button), override (keyboard 'i'),
    # or bail (keyboard 'q'). A single getch() per tick — checking 'q' and 'i'
    # with two separate getch() calls would drop keystrokes, since the first
    # call already drains the one buffered character.
    while True:
        ch = kp.getch()
        if ch is not None:
            ch = ch.lower()
            if ch == "q":
                print("[dagger] quit requested during intervention wait.")
                return episode_id, True
            if ch == "i":
                print("[dagger] Ignoring detected failure; resuming rollout.\n")
                _reset_triggered(failures, per_arm_ee_error, per_arm_action_variance,
                                  task_worker, policy)
                return episode_id, False
        leader_info = {a: leaders[a].get_info() for a in arms_list}
        if all(leader_info[a][1][0] > SYNC_BUTTON_THRESHOLD for a in arms_list):
            break
        time.sleep(0.01)

    print("[dagger] Syncing leader to the follower's current pose ...")
    for a in arms_list:
        leaders[a].update_kp_kd(kp=leader_kp[a] * bilateral_kp, kd=np.zeros(6))
        follower_current6 = ri.get_joints(a)[:6]
        leader_current6 = leader_info[a][0][:6]
        _slow_move_leader(leaders[a], follower_current6, leader_current6, dt=leader_sync_dt)

    # Debounce the engaging press.
    while all(leaders[a].get_info()[1][0] > SYNC_BUTTON_THRESHOLD for a in arms_list):
        time.sleep(0.01)

    # The arm is now synced and holding (kp/kd active), but the gripper
    # trigger is a passive encoder — not driven by kp/kd — so the operator
    # can freely reposition it to match the follower's held gripper state
    # before anything actually starts moving. Wait for a second press so
    # they have that time, instead of driving off the handle immediately.
    print("[dagger] Leader synced. Position the gripper trigger to match, then "
          "press the sync button again to start driving.")
    while True:
        if _quit_requested():
            print("[dagger] quit requested before teleop start.")
            for a in arms_list:
                leaders[a].update_kp_kd(kp=np.zeros(6), kd=np.zeros(6))
            return episode_id, True
        ready_info = {a: leaders[a].get_info() for a in arms_list}
        if all(ready_info[a][1][0] > SYNC_BUTTON_THRESHOLD for a in arms_list):
            break
        time.sleep(0.01)

    # Debounce the "start driving" press.
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

    _reset_triggered(failures, per_arm_ee_error, per_arm_action_variance, task_worker, policy)

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
    action_variance_min_motion_deg,
    leader_sync_dt,
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

    per_arm_ee_error, per_arm_action_variance, task_pipeline = build_pipelines(arms_list, only)
    warmup_pipelines(task_pipeline, *per_arm_ee_error.values(), *per_arm_action_variance.values())
    # robometer's VLM call measures ~96ms/call on this GPU (~2.9x a 30Hz tick
    # budget) — run it off the control-loop thread so it can never stall
    # robot commands; see AsyncPipelineWorker.
    task_worker = AsyncPipelineWorker(task_pipeline) if task_pipeline is not None else None
    motion_gate = MotionGate(min_motion_rad=np.deg2rad(action_variance_min_motion_deg))

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
                            per_arm_ee_error, per_arm_action_variance, motion_gate,
                            task_worker, step_i, step_i / frequency, obs, commanded_joints,
                        )
                        if failures:
                            episode_id, quit_requested = handle_failure(
                                ri, leaders, cameras, arms_list, kp, failures, policy,
                                per_arm_ee_error, per_arm_action_variance, task_worker,
                                dagger_dir, episode_id, policy_path, bilateral_kp,
                                strict_cameras, frequency, leader_sync_dt,
                            )
                            if quit_requested:
                                return
                            break  # re-enter the RateLoop fresh after a correction

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected, exiting.")
        return
    finally:
        _SHUTTING_DOWN.set()
        if task_worker is not None:
            task_worker.close()
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

#python DAgger.py --arms both --policy-path /home/mecka/EgoVerse/logs/yam_pick_hat_wrist/pi05_yam_pick_hat_wrist_2026-07-03_05-41-12/checkpoints/step10000.ckpt --cartesian --annotation "pick up the black hat using the left arm and drop it into a brown box" --left-follower-can can_follower_l --right-follower-can can_follower_r --left-leader-can can_leader_l --right-leader-can can_leader_r --ee-convention libero --strict-cameras
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
    parser.add_argument(
        "--action-variance-min-motion-deg", type=float,
        default=DEFAULT_ACTION_VARIANCE_MIN_MOTION_DEG,
        help="Minimum average per-tick arm-joint motion (degrees, gripper excluded) "
        "for action_variance to run at all on that arm this tick. Below this, an "
        "arm is treated as stationary and its noise floor is never fed to "
        "action_variance (see MotionGate) — a genuinely still arm's IK/numerical "
        "noise otherwise trips action_variance's jitter score even though nothing "
        "is actually moving. Lower this if a real slow jitter needs to be caught "
        "closer to standstill; raise it if idle arms are still triggering.",
    )
    parser.add_argument(
        "--leader-sync-dt", type=float, default=3 * SLOW_MOVE_DT,
        help="Seconds per interpolation step (100 steps total) when syncing the "
        "leader handle to the follower's current pose after a failure triggers "
        "(default: 3x collect_yam_demo.py's follower-catch-up speed, since this "
        "just needs to look reasonable to a waiting operator, not track a live "
        "human). Raise this further if the handle still snaps too fast.",
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
        action_variance_min_motion_deg=args.action_variance_min_motion_deg,
        leader_sync_dt=args.leader_sync_dt,
    )


if __name__ == "__main__":
    run_from_args(build_arg_parser().parse_args())
