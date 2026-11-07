"""Pure-NumPy checks and synchronous YARP scheduling, testable without hardware."""
import time
import numpy as np

from .common import validate_pose, wxyz_to_xyzw
from .protocol import ACTION_SPACE, VERSION


def rotation_distance(a, b):
    a, b = validate_pose(a)[..., 3:].astype(np.float64), validate_pose(b)[..., 3:].astype(np.float64)
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    b /= np.linalg.norm(b, axis=-1, keepdims=True)
    dot = np.abs(np.sum(a * b, axis=-1))
    return 2 * np.arccos(np.clip(dot, 0, 1))


def checked_chunk(response, request_id, current_pose, *, replan_steps=4,
                  position_min, position_max, max_step_m=0.025, max_step_rad=0.15):
    if response.get("request_id") != request_id:
        raise ValueError("Stale/mismatched response ID")
    if response.get("protocol_version") != VERSION or response.get("action_space") != ACTION_SPACE:
        raise ValueError("Unexpected action protocol or coordinate frame")
    if response.get("action_dt") != 0.1 or not 1 <= replan_steps <= 16:
        raise ValueError("Expected 10 Hz trajectory and 1..16 execution steps")
    poses = validate_pose(response["target_pose_wxyz"])
    g = np.asarray(response["gripper_open"], dtype=np.float32)
    if poses.shape != (16, 7) or g.shape != (16,) or not np.isfinite(g).all():
        raise ValueError("Invalid action shapes/values")
    if np.any((g < 0) | (g > 1)):
        raise ValueError("Invalid gripper range")
    poses, g = poses[:replan_steps], g[:replan_steps]
    lo, hi = np.asarray(position_min), np.asarray(position_max)
    if lo.shape != (3,) or hi.shape != (3,) or not np.all(np.isfinite([lo, hi])) or np.any(hi <= lo):
        raise ValueError("Invalid Cartesian workspace")
    current_pose = validate_pose(current_pose)
    path = np.concatenate((current_pose[None], poses), axis=0)
    if np.any((path[:, :3] < lo) | (path[:, :3] > hi)):
        raise ValueError("Requested trajectory is outside configured workspace")
    if np.any(np.linalg.norm(np.diff(path[:, :3], axis=0), axis=-1) > max_step_m):
        raise ValueError("Requested translation step is too large")
    if np.any(rotation_distance(path[:-1], path[1:]) > max_step_rad):
        raise ValueError("Requested rotation step is too large")
    return poses, g


def execute_chunk(arm, gripper, poses, grip, current_pose, last_gripper_open, stop_event,
                  *, dt=0.1, max_pos_speed=0.05, max_rot_speed=0.2,
                  clock=time.monotonic, wall_clock=time.time):
    """Schedule a bounded arm chunk, then send gripper transitions at its deadlines.

    YARP's gripper schedule_waypoint is unimplemented. Never enqueue all gripper
    commands immediately alongside a future arm trajectory. A velocity limit may
    stretch a step; its gripper event follows that stretched arrival time.
    """
    if min(dt, max_pos_speed, max_rot_speed) <= 0:
        raise ValueError("Execution timing and speed limits must be positive")
    poses = validate_pose(poses)
    grip = np.asarray(grip)
    if poses.ndim != 2 or not 1 <= len(poses) <= 16 or grip.shape != (len(poses),):
        raise ValueError("Expected 1..16 poses and one gripper value per pose")
    if not np.isfinite(grip).all() or np.any((grip < 0) | (grip > 1)):
        raise ValueError("Invalid gripper targets")
    previous = validate_pose(current_pose)
    # Lead time once per chunk, not once per action. Unsaturated waypoint spacing
    # stays at 0.1 s. YARP consumes at most one queued command per 100 Hz tick.
    origin = clock()
    wall_origin = wall_clock()
    elapsed = 0.03
    arrivals = []
    for pose, g in zip(poses, grip):
        if stop_event.is_set():
            return last_gripper_open
        duration = max(dt,
            float(np.linalg.norm(pose[:3] - previous[:3])) / max_pos_speed,
            float(rotation_distance(pose, previous)) / max_rot_speed)
        elapsed += duration
        arm.schedule_waypoint(wxyz_to_xyzw(pose), wall_origin + elapsed)
        arrivals.append(origin + elapsed)
        previous = pose
    for arrival, g in zip(arrivals, grip):
        if stop_event.wait(max(0., arrival - clock())):
            break
        desired_open = bool(g >= 0.5)
        if desired_open != last_gripper_open:
            if desired_open:
                gripper.release()
            else:
                gripper.grasp(speed=0.1, force=10.)
            last_gripper_open = desired_open
    return last_gripper_open
