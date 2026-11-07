#!/usr/bin/env python3
"""YARP EE-pose client. Imports robot drivers only after argument parsing."""
import argparse
from contextlib import ExitStack
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np

from fastwam.realworld.common import (CAMERAS, CAMERA_SERIALS, PREPROCESSING,
    preprocess_rgb, xyzw_to_wxyz, gripper_open)
from fastwam.realworld.execution import checked_chunk, execute_chunk, rotation_distance
from fastwam.realworld.protocol import pack, unpack, ACTION_SPACE, VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yarp-root", type=Path, required=True)
    parser.add_argument("--server", default="ws://127.0.0.1:8009")
    parser.add_argument("--robot-host", default="localhost")
    parser.add_argument("--camera-color-order", choices=("RGB", "BGR"), default="RGB",
                        help="Raw camera output: RGB for rs.format.rgb8 (default), BGR for bgr8")
    parser.add_argument("--prompt", required=True, help="Exact prepared dataset instruction")
    parser.add_argument("--replan-steps", type=int, default=4)
    parser.add_argument("--max-latency", type=float, default=5., help="Reject replies slower than this many seconds")
    parser.add_argument("--max-cycles", type=int, default=0, help="0 runs until Ctrl-C")
    parser.add_argument("--max-pos-speed", type=float, default=0.05)
    parser.add_argument("--max-rot-speed", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true", help="Read real observations but issue no arm/gripper commands")
    args = parser.parse_args()
    if not 1 <= args.replan_steps <= 16 or min(args.max_latency, args.max_pos_speed, args.max_rot_speed) <= 0:
        parser.error("Invalid replan steps, latency or speed")
    from websockets.sync.client import connect
    sys.path.insert(0, str(args.yarp_root.resolve()))
    from src.yarp.camera.realsense_safe import RealSenseCamera
    from src.yarp.robot.franka.controller import FrankaController
    from src.yarp.robot.franka.interface import FrankaInterface
    from src.yarp.robot.robotiq2f.controller import Robotiq2FController
    from src.yarp.common.types import ActionType

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    with connect(args.server, compression=None, max_size=8 * 1024 * 1024,
                 open_timeout=10) as connection:
        metadata = unpack(connection.recv(timeout=10))
        if (metadata.get("protocol_version") != VERSION or metadata.get("action_space") != ACTION_SPACE
            or tuple(metadata.get("camera_keys", ())) != CAMERAS
            or metadata.get("preprocessing") != PREPROCESSING):
            raise ValueError("Server metadata does not match left+wrist YARP protocol")
        if args.prompt not in metadata["instructions"]:
            raise ValueError(f"Prompt not prepared; choose from {metadata['instructions']}")
        threshold = metadata["gripper_open_threshold_m"]
        codec = metadata["codec"]
        lo, hi = codec["absolute_position_min"], codec["absolute_position_max"]
        with SharedMemoryManager() as manager, ExitStack() as stack:
            cameras = {key: stack.enter_context(RealSenseCamera(shm_manager=manager,
                serial_number=CAMERA_SERIALS[key], use_depth=False, fps=15,
                width=640, height=480, get_max_k=150)) for key in CAMERAS}
            gripper = stack.enter_context(Robotiq2FController(shm_manager=manager,
                robot_ip=args.robot_host, frequency=50, max_speed=0.1,
                init_width=None))  # No implicit opening on connection.
            if args.dry_run:
                # Unlike FrankaController, this interface never starts a control policy.
                observer = FrankaInterface(ip_address=args.robot_host)
                arm = None
                read_pose = lambda: xyzw_to_wxyz(observer.get_ee_pose())
            else:
                arm = stack.enter_context(FrankaController(shm_manager=manager,
                    robot_ip=args.robot_host, frequency=100, action_type=ActionType.EE_POSE,
                    joints_init=None, max_ee_pos_speed=args.max_pos_speed,
                    max_ee_rot_speed=args.max_rot_speed))
                read_pose = lambda: xyzw_to_wxyz(arm.get_state()["ee_pose"])
            last_open = bool(gripper_open(gripper.get_state()["width"], threshold))
            request_id = 0
            while not stop.is_set() and (args.max_cycles == 0 or request_id < args.max_cycles):
                images = {key: preprocess_rgb(camera.get_state()["color"], color_order=args.camera_color_order)
                          for key, camera in cameras.items()}
                observed_pose = read_pose()
                observation = dict(protocol_version=VERSION, request_id=request_id,
                    ee_link="fr3_link8", color_order="RGB", images=images,
                    ee_pose_wxyz=observed_pose,
                    gripper_open=float(gripper_open(gripper.get_state()["width"], threshold)),
                    prompt=args.prompt)
                started = time.monotonic()
                connection.send(pack(observation))
                # Short receive polls allow Ctrl-C even while inference is running.
                while not stop.is_set():
                    if time.monotonic() - started > args.max_latency:
                        raise TimeoutError("Inference response expired; no actions submitted")
                    try:
                        reply = connection.recv(timeout=0.1)
                        break
                    except TimeoutError:
                        continue
                if stop.is_set():
                    break
                if isinstance(reply, str):
                    raise RuntimeError(reply)
                if time.monotonic() - started > args.max_latency:
                    raise TimeoutError("Inference response expired")
                response = unpack(reply)
                current_pose = read_pose()
                if (np.linalg.norm(current_pose[:3] - observed_pose[:3]) > 0.02
                    or rotation_distance(current_pose, observed_pose) > 0.15):
                    raise ValueError("Robot moved while inference was pending; observation is stale")
                poses, grip = checked_chunk(response, request_id, current_pose,
                    replan_steps=args.replan_steps, position_min=lo, position_max=hi)
                print(f"request={request_id} latency={time.monotonic()-started:.3f}s steps={len(poses)} dry_run={args.dry_run}", flush=True)
                delta_mm = (poses[:, :3] - current_pose[:3]) * 1000
                rotation_deg = np.rad2deg(rotation_distance(current_pose, poses))
                all_targets = np.asarray(response["target_pose_wxyz"])
                print(f"  current_xyz_m={np.round(current_pose[:3], 6).tolist()}", flush=True)
                print(f"  selected_delta_xyz_mm={np.round(delta_mm, 3).tolist()}"
                      f" selected_rotation_deg={np.round(rotation_deg, 3).tolist()}"
                      f" gripper_open={np.round(grip, 3).tolist()}", flush=True)
                print(f"  full16_delta_norm_mm={np.round(np.linalg.norm(all_targets[:, :3] - current_pose[:3], axis=-1) * 1000, 3).tolist()}", flush=True)
                if not args.dry_run:
                    if not arm.is_alive() or not arm.is_ready:
                        raise RuntimeError("Franka controller is not alive/ready; no actions submitted")
                    before_state = arm.get_state()
                    execution_started = time.monotonic()
                    last_open = execute_chunk(arm, gripper, poses, grip, current_pose,
                        last_open, stop, max_pos_speed=args.max_pos_speed,
                        max_rot_speed=args.max_rot_speed)
                    after_state = arm.get_state()
                    actual = xyzw_to_wxyz(after_state["ee_pose"])
                    print(f"  execution_s={time.monotonic()-execution_started:.3f}"
                          f" measured_delta_xyz_mm={np.round((actual[:3]-current_pose[:3])*1000, 3).tolist()}"
                          f" final_target_error_mm={np.linalg.norm(actual[:3]-poses[-1, :3])*1000:.3f}"
                          f" state_time_advance_s={float(after_state['robot_timestamp'])-float(before_state['robot_timestamp']):.3f}"
                          f" controller_alive={arm.is_alive()}", flush=True)
                request_id += 1


if __name__ == "__main__":
    main()
