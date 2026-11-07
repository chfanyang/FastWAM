"""CPU-only conventions shared by dataset preparation and the YARP client."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

TASKS = ("insert_pen", "pick_place", "stack_bowls_three", "white_board")
CAMERA_SERIALS = {"right_camera": "216322073340", "left_camera": "112322074336",
                  "wrist_camera": "216322072028"}
CAMERAS = ("left_camera", "wrist_camera")
PROMPT_TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: {task}"
PREPROCESSING = "rgb_pil_contain_lanczos_224_center_black_v1"
POSE_CONVENTION = "fr3_link8_base_xyz_m_wxyz_future_measured_t1_to_t16"
GRIPPER_OPEN_THRESHOLD_M = 0.08


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_inventory_fingerprint(episode_path, length):
    """Match preparation's file inventory; detect changed/missing PNGs before cache use."""
    digest = hashlib.sha256()
    for camera in ("right_camera", "left_camera", "wrist_camera"):
        paths = sorted((Path(episode_path) / "image" / camera).glob("*.png"))
        if [p.name for p in paths] != [f"{i:06d}.png" for i in range(length)]:
            raise ValueError(f"Missing/misaligned images: {episode_path}/{camera}")
        for path in paths:
            stat = path.stat()
            digest.update(f"{camera}/{path.name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def write_json_checked(path, value):
    """Idempotent preparation; never silently replace a different manifest."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Existing artifact differs; use a new output directory: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def validate_pose(pose):
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape[-1:] != (7,) or not np.isfinite(pose).all():
        raise ValueError("Pose must be finite [...,7]")
    norm = np.linalg.norm(pose[..., 3:], axis=-1, keepdims=True)
    if np.any(np.abs(norm - 1) > 0.05):
        raise ValueError("Invalid quaternion norm")
    result = pose.copy()
    result[..., 3:] /= norm
    return result


def xyzw_to_wxyz(pose):
    return validate_pose(pose)[..., [0, 1, 2, 6, 3, 4, 5]].copy()


def wxyz_to_xyzw(pose):
    return validate_pose(pose)[..., [0, 1, 2, 4, 5, 6, 3]].copy()


def gripper_open(width, threshold=GRIPPER_OPEN_THRESHOLD_M):
    width = np.asarray(width)
    if not np.isfinite(width).all() or np.any((width < -0.001) | (width > 0.11)):
        raise ValueError("Invalid Robotiq measured width in metres")
    # Fully open is 0.085 m. A 0.06 m threshold mislabels wide-object grasps.
    return (width >= threshold).astype(np.float32)


def preprocess_rgb(image, *, color_order="RGB"):
    """Match resize_stack_blocks.py exactly; already resized PNGs are unchanged."""
    value = np.asarray(image)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError("Expected uint8 HWC three-channel image")
    if color_order == "BGR":
        value = value[..., ::-1]
    elif color_order != "RGB":
        raise ValueError("color_order must be RGB or BGR")
    im = Image.fromarray(np.ascontiguousarray(value), "RGB")
    resized = ImageOps.contain(im, (224, 224), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (224, 224), (0, 0, 0))
    canvas.paste(resized, ((224 - resized.width) // 2, (224 - resized.height) // 2))
    return np.array(canvas)


def rgb_canvas(images, camera_keys=CAMERAS):
    return np.concatenate([preprocess_rgb(images[key]) for key in camera_keys], axis=1)


def load_episode(path):
    with np.load(Path(path) / "states.npz", allow_pickle=False) as state:
        pose = xyzw_to_wxyz(state["robot/ee_pose"])
        width = np.asarray(state["gripper/width"], dtype=np.float32)
        stamp = np.asarray(state["robot/robot_timestamp"], dtype=np.float64)
        joints = np.asarray(state["robot/joint_positions"], dtype=np.float32)
    if pose.ndim != 2 or width.shape != (len(pose),) or stamp.shape != width.shape:
        raise ValueError(f"Inconsistent state lengths: {path}")
    if joints.shape != (len(pose), 7) or not np.isfinite(joints).all():
        raise ValueError(f"Invalid joints: {path}")
    if not np.isfinite(stamp).all() or np.any(np.diff(stamp) <= 0):
        raise ValueError(f"Non-increasing robot timestamps: {path}")
    gripper_open(width)
    return pose, width, stamp, joints
