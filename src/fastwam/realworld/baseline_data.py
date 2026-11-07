"""Native YARP data for the original FastWAM video + ActionDiT architecture."""
from __future__ import annotations

import bisect
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .common import (CAMERAS, POSE_CONVENTION, PREPROCESSING, PROMPT_TEMPLATE,
                     gripper_open, load_episode, sha256_file, validate_pose)
from .dataset import load_context

REPRESENTATION = "fr3_current_ee_relative_xyz_rotvec_gripper_v1"


def quaternion_multiply(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    w = a[..., :1] * b[..., :1] - np.sum(a[..., 1:] * b[..., 1:], axis=-1, keepdims=True)
    xyz = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + np.cross(a[..., 1:], b[..., 1:])
    return np.concatenate((w, xyz), axis=-1)


def rotate(q, v):
    q, v = np.asarray(q, dtype=np.float64), np.asarray(v, dtype=np.float64)
    uv = np.cross(q[..., 1:], v)
    return v + 2 * (q[..., :1] * uv + np.cross(q[..., 1:], uv))


def canonical_quaternion(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    # Largest component breaks the sign tie for exactly pi rotations.
    fallback = np.take_along_axis(q, np.abs(q).argmax(axis=-1)[..., None], axis=-1)
    sign_source = np.where(np.abs(q[..., :1]) < 1e-12, fallback, q[..., :1])
    return np.where(sign_source < 0, -q, q)


class ActionCodec:
    """7D numerical actions; current absolute pose is an 8D proprio token.

    Each future pose is relative to the SAME current pose, not its predecessor.
    Rotation vectors are principal axis-angle vectors in radians, scaled by pi.
    """
    def __init__(self, stats_path, split_manifest=None):
        self.stats = json.loads(Path(stats_path).read_text())
        self.fingerprint = sha256_file(stats_path)
        if self.stats.get("representation") != REPRESENTATION:
            raise ValueError("Not original-FastWAM realworld action statistics")
        if split_manifest and self.stats["split_manifest_sha256"] != sha256_file(split_manifest):
            raise ValueError("Action statistics / data split mismatch")
        self.lo = np.asarray(self.stats["absolute_position_min"], dtype=np.float64)
        self.hi = np.asarray(self.stats["absolute_position_max"], dtype=np.float64)
        self.scale = np.asarray(self.stats["relative_position_bounds"], dtype=np.float64)
        if (self.lo.shape != (3,) or self.hi.shape != (3,) or self.scale.shape != (3,)
                or not np.isfinite([self.lo, self.hi, self.scale]).all()
                or np.any(self.hi <= self.lo) or np.any(self.scale <= 0)):
            raise ValueError("Invalid action normalization bounds")

    def proprio(self, poses, grip):
        poses = validate_pose(poses)
        grip = np.asarray(grip, dtype=np.float64)
        if grip.shape != poses.shape[:-1] or not np.isfinite(grip).all() or np.any((grip < 0) | (grip > 1)):
            raise ValueError("Invalid gripper states")
        return np.concatenate((2 * (poses[..., :3] - self.lo) / (self.hi - self.lo) - 1,
            canonical_quaternion(poses[..., 3:]), 2 * grip[..., None] - 1), axis=-1).astype(np.float32)

    def encode(self, current, future, grip):
        current, future = validate_pose(current), validate_pose(future)
        if current.shape != (7,) or future.ndim != 2:
            raise ValueError("Expected current [7] and future [T,7]")
        grip = np.asarray(grip)
        if grip.shape != (len(future),) or not np.isfinite(grip).all() or np.any((grip < 0) | (grip > 1)):
            raise ValueError("Invalid future gripper states")
        inverse = current[3:].astype(np.float64) * np.array([1, -1, -1, -1])
        position = rotate(inverse, future[:, :3] - current[:3]) / self.scale
        q = canonical_quaternion(quaternion_multiply(inverse, future[:, 3:]))
        sine = np.linalg.norm(q[:, 1:], axis=-1, keepdims=True)
        angle = 2 * np.arctan2(sine, q[:, :1])
        rotation = q[:, 1:] * np.divide(angle, sine, out=np.full_like(sine, 2.), where=sine > 1e-12)
        return np.concatenate((position, rotation / np.pi, 2 * grip[:, None] - 1), axis=-1).astype(np.float32)

    def decode(self, current, actions):
        current = validate_pose(current)
        actions = np.asarray(actions, dtype=np.float64)
        if current.shape != (7,) or actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise ValueError("Expected finite normalized actions [T,7]")
        vector = actions[:, 3:6] * np.pi
        angle = np.linalg.norm(vector, axis=-1, keepdims=True)
        scale = .5 * np.sinc(angle / (2 * np.pi))
        relative_q = np.concatenate((np.cos(angle / 2), vector * scale), axis=-1)
        q = canonical_quaternion(quaternion_multiply(current[3:], relative_q))
        position = current[:3] + rotate(current[3:], actions[:, :3] * self.scale)
        pose = validate_pose(np.concatenate((position, q), axis=-1))
        return pose, np.clip((actions[:, 6] + 1) / 2, 0, 1).astype(np.float32)


class OriginalFastWAMDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, split_manifest, pretrained_norm_stats,
                 text_embedding_cache_dir, split="train", include_text_context=True,
                 task_name=None):
        if split not in ("train", "val"):
            raise ValueError("Expected train or val")
        self.root = Path(dataset_dir)
        self.manifest = json.loads(Path(split_manifest).read_text())
        if (tuple(self.manifest["camera_keys"]) != CAMERAS or self.manifest["horizon"] != 16
                or self.manifest["preprocessing"] != PREPROCESSING
                or self.manifest["pose_convention"] != POSE_CONVENTION):
            raise ValueError("Unexpected realworld data contract")
        self.codec = ActionCodec(pretrained_norm_stats, split_manifest)
        if self.codec.stats["gripper_open_threshold_m"] != self.manifest["gripper_open_threshold_m"]:
            raise ValueError("Gripper convention mismatch")
        available_tasks = sorted({r["task"] for r in self.manifest["episodes"]})
        if task_name is not None and task_name not in available_tasks:
            raise ValueError(f"Unknown realworld task {task_name!r}; expected one of {available_tasks}")
        self.task_name = task_name
        self.records = [r for r in self.manifest["episodes"]
                        if r["split"] == split and (task_name is None or r["task"] == task_name)]
        self.offsets = [0]
        for r in self.records:
            self.offsets.append(self.offsets[-1] + len(r["valid_starts"]))
        if not self.offsets[-1]:
            raise ValueError("Empty data split")
        self.num_frames, self.camera_keys = 17, CAMERAS
        self.episode_split_metadata = dict(split=split, sha256=sha256_file(split_manifest))
        self.text_cache = text_embedding_cache_dir
        self.include_text_context = include_text_context
        self._reset_caches()

    def _reset_caches(self):
        self._episode = lru_cache(maxsize=4)(self._read_episode)
        self._context = lru_cache(maxsize=32)(self._read_context)

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k not in ("_episode", "_context")}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._reset_caches()

    def _read_episode(self, index):
        record = self.records[index]
        ep = self.root / record["path"]
        if sha256_file(ep / "states.npz") != record["states_sha256"]:
            raise ValueError(f"Source states changed: {ep}")
        return load_episode(ep)

    def _read_context(self, instruction):
        return load_context(self.text_cache, instruction)

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        ep = bisect.bisect_right(self.offsets, index) - 1
        record = self.records[ep]
        start = record["valid_starts"][index - self.offsets[ep]]
        poses, widths, _, _ = self._episode(ep)
        poses = poses[start:start + 17]
        grip = gripper_open(widths[start:start + 17], self.manifest["gripper_open_threshold_m"])
        frames = []
        for t in range(start, start + 17):
            cameras = []
            for camera in self.camera_keys:
                with Image.open(self.root / record["path"] / "image" / camera / f"{t:06d}.png") as im:
                    if im.size != (224, 224):
                        raise ValueError("Expected resized 224x224 source images")
                    cameras.append(np.array(im.convert("RGB")))
            frames.append(np.concatenate(cameras, axis=1))
        result = dict(video=torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float().div(127.5).sub(1),
            action=torch.from_numpy(self.codec.encode(poses[0], poses[1:], grip[1:])),
            proprio=torch.from_numpy(self.codec.proprio(poses, grip)),
            current_endpose=torch.from_numpy(poses[0].copy()),
            future_endpose=torch.from_numpy(poses[1:].copy()), future_gripper=torch.from_numpy(grip[1:, None]),
            image_is_pad=torch.zeros(17, dtype=torch.bool), action_is_pad=torch.zeros(16, dtype=torch.bool),
            prompt=PROMPT_TEMPLATE.format(task=record["instruction"]))
        if self.include_text_context:
            context, mask = self._context(record["instruction"])
            result.update(context=context.clone(), context_mask=mask.clone())
        return result
