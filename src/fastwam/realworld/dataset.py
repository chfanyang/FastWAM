"""Native YARP PNG/NPZ windows with absolute RAY0 and relative future poses."""
from __future__ import annotations

import bisect
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fastwam.datasets.latent_cache import LatentCacheReader
from fastwam.representations.libero_rothko import LiberoRothkoCodec, LiberoRothkoCodecConfig
from fastwam.representations.rothko import RothkoNormStats
from .common import (CAMERAS, PROMPT_TEMPLATE, PREPROCESSING, POSE_CONVENTION, gripper_open,
                     load_episode, sha256_file, rgb_canvas)


def make_codec(norm_stats, rothko_config=None):
    stats = RothkoNormStats.load(norm_stats) if isinstance(norm_stats, (str, Path)) else norm_stats
    canonical = stats.metadata["realworld_codec_config"]
    if rothko_config is not None and dict(rothko_config) != canonical:
        raise ValueError("Realworld codec config differs from prepared training stats")
    cfg = LiberoRothkoCodecConfig(**canonical)
    if cfg.frame0_pose_mode != "absolute":
        raise ValueError("Realworld experiment requires absolute RAY0")
    return LiberoRothkoCodec(config=cfg, norm_stats=stats, expected_action_horizon=16)


def load_context(cache_dir, instruction, context_len=128):
    prompt = PROMPT_TEMPLATE.format(task=instruction)
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    path = Path(cache_dir) / f"{digest}.pt"
    if not (Path(cache_dir) / "_SUCCESS").is_file():
        raise ValueError(f"Incomplete text cache: {cache_dir}")
    manifest = json.loads((Path(cache_dir) / "manifest.json").read_text())
    if manifest["files"].get(path.name) != sha256_file(path):
        raise ValueError(f"Text cache checksum mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = payload["metadata"]
    if any(meta.get(k) != v for k, v in manifest["identity"].items()):
        raise ValueError(f"Text encoder identity mismatch: {path}")
    if meta["prompt"] != prompt or meta["context_len"] != context_len:
        raise ValueError(f"Text cache mismatch: {path}")
    context, mask = payload["context"], payload["context_mask"]
    if context.shape != (context_len, 4096) or mask.shape != (context_len,):
        raise ValueError(f"Invalid text context shape: {path}")
    if not torch.isfinite(context).all() or not mask.bool().all():
        raise ValueError(f"Invalid encoded context/mask: {path}")
    return context, mask.bool()


class RealWorldVideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, split_manifest, rothko_norm_stats, split="train",
                 pretrained_norm_stats=None, text_embedding_cache_dir=None,
                 context_len=128, include_text_context=True, video_size=(224, 448),
                 rothko_config=None, latent_cache_dir=None, latent_cache_only=False,
                 sample_error_mode="raise"):
        if split not in ("train", "val") or sample_error_mode != "raise":
            raise ValueError("Expected train/val split and sample_error_mode=raise")
        if list(video_size) != [224, 448]:
            raise ValueError("First realworld experiment uses two 224x224 cameras")
        self.root = Path(dataset_dir)
        self.manifest = json.loads(Path(split_manifest).read_text())
        if (tuple(self.manifest["camera_keys"]) != CAMERAS or
            self.manifest["preprocessing"] != PREPROCESSING or self.manifest["horizon"] != 16):
            raise ValueError("Manifest differs from left+wrist absolute-RAY0 experiment")
        self.num_frames = 17
        self.split = split
        self.camera_keys = tuple(self.manifest["camera_keys"])
        self.threshold = self.manifest["gripper_open_threshold_m"]
        self.records = [r for r in self.manifest["episodes"] if r["split"] == split]
        self.offsets = [0]
        for record in self.records:
            self.offsets.append(self.offsets[-1] + len(record["valid_starts"]))
        if not self.records or not self.offsets[-1]:
            raise ValueError("Empty dataset split")
        self.raymap_codec = make_codec(rothko_norm_stats, rothko_config)
        self.codec = self.raymap_codec
        digest = sha256_file(split_manifest)
        if self.codec.norm_stats.metadata["split_manifest_sha256"] != digest:
            raise ValueError("Stats were fitted with a different split manifest")
        if pretrained_norm_stats is not None:
            if json.loads(Path(pretrained_norm_stats).read_text()) != self.codec.norm_stats.metadata:
                raise ValueError("Stats JSON sidecar mismatch")
        self.episode_split_metadata = {"split": split, "sha256": digest}
        self.include_text_context = include_text_context
        self.cache_dir = text_embedding_cache_dir
        self.context_len = int(context_len)
        if include_text_context and not self.cache_dir:
            raise ValueError("Training requires a prepared text cache")
        self.latent_cache_only = bool(latent_cache_only)
        if self.latent_cache_only and not latent_cache_dir:
            raise ValueError("latent_cache_only requires a cache directory")
        self.latent_cache_dataset_contract = dict(
            dataset_roots=[{"path": str(self.root.resolve()), "manifest_sha256": digest}],
            dataset_length=len(self), num_frames=17, video_size=[224, 448],
            raymap_representation="libero_rothko",  # Reuse the existing single-arm codec unchanged.
            raymap_codec_metadata=self.codec.metadata(),
            norm_stats_sha256=self.codec.norm_stats.fingerprint(),
            split=split, camera_keys=list(self.camera_keys),
            rgb_preprocessing=PREPROCESSING, pose_convention=POSE_CONVENTION,
            window_policy="full_17_frames_valid_timestamp_intervals_only",
            gripper_open_threshold_m=self.threshold)
        self.latent_cache = None
        self.latent_cache_metadata = None
        if latent_cache_dir:
            self.latent_cache = LatentCacheReader(latent_cache_dir,
                expected_dataset_contract=self.latent_cache_dataset_contract)
            self.latent_cache_metadata = self.latent_cache.metadata
        self._reset_worker_caches()

    def _reset_worker_caches(self):
        self._episode = lru_cache(maxsize=4)(self._read_episode)
        self._context = lru_cache(maxsize=32)(self._read_context)

    def __getstate__(self):
        # Bound lru_cache wrappers cannot be pickled by spawn/forkserver workers.
        state = self.__dict__.copy()
        state.pop("_episode", None)
        state.pop("_context", None)
        if self.latent_cache is not None:
            import copy
            state["latent_cache"] = copy.copy(self.latent_cache)
            state["latent_cache"]._maps = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._reset_worker_caches()

    def __len__(self):
        return self.offsets[-1]

    def _read_episode(self, record_index):
        record = self.records[record_index]
        path = self.root / record["path"]
        if sha256_file(path / "states.npz") != record["states_sha256"]:
            raise ValueError(f"Source states changed: {path}")
        return load_episode(path)

    def _read_context(self, instruction):
        return load_context(self.cache_dir, instruction, self.context_len)

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        ep = bisect.bisect_right(self.offsets, index) - 1
        record = self.records[ep]
        start = record["valid_starts"][index - self.offsets[ep]]
        pose, width, _, joints = self._episode(ep)
        poses = torch.from_numpy(pose[start:start + 17].copy())
        gripper = torch.from_numpy(gripper_open(width[start:start + 17], self.threshold)[:, None])
        # These auxiliary tensors are consumed by trainer validation, not an action head.
        state = torch.cat((torch.from_numpy(joints[start:start + 17].copy()), gripper), -1)
        result = dict(action=torch.cat((poses[1:], gripper[1:]), -1), proprio=state,
            current_endpose=poses[0], future_endpose=poses[1:], future_gripper=gripper[1:],
            prompt=PROMPT_TEMPLATE.format(task=record["instruction"]),
            image_is_pad=torch.zeros(17, dtype=torch.bool),
            raymap_is_pad=torch.zeros(17, dtype=torch.bool),
            proprio_is_pad=torch.zeros(17, dtype=torch.bool),
            action_is_pad=torch.zeros(16, dtype=torch.bool))
        if not self.latent_cache_only:
            frames = []
            for t in range(start, start + 17):
                images = {}
                for camera in self.camera_keys:
                    path = self.root / record["path"] / "image" / camera / f"{t:06d}.png"
                    with Image.open(path) as image:
                        if image.size != (224, 224):
                            raise ValueError(f"Expected preprocessed 224x224 PNG: {path}")
                        images[camera] = np.array(image.convert("RGB"))
                frames.append(rgb_canvas(images, self.camera_keys))
            result["video"] = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float().div(127.5).sub(1)
            result["raymap"] = self.codec.encode(poses, gripper)
        if self.include_text_context:
            context, mask = self._context(record["instruction"])
            result.update(context=context.clone(), context_mask=mask.clone())
        if self.latent_cache is not None:
            rgb, ray = self.latent_cache[index]
            result.update(rgb_latents=rgb, raymap_latents=ray, latent_cache_index=index)
        return result
