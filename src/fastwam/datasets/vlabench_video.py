"""Opt-in VLABench RGB/Rothko samples; no changes to existing benchmarks."""
import hashlib
import json
from pathlib import Path

import torch

from .vlabench_image import VLABenchImageWindowDataset
from .libero_rgb import build_libero_rgb_canvas
from .latent_cache import LatentCacheReader, build_dataset_contract
from .vlabench_rgb import CAMERA_KEYS, build_vlabench_rgb_canvas
from fastwam.representations.vlabench_rothko import VLABenchRothkoCodec, VLABenchRothkoCodecConfig


def pose_xyz_euler_to_wxyz(value):
    """Export uses extrinsic xyz Euler radians (Rz @ Ry @ Rx)."""
    x, y, z = (value[..., 3:6] / 2).unbind(-1)
    cx, cy, cz = x.cos(), y.cos(), z.cos()
    sx, sy, sz = x.sin(), y.sin(), z.sin()
    quaternion = torch.stack((cx*cy*cz + sx*sy*sz,
                              sx*cy*cz - cx*sy*sz,
                              cx*sy*cz + sx*cy*sz,
                              cx*cy*sz - sx*sy*cz), -1)
    return torch.cat((value[..., :3], quaternion), -1)


class VLABenchVideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, split_manifest, rothko_norm_stats,
                 split="train", text_embedding_cache_dir=None, context_len=128,
                 include_text_context=True, episode_cache_size=2,
                 rothko_config=None, pretrained_norm_stats=None,
                 video_size=(224, 448), latent_cache_dir=None,
                 latent_cache_only=False, sample_error_mode="raise", task_names=None,
                 camera_layout="two_camera", cache_scope="full_split",
                 raymap_representation="vlabench_rothko"):
        self.three_camera = camera_layout == "three_camera_192"
        if camera_layout not in {"two_camera", "three_camera_192"}:
            raise ValueError("Unknown VLABench camera layout")
        if list(video_size) != ([192, 576] if self.three_camera else [224, 448]):
            raise ValueError("VLABench canvas size does not match camera layout")
        if cache_scope not in {"full_split", "task"}:
            raise ValueError("Unknown VLABench cache scope")
        if cache_scope == "task" and (task_names is None or len(task_names) != 1):
            raise ValueError("Task-local cache requires exactly one task")
        if sample_error_mode != "raise":
            raise ValueError("VLABench currently requires sample_error_mode=raise")
        if latent_cache_only and not latent_cache_dir:
            raise ValueError("latent_cache_only requires a complete cache")
        self.latent_cache_only = bool(latent_cache_only)
        self.sample_indices = None
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        manifest = json.loads(Path(split_manifest).read_text())
        episode_ids = manifest[f"{split}_episode_indices"]
        if cache_scope == "task":
            records = {r["episode_index"]: r for r in manifest["records"]}
            episode_ids = [e for e in episode_ids if records[e]["base_task"] == task_names[0]]
            if not episode_ids:
                raise ValueError("Empty task-local dataset")
        self.windows = VLABenchImageWindowDataset(
            dataset_dir, episode_indices=episode_ids,
            camera_keys=CAMERA_KEYS if self.three_camera else ("image", "wrist_image"),
            lazy=True, episode_cache_size=episode_cache_size,
            return_images=not self.latent_cache_only)
        self.num_frames = 17
        codec_class = VLABenchRothkoCodec
        if raymap_representation == "vlabench_rothko_all_absolute":
            from fastwam.representations.vlabench_rothko_all_absolute import VLABenchAllAbsoluteRothkoCodec
            codec_class = VLABenchAllAbsoluteRothkoCodec
        elif raymap_representation != "vlabench_rothko":
            raise ValueError("Unsupported VLABench raymap representation")
        self.codec = codec_class(
                                        config=VLABenchRothkoCodecConfig(**dict(rothko_config or {})),
                                        norm_stats=rothko_norm_stats,
                                        expected_action_horizon=16)
        self.raymap_codec = self.codec
        if self.three_camera != (getattr(self.codec.config, "horizontal_copies", 2) == 3):
            raise ValueError("RGB and Rothko tile layouts must match")
        digest = hashlib.sha256(Path(split_manifest).read_bytes()).hexdigest()
        self.episode_split_metadata = {"split": split, "sha256": digest}
        if self.codec.norm_stats.metadata.get("split_manifest_sha256") != digest:
            raise ValueError("VLABench stats were fitted with a different split manifest")
        # Shared runtime records this JSON's identity in the checkpoint. It is
        # the Rothko metadata sidecar, NOT an additional action normalizer.
        if pretrained_norm_stats is not None:
            metadata = json.loads(Path(pretrained_norm_stats).read_text())
            if metadata != self.codec.norm_stats.metadata:
                raise ValueError("VLABench stats JSON must match the Rothko tensor metadata")
        self.include_text_context = include_text_context
        self.cache_dir = None if text_embedding_cache_dir is None else Path(text_embedding_cache_dir)
        self.context_len = context_len
        if include_text_context and self.cache_dir is None:
            raise ValueError("Training requires a real text_embedding_cache_dir")
        self.latent_cache_dataset_contract = build_dataset_contract(
            dataset_dirs=[str(dataset_dir)], dataset_length=len(self), num_frames=17,
            video_size=list(video_size), raymap_representation=self.codec.representation,
            raymap_codec_metadata=self.codec.metadata(),
            norm_stats_sha256=self.codec.norm_stats.fingerprint())
        self.latent_cache_dataset_contract.update(
            split=split, split_manifest_sha256=digest,
            rgb_preprocessing="front_wrist_bilinear_antialias_224_v1",
            pose_convention="state0_actions0to15_xyz_euler_xyz_to_wxyz_v1",
            gripper_convention="state_1minus_raw_action_raw_v1")
        if self.three_camera:
            self.latent_cache_dataset_contract["rgb_preprocessing"] = "image_second_image_wrist_bilinear_antialias_192_v1"
        if cache_scope == "task":
            self.latent_cache_dataset_contract.update(cache_scope="task", task_names=list(task_names), episode_indices=episode_ids)
        self.latent_cache = None
        self.latent_cache_metadata = None
        if latent_cache_dir:
            self.latent_cache = LatentCacheReader(latent_cache_dir,
                expected_dataset_contract=self.latent_cache_dataset_contract)
            self.latent_cache_metadata = self.latent_cache.metadata
        # Map subset indices AFTER validating the full split cache contract.
        if task_names is not None and cache_scope == "full_split":
            records = {r["episode_index"]: r for r in manifest["records"]}
            selected = set(task_names)
            available = {records[e]["base_task"] for e in manifest[f"{split}_episode_indices"]}
            if not selected or not selected <= available:
                raise ValueError(f"Unknown/empty VLABench task selection: {selected}")
            self.sample_indices = []
            offset = 0
            for episode in manifest[f"{split}_episode_indices"]:
                record = records[episode]
                if record["base_task"] in selected:
                    self.sample_indices.extend(range(offset, offset+record["length"]))
                offset += record["length"]

    def __len__(self):
        return len(self.windows) if self.sample_indices is None else len(self.sample_indices)

    def __getitem__(self, index):
        if self.sample_indices is not None:
            index = self.sample_indices[index]
        sample = self.windows[index]
        state, action = sample["raw_state"]["default"], sample["raw_action"]["default"]
        current = pose_xyz_euler_to_wxyz(state[0])
        future = pose_xyz_euler_to_wxyz(action)
        poses = torch.cat((current[None], future))
        gripper = torch.cat((sample["raw_state"]["gripper_open"][:1],
                             sample["raw_action"]["gripper_open"]))
        prompt = "A video recorded from a robot's point of view executing the following instruction: " + sample["task"]
        result = dict(action=action, proprio=state, prompt=prompt,
                      current_endpose=current, future_endpose=future,
                      future_gripper=gripper[1:],
                      image_is_pad=sample["image_is_pad"],
                      action_is_pad=sample["action_is_pad"],
                      proprio_is_pad=sample["state_is_pad"],
                      raymap_is_pad=torch.cat((sample["state_is_pad"][:1], sample["action_is_pad"])))
        if not self.latent_cache_only:
            video = (build_vlabench_rgb_canvas(sample["images"]) if self.three_camera else
                     build_libero_rgb_canvas(sample["images"]["image"], sample["images"]["wrist_image"]))
            result.update(video=video.permute(1,0,2,3).contiguous(),
                          raymap=self.codec.encode(poses,gripper))
        if self.include_text_context:
            digest = hashlib.sha256(prompt.encode()).hexdigest()
            path = self.cache_dir / f"{digest}.t5_len{self.context_len}.wan22ti2v5b.pt"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            expected = dict(prompt_sha256=digest, context_len=self.context_len,
                            encoder_id="wan22ti2v5b")
            metadata = payload.get("cache_metadata", {})
            if any(metadata.get(k) != v for k, v in expected.items()):
                raise ValueError(f"Text cache metadata mismatch: {path}")
            context, mask = payload["context"].clone(), payload["mask"].bool()
            if context.ndim != 2 or context.shape[0] != self.context_len or mask.shape != (self.context_len,):
                raise ValueError(f"Text cache shape mismatch: {path}")
            context[~mask] = 0
            result.update(context=context, context_mask=torch.ones_like(mask))
        if self.latent_cache is not None:
            rgb, ray = self.latent_cache[index]
            result.update(rgb_latents=rgb, raymap_latents=ray, latent_cache_index=index)
        return result
