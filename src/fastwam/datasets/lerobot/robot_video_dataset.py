import hashlib
import os
from typing import Optional
import time
import numpy as np
import traceback
import torch
from contextlib import contextmanager

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .episode_splits import load_grouped_episode_split
from .robotwin_tasks import resolve_robotwin_episode_indices
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from ..robotwin_rgb import build_robotwin_rgb_canvas
from ..libero_rgb import build_libero_rgb_canvas
from ..latent_cache import LatentCacheReader, build_dataset_contract
from fastwam.utils.logging_config import get_logger
from fastwam.utils import misc, pytorch_utils
from fastwam.representations.rothko import RothkoCodec, RothkoCodecConfig
from fastwam.representations.libero_rothko import (
    LiberoRothkoCodec,
    LiberoRothkoCodecConfig,
)
from accelerate import PartialState
logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def resolve_libero_future_gripper(
    raw_action: dict,
    *,
    action_horizon: int,
    explicit_key: Optional[str] = None,
) -> torch.Tensor:
    """Return future LIBERO gripper targets in ``0=closed,1=open`` form.

    Original LIBERO already stores that convention in the final dimension of
    ``action``.  LIBERO-Plus keeps its environment command in ``action`` and
    provides an explicit converted side channel.  Keeping the selection behind
    an opt-in key preserves every existing LIBERO configuration exactly.
    """
    if explicit_key is None:
        default_action = raw_action.get("default")
        if default_action is None:
            raise ValueError("Missing raw_action.default for LIBERO gripper targets.")
        if default_action.ndim != 2 or default_action.shape[0] != action_horizon:
            raise ValueError(
                "Expected raw_action.default with shape "
                f"[{action_horizon},D], got {tuple(default_action.shape)}."
            )
        return default_action[:, -1:].float().clamp(0, 1)

    explicit_gripper = raw_action.get(explicit_key)
    if explicit_gripper is None:
        raise ValueError(
            f"Missing raw_action.{explicit_key} configured as the explicit "
            "LIBERO gripper target."
        )
    if explicit_gripper.shape != (action_horizon, 1):
        raise ValueError(
            f"Expected raw_action.{explicit_key} {(action_horizon, 1)}, "
            f"got {tuple(explicit_gripper.shape)}."
        )
    explicit_gripper = explicit_gripper.float()
    if not bool(torch.isfinite(explicit_gripper).all().item()):
        raise ValueError(f"raw_action.{explicit_key} contains non-finite values.")
    if bool(((explicit_gripper < 0.0) | (explicit_gripper > 1.0)).any().item()):
        raise ValueError(
            f"raw_action.{explicit_key} must use 0=closed,1=open in [0,1]."
        )
    return explicit_gripper

class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        robotwin_task_names=None,
        robotwin_data_variant: str = "all",
        robotwin_ee_pose_key: str = "endpose",
        episode_split_manifest=None,
        episode_split: Optional[str] = None,
        raw_action_meta=None,
        raw_state_meta=None,
        raymap_representation: Optional[str] = None,
        libero_action_gripper_key: Optional[str] = None,
        rothko_norm_stats: Optional[str] = None,
        rothko_config=None,
        sample_error_mode: str = "fallback",
        latent_cache_dir: Optional[str] = None,
        latent_cache_only: bool = False,
        text_context_cache_max_entries: Optional[int] = None,
    ):
        dataset_dirs = [str(path) for path in dataset_dirs]
        self.dataset_dirs = dataset_dirs
        self.robotwin_ee_pose_key = str(robotwin_ee_pose_key)
        if self.robotwin_ee_pose_key not in {"endpose", "ee_pose_wxyz"}:
            raise ValueError("robotwin_ee_pose_key must be endpose or ee_pose_wxyz")
        self.latent_cache_only = bool(latent_cache_only)
        if self.latent_cache_only and latent_cache_dir in (None, "", "null"):
            raise ValueError("`latent_cache_only=true` requires `latent_cache_dir`.")
        self.sample_error_mode = str(sample_error_mode)
        if self.sample_error_mode not in {"fallback", "raise"}:
            raise ValueError(
                "`sample_error_mode` must be 'fallback' or 'raise', got "
                f"{self.sample_error_mode!r}."
            )
        episode_indices = None
        if robotwin_data_variant not in {"all", "clean", "randomized"}:
            raise ValueError(f"Unknown RoboTwin data variant: {robotwin_data_variant!r}")
        if robotwin_task_names is not None or robotwin_data_variant != "all":
            if robotwin_task_names is None:
                from .robotwin_tasks import ROBOTWIN_TASK_NAMES
                robotwin_task_names = list(ROBOTWIN_TASK_NAMES)
            robotwin_task_names = [str(name) for name in robotwin_task_names]
            episode_indices = resolve_robotwin_episode_indices(robotwin_task_names, robotwin_data_variant)
            logger.info(
                "Selecting %d RoboTwin tasks (%d episodes): %s",
                len(set(robotwin_task_names)),
                len(episode_indices),
                ", ".join(robotwin_task_names),
            )
        self.episode_split_metadata = None
        if episode_split_manifest is not None:
            if robotwin_task_names is not None:
                raise ValueError(
                    "`episode_split_manifest` and `robotwin_task_names` are mutually exclusive."
                )
            if len(dataset_dirs) != 1:
                raise ValueError(
                    "Grouped episode manifests currently require exactly one dataset root."
                )
            if val_set_proportion >= 1e-6:
                raise ValueError(
                    "Set `val_set_proportion=0` when using an explicit episode split manifest."
                )
            if episode_split is None:
                raise ValueError(
                    "`episode_split` is required when `episode_split_manifest` is set."
                )
            expected_training_flag = str(episode_split) == "train"
            if bool(is_training_set) != expected_training_flag:
                raise ValueError(
                    "`is_training_set` is inconsistent with explicit episode split: "
                    f"split={episode_split!r}, is_training_set={is_training_set}."
                )
            episode_indices, self.episode_split_metadata = load_grouped_episode_split(
                episode_split_manifest,
                str(episode_split),
            )
            logger.info(
                "Using grouped episode split: path=%s sha256=%s split=%s "
                "episodes=%d source_trajectories=%d tasks=%d",
                self.episode_split_metadata["path"],
                self.episode_split_metadata["sha256"],
                self.episode_split_metadata["split"],
                self.episode_split_metadata["episodes"],
                self.episode_split_metadata["source_trajectories"],
                self.episode_split_metadata["tasks"],
            )
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            episode_indices=episode_indices,
            raw_action_meta=None if self.latent_cache_only else raw_action_meta,
            raw_state_meta=None if self.latent_cache_only else raw_state_meta,
            sample_error_mode=self.sample_error_mode,
        )
    
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(not self.latent_cache_only)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self._warned_legacy_text_cache = False
        # None preserves historical vocabulary-sized caching. Large-vocabulary
        # datasets can opt into a per-worker LRU bound; zero disables caching.
        if text_context_cache_max_entries is not None and text_context_cache_max_entries < 0:
            raise ValueError("text_context_cache_max_entries must be non-negative or None")
        self.text_context_cache_max_entries = text_context_cache_max_entries
        self._text_context_memory_cache = {}
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.latent_cache_dir = latent_cache_dir
        self.latent_cache = None
        self.latent_cache_metadata = None
        self.raymap_representation = raymap_representation
        self.libero_action_gripper_key = (
            None
            if libero_action_gripper_key in (None, "", "null")
            else str(libero_action_gripper_key)
        )
        if (
            self.libero_action_gripper_key is not None
            and raymap_representation != "libero_rothko"
        ):
            raise ValueError(
                "`libero_action_gripper_key` is only valid with "
                "`raymap_representation=libero_rothko`."
            )
        self.raymap_codec = None
        if raymap_representation is not None:
            if raymap_representation not in ("rothko", "libero_rothko"):
                raise ValueError(
                    "`raymap_representation` must be one of "
                    "['rothko', 'libero_rothko'], "
                    f"got {raymap_representation!r}."
                )
            if action_video_freq_ratio != 1:
                raise ValueError(
                    "Video-only Rothko training requires RGB/action frequency ratio 1, "
                    f"got {action_video_freq_ratio}."
                )
            if rothko_norm_stats is None:
                raise ValueError("`rothko_norm_stats` is required for Rothko training.")
            if isinstance(rothko_config, DictConfig):
                rothko_config = OmegaConf.to_container(rothko_config, resolve=True)
            if rothko_config is None:
                rothko_config = {}
            if not isinstance(rothko_config, dict):
                raise ValueError(
                    "`rothko_config` must be dict-like, got "
                    f"{type(rothko_config)}."
                )
            if raymap_representation == "rothko":
                codec_config_kwargs = {
                    "image_height": int(video_size[0]),
                    "image_width": int(video_size[1]),
                    **rothko_config,
                }
                codec_config = RothkoCodecConfig(**codec_config_kwargs)
                self.raymap_codec = RothkoCodec(
                    config=codec_config,
                    norm_stats=rothko_norm_stats,
                )
            else:
                codec_config_kwargs = {
                    "image_height": int(video_size[0]),
                    "image_width": int(video_size[1]),
                    "tile_height": int(video_size[0]),
                    "tile_width": int(video_size[1]) // 2,
                    **rothko_config,
                }
                codec_config = LiberoRothkoCodecConfig(**codec_config_kwargs)
                self.raymap_codec = LiberoRothkoCodec(
                    config=codec_config,
                    norm_stats=rothko_norm_stats,
                    expected_action_horizon=self.num_frames - 1,
                )

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if self.latent_cache_only:
                processor.set_process_images(False)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)

        norm_stats = (
            None
            if self.raymap_codec is None
            else getattr(self.raymap_codec, "norm_stats", None)
        )
        self.latent_cache_dataset_contract = build_dataset_contract(
            dataset_dirs=self.dataset_dirs,
            dataset_length=len(self.lerobot_dataset),
            num_frames=self.num_frames,
            video_size=list(self.video_size),
            raymap_representation=self.raymap_representation,
            raymap_codec_metadata=(
                None if self.raymap_codec is None else self.raymap_codec.metadata()
            ),
            norm_stats_sha256=(
                None if norm_stats is None else norm_stats.fingerprint()
            ),
        )
        if self.raymap_representation == "rothko" and self.robotwin_ee_pose_key != "endpose":
            self.latent_cache_dataset_contract["robotwin_ee_pose_key"] = self.robotwin_ee_pose_key
        if latent_cache_dir not in (None, "", "null"):
            self.latent_cache = LatentCacheReader(
                latent_cache_dir,
                expected_dataset_contract=self.latent_cache_dataset_contract,
            )
            if len(self.latent_cache) != len(self.lerobot_dataset):
                raise ValueError(
                    "Latent cache length mismatch: "
                    f"cache={len(self.latent_cache)}, "
                    f"dataset={len(self.lerobot_dataset)}."
                )
            self.latent_cache_metadata = self.latent_cache.metadata
            logger.info(
                "Using precomputed RGB/Rothko latent cache: %s (%d samples)",
                self.latent_cache.cache_dir,
                len(self.latent_cache),
            )
        
    def __len__(self):
        return len(self.lerobot_dataset)

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]
            # BaseLerobotDataset may replace an unreadable sample in fallback
            # mode. Keep every downstream side channel and latent-cache lookup
            # aligned to the sample that was actually returned.
            sample_idx = int(sample.get("idx", sample_idx))

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))

        if self.latent_cache_only:
            return self._get_latent_cache_only(sample_idx, sample)
        
        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            video = video[:, self.video_sample_indices, :, :, :] # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            video = video[self.video_sample_indices, :, :, :] # [T_video, C, H, W]
            T_video, C, H, W = video.shape
        image_is_pad = image_is_pad[self.video_sample_indices]

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            video = build_robotwin_rgb_canvas(
                head=video[0],
                left_wrist=video[1],
                right_wrist=video[2],
            )
        elif self.raymap_representation == "libero_rothko":
            if num_cameras != 2 or self.concat_multi_camera != "horizontal":
                raise ValueError(
                    "LIBERO Rothko requires two horizontally concatenated cameras, "
                    f"got num_cameras={num_cameras}, concat={self.concat_multi_camera!r}."
                )
            video = build_libero_rgb_canvas(
                agentview=video[0],
                wrist=video[1],
                camera_height=int(self.video_size[0]),
                camera_width=int(self.video_size[1]) // 2,
            )
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]

        if (
            self.concat_multi_camera != "robotwin"
            and self.raymap_representation != "libero_rothko"
        ):
            # The shared RoboTwin builder already returns the exact target
            # shape and applies the same [-1, 1] normalization.
            video = self.resize_transform(video)
            video = self.crop_transform(video)
            video = self.normalize_transform(video)  # [T_video, C, H, W]

        video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )

        task = sample["instruction"]
        
        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(instruction)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)

        raymap_video = None
        raymap_is_pad = None
        current_endpose = None
        future_endpose = None
        future_gripper = None
        if self.raymap_codec is not None:
            raw_action = sample.get("raw_action") or {}
            raw_state = sample.get("raw_state") or {}
            if self.raymap_representation == "rothko":
                required_raw = {
                    "raw_action.default": raw_action.get("default"),
                    f"raw_action.{self.robotwin_ee_pose_key}": raw_action.get(self.robotwin_ee_pose_key),
                    "raw_state.default": raw_state.get("default"),
                    f"raw_state.{self.robotwin_ee_pose_key}": raw_state.get(self.robotwin_ee_pose_key),
                }
            else:
                required_raw = {
                    "raw_action.default": raw_action.get("default"),
                    "raw_action.osc_target_pose_wxyz": raw_action.get(
                        "osc_target_pose_wxyz"
                    ),
                    "raw_state.ee_pose_wxyz": raw_state.get("ee_pose_wxyz"),
                    "raw_state.gripper_open": raw_state.get("gripper_open"),
                }
                if self.libero_action_gripper_key is not None:
                    required_raw[
                        f"raw_action.{self.libero_action_gripper_key}"
                    ] = raw_action.get(self.libero_action_gripper_key)
            missing = [name for name, value in required_raw.items() if value is None]
            if missing:
                raise ValueError(
                    f"{self.raymap_representation} dataset is missing raw side-channel fields: "
                    + ", ".join(missing)
                )
            if self.raymap_representation == "rothko":
                raw_action_qpos = raw_action["default"].float()
                raw_state_qpos = raw_state["default"].float()
                action_endpose = raw_action[self.robotwin_ee_pose_key].float()
                state_endpose = raw_state[self.robotwin_ee_pose_key].float()
                if action_endpose.shape != (self.num_frames - 1, 14):
                    raise ValueError(
                        f"Expected action endpose {(self.num_frames - 1, 14)}, "
                        f"got {tuple(action_endpose.shape)}."
                    )
                if state_endpose.shape != (self.num_frames, 14):
                    raise ValueError(
                        f"Expected state endpose {(self.num_frames, 14)}, "
                        f"got {tuple(state_endpose.shape)}."
                    )
                current_endpose = state_endpose[0]
                future_endpose = action_endpose
                current_gripper = raw_state_qpos[0, [6, 13]]
                future_gripper = raw_action_qpos[:, [6, 13]]
            else:
                action_target = raw_action["osc_target_pose_wxyz"].float()
                state_pose = raw_state["ee_pose_wxyz"].float()
                state_gripper = raw_state["gripper_open"].float()
                if action_target.shape != (self.num_frames - 1, 7):
                    raise ValueError(
                        "Expected LIBERO OSC target "
                        f"{(self.num_frames - 1, 7)}, got {tuple(action_target.shape)}."
                    )
                if state_pose.shape != (self.num_frames, 7):
                    raise ValueError(
                        f"Expected LIBERO state pose {(self.num_frames, 7)}, "
                        f"got {tuple(state_pose.shape)}."
                    )
                if state_gripper.shape != (self.num_frames, 1):
                    raise ValueError(
                        f"Expected LIBERO state gripper {(self.num_frames, 1)}, "
                        f"got {tuple(state_gripper.shape)}."
                    )
                current_endpose = state_pose[0]
                future_endpose = action_target
                current_gripper = state_gripper[0]
                future_gripper = resolve_libero_future_gripper(
                    raw_action,
                    action_horizon=self.num_frames - 1,
                    explicit_key=self.libero_action_gripper_key,
                )

            pose_sequence = torch.cat(
                (current_endpose.unsqueeze(0), future_endpose), dim=0
            )
            gripper_sequence = torch.cat(
                (current_gripper.unsqueeze(0), future_gripper), dim=0
            )
            raymap_video = self.raymap_codec.encode(
                pose_sequence, gripper_sequence
            )
            raymap_is_pad = torch.cat(
                (
                    sample["proprio_is_pad"][:1].bool(),
                    sample["action_is_pad"].bool(),
                ),
                dim=0,
            )

        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        if raymap_video is not None:
            data.update(
                {
                    "raymap": raymap_video,
                    "raymap_is_pad": raymap_is_pad,
                    "current_endpose": current_endpose,
                    "future_endpose": future_endpose,
                    "future_gripper": future_gripper,
                }
            )
        if self.latent_cache is not None:
            rgb_latents, raymap_latents = self.latent_cache[sample_idx]
            data.update(
                {
                    "rgb_latents": rgb_latents,
                    "raymap_latents": raymap_latents,
                    "latent_cache_index": sample_idx,
                }
            )
        return data

    def _get_latent_cache_only(self, sample_idx: int, sample: dict):
        if self.latent_cache is None:
            raise RuntimeError("Latent-cache-only dataset has no initialized cache.")
        task = sample["instruction"]
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)
        context, context_mask = self._get_cached_text_context(instruction)
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)

        image_is_pad = sample["image_is_pad"][self.video_sample_indices].bool()
        action_is_pad = sample["action_is_pad"].bool()
        proprio_is_pad = sample["proprio_is_pad"].bool()
        raymap_is_pad = torch.cat(
            (proprio_is_pad[:1], action_is_pad), dim=0
        )
        rgb_latents, raymap_latents = self.latent_cache[sample_idx]
        return {
            "action": sample["action"],
            "proprio": sample["proprio"][:-1],
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": action_is_pad,
            "proprio_is_pad": proprio_is_pad,
            "raymap_is_pad": raymap_is_pad,
            "rgb_latents": rgb_latents,
            "raymap_latents": raymap_latents,
            "latent_cache_index": sample_idx,
        }

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        memory_cached = self._text_context_memory_cache.get(cache_path)
        if memory_cached is not None:
            if self.text_context_cache_max_entries is not None:
                self._text_context_memory_cache.pop(cache_path)
                self._text_context_memory_cache[cache_path] = memory_cached
            return memory_cached
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        cache_metadata = payload.get("cache_metadata")
        if cache_metadata is None:
            if not self._warned_legacy_text_cache:
                logger.warning(
                    "Legacy text embedding cache has no identity metadata: %s. "
                    "Regenerate it before changing the text encoder/tokenizer.",
                    cache_path,
                )
                self._warned_legacy_text_cache = True
        else:
            expected_cache_metadata = {
                "prompt_sha256": hashed,
                "context_len": self.context_len,
                "encoder_id": "wan22ti2v5b",
            }
            for key, expected in expected_cache_metadata.items():
                actual = cache_metadata.get(key)
                if actual != expected:
                    raise ValueError(
                        f"Text embedding cache metadata mismatch for {key}: "
                        f"cache={actual!r}, expected={expected!r}, path={cache_path}."
                    )
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        limit = self.text_context_cache_max_entries
        if limit != 0:
            self._text_context_memory_cache[cache_path] = (context, context_mask)
            if limit is not None and len(self._text_context_memory_cache) > limit:
                self._text_context_memory_cache.pop(next(iter(self._text_context_memory_cache)))
        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            if self.sample_error_mode == "raise":
                raise RuntimeError(f"Error processing sample idx {idx}") from e
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
