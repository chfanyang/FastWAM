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
from .robotwin_tasks import resolve_robotwin_episode_indices
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from ..robotwin_rgb import build_robotwin_rgb_canvas
from fastwam.utils.logging_config import get_logger
from fastwam.utils import misc, pytorch_utils
from fastwam.representations.rothko import RothkoCodec, RothkoCodecConfig
from accelerate import PartialState
logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

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
        raw_action_meta=None,
        raw_state_meta=None,
        raymap_representation: Optional[str] = None,
        rothko_norm_stats: Optional[str] = None,
    ):
        episode_indices = None
        if robotwin_task_names is not None:
            robotwin_task_names = [str(name) for name in robotwin_task_names]
            episode_indices = resolve_robotwin_episode_indices(robotwin_task_names)
            logger.info(
                "Selecting %d RoboTwin tasks (%d episodes): %s",
                len(set(robotwin_task_names)),
                len(episode_indices),
                ", ".join(robotwin_task_names),
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
            raw_action_meta=raw_action_meta,
            raw_state_meta=raw_state_meta,
        )
    
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.raymap_representation = raymap_representation
        self.raymap_codec = None
        if raymap_representation is not None:
            if raymap_representation != "rothko":
                raise ValueError(
                    "Only raymap_representation='rothko' is implemented in the first version, "
                    f"got {raymap_representation!r}."
                )
            if action_video_freq_ratio != 1:
                raise ValueError(
                    "Video-only Rothko training requires RGB/action frequency ratio 1, "
                    f"got {action_video_freq_ratio}."
                )
            if rothko_norm_stats is None:
                raise ValueError("`rothko_norm_stats` is required for Rothko training.")
            codec_config = RothkoCodecConfig(
                image_height=int(video_size[0]),
                image_width=int(video_size[1]),
            )
            self.raymap_codec = RothkoCodec(
                config=codec_config,
                norm_stats=rothko_norm_stats,
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
        
    def __len__(self):
        return len(self.lerobot_dataset)

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

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

        if self.concat_multi_camera != "robotwin":
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
            required_raw = {
                "raw_action.default": raw_action.get("default"),
                "raw_action.endpose": raw_action.get("endpose"),
                "raw_state.default": raw_state.get("default"),
                "raw_state.endpose": raw_state.get("endpose"),
            }
            missing = [name for name, value in required_raw.items() if value is None]
            if missing:
                raise ValueError(
                    "Rothko dataset is missing raw side-channel fields: "
                    + ", ".join(missing)
                )
            raw_action_qpos = raw_action["default"].float()
            raw_state_qpos = raw_state["default"].float()
            action_endpose = raw_action["endpose"].float()
            state_endpose = raw_state["endpose"].float()
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
            pose_sequence = torch.cat((current_endpose.unsqueeze(0), future_endpose), dim=0)
            current_gripper = raw_state_qpos[0, [6, 13]]
            future_gripper = raw_action_qpos[:, [6, 13]]
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
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
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

        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
