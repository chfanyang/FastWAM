import logging
import os
import sys
import time
import inspect
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.video_io import save_mp4

logger = logging.getLogger(__name__)


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _prepare_model_cfg(
    model_cfg: DictConfig,
    *,
    finetune_method: str,
) -> DictConfig:
    method = str(finetune_method).strip().lower()
    if method not in {"full", "lora"}:
        raise ValueError(
            f"Unsupported finetune_method={finetune_method!r}. "
            "Expected one of: ['full', 'lora']."
        )

    model_cfg_copy = OmegaConf.create(
        OmegaConf.to_container(model_cfg, resolve=True)
    )
    model_cfg_copy.load_text_encoder = True

    # Full checkpoints contain the complete DiT and may skip the pretrained
    # backbone load. Adapter-only LoRA checkpoints must be applied on top of
    # the original Wan backbone.
    if method == "lora":
        model_cfg_copy.skip_dit_load_from_pretrain = False

    rothko_stats = model_cfg_copy.get("rothko_norm_stats")
    if not _is_none_like(rothko_stats):
        stats_path = Path(
            os.path.expandvars(os.path.expanduser(str(rothko_stats)))
        )
        if not stats_path.is_absolute():
            stats_path = PROJECT_ROOT / stats_path
        stats_path = stats_path.resolve()
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Rothko normalization stats not found: {stats_path}"
            )
        model_cfg_copy.rothko_norm_stats = str(stats_path)

    return model_cfg_copy


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(
                f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}"
            ) from exc
        return relative.as_posix()

    if _is_none_like(sim_cfg_name):
        return "sim_robotwin.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
) -> DictConfig:
    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. "
            "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


class WorldActionRobotWinPolicy:
    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        finetune_method: str,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        timing_enabled: bool,
        num_video_frames: int,
        save_prediction_videos: bool,
        prediction_output_dir: Optional[Path],
    ) -> None:
        model_cfg_copy = _prepare_model_cfg(
            model_cfg,
            finetune_method=finetune_method,
        )

        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device).eval()
        self.is_visual_action_model = bool(
            getattr(self.model, "is_visual_action_model", False)
        )
        self.action_type = "ee" if self.is_visual_action_model else "qpos"

        self.processor: FastWAMProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.timing_enabled = bool(timing_enabled)
        self._num_video_frames = int(num_video_frames)
        self.save_prediction_videos = bool(save_prediction_videos)
        self.prediction_output_dir = (
            None
            if prediction_output_dir is None
            else Path(prediction_output_dir).expanduser().resolve()
        )
        if self.save_prediction_videos:
            if not self.is_visual_action_model:
                logger.warning(
                    "Prediction-video saving is only supported by the visual-action "
                    "model; disabling it for this checkpoint."
                )
                self.save_prediction_videos = False
            elif self.prediction_output_dir is None:
                raise ValueError(
                    "`prediction_output_dir` is required when "
                    "`save_prediction_videos=True`."
                )
            else:
                self.prediction_output_dir.mkdir(parents=True, exist_ok=True)

        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = 0
        self.replan_count = 0
        self.step_count = 0
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}

        logger.info(
            "Initialized WorldActionRobotWinPolicy | ckpt=%s | stats=%s | "
            "finetune=%s | horizon=%d | replan=%d | action_type=%s | "
            "save_prediction_videos=%s | prediction_output_dir=%s",
            checkpoint_path,
            dataset_stats_path,
            finetune_method,
            self.action_horizon,
            self.replan_steps,
            self.action_type,
            self.save_prediction_videos,
            self.prediction_output_dir,
        )

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _build_robotwin_image_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        obs_data = observation["observation"]
        head = _resize_rgb(obs_data["head_camera"]["rgb"], (320, 256))
        left = _resize_rgb(obs_data["left_camera"]["rgb"], (160, 128))
        right = _resize_rgb(obs_data["right_camera"]["rgb"], (160, 128))
        bottom = np.concatenate([left, right], axis=1)
        image = np.concatenate([head, bottom], axis=0)  # [384, 320, 3]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        return image_tensor

    def _build_current_raymap(
        self, task_env
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left_pose = np.asarray(
            task_env.robot.get_left_ee_pose(), dtype=np.float32
        )
        right_pose = np.asarray(
            task_env.robot.get_right_ee_pose(), dtype=np.float32
        )
        current_endpose = torch.from_numpy(
            np.concatenate((left_pose, right_pose), axis=0)
        ).unsqueeze(0)
        gripper = torch.tensor(
            [
                [
                    float(task_env.robot.get_left_gripper_val()),
                    float(task_env.robot.get_right_gripper_val()),
                ]
            ],
            dtype=torch.float32,
        )
        pose_sequence = current_endpose.unsqueeze(1)
        gripper_sequence = gripper.unsqueeze(1)
        raymap = self.model.raymap_codec.encode(
            pose_sequence, gripper_sequence
        )[:, :, 0]
        return (
            raymap.to(
                device=self.model.device,
                dtype=self.model.torch_dtype,
            ),
            current_endpose.to(
                device=self.model.device,
                dtype=torch.float32,
            ),
        )

    @staticmethod
    def _normalized_video_to_pil(video: torch.Tensor) -> list[Image.Image]:
        if video.ndim == 5:
            if video.shape[0] != 1:
                raise ValueError(
                    "Prediction-video saving expects batch size 1, got "
                    f"{tuple(video.shape)}."
                )
            video = video[0]
        if video.ndim != 4 or video.shape[0] != 3:
            raise ValueError(
                "Prediction video must be [1,3,T,H,W] or [3,T,H,W], got "
                f"{tuple(video.shape)}."
            )
        uint8 = (
            (video.detach().float().cpu().clamp(-1.0, 1.0) + 1.0)
            * 127.5
        ).round().to(torch.uint8)
        return [
            Image.fromarray(
                uint8[:, frame_index].permute(1, 2, 0).contiguous().numpy(),
                mode="RGB",
            )
            for frame_index in range(uint8.shape[1])
        ]

    def _save_visual_prediction(self, prediction: Dict[str, Any]) -> None:
        if not self.save_prediction_videos:
            return
        if self.prediction_output_dir is None:
            raise RuntimeError("Prediction output directory is not initialized.")

        episode_index = max(self.episode_count - 1, 0)
        episode_dir = (
            self.prediction_output_dir / f"episode_{episode_index:03d}"
        )
        stem = (
            f"replan_{self.replan_count:03d}"
            f"_env_step_{self.step_count:04d}"
        )

        rgb_frames = prediction.get("video")
        if not isinstance(rgb_frames, (list, tuple)) or not rgb_frames:
            raise ValueError(
                "Visual-action prediction is missing a non-empty `video` frame list."
            )
        if not all(isinstance(frame, Image.Image) for frame in rgb_frames):
            raise TypeError("Every predicted RGB frame must be a PIL image.")

        raymap = prediction.get("raymap")
        if not isinstance(raymap, torch.Tensor):
            raise ValueError(
                "Visual-action prediction is missing tensor `raymap`."
            )
        raymap_frames = self._normalized_video_to_pil(raymap)

        rgb_path = episode_dir / f"{stem}_rgb.mp4"
        raymap_path = episode_dir / f"{stem}_raymap.mp4"
        save_mp4(rgb_frames, str(rgb_path), fps=10)
        save_mp4(raymap_frames, str(raymap_path), fps=10)
        logger.info(
            "Saved predicted RGB/Rothko videos: %s | %s",
            rgb_path,
            raymap_path,
        )
        self.replan_count += 1

    def _infer_action_chunk(
        self,
        task_env,
        observation: Dict[str, Any],
        instruction: str,
    ) -> np.ndarray:
        image_tensor = self._build_robotwin_image_tensor(observation)
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self._normalize_state(state_vector)

        prompt = DEFAULT_PROMPT.format(task=instruction)
        if self.is_visual_action_model:
            input_raymap, current_endpose = self._build_current_raymap(task_env)
            infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
            with torch.no_grad():
                pred = self.model.infer(
                    prompt=prompt,
                    input_image=image_tensor,
                    input_raymap=input_raymap,
                    proprio=proprio,
                    current_endpose=current_endpose,
                    num_frames=self._num_video_frames,
                    num_inference_steps=self.num_inference_steps,
                    sigma_shift=self.sigma_shift,
                    seed=self.seed,
                    rand_device=self.rand_device,
                    tiled=self.tiled,
                )
            if self.timing_enabled:
                self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0
            self._save_visual_prediction(pred)
            action_tensor = pred.get("action")
            if not isinstance(action_tensor, torch.Tensor):
                raise ValueError(
                    "Visual-action inference did not return decoded EE `action`."
                )
            if action_tensor.ndim != 3 or action_tensor.shape[0] != 1:
                raise ValueError(
                    "Expected visual EE action [1,T,16], got "
                    f"{tuple(action_tensor.shape)}."
                )
            if action_tensor.shape[1:] != (self.action_horizon, 16):
                raise ValueError(
                    "Visual EE action horizon/dimension mismatch: expected "
                    f"(1,{self.action_horizon},16), got "
                    f"{tuple(action_tensor.shape)}."
                )
            if not torch.isfinite(action_tensor).all():
                raise ValueError("Visual EE action contains NaN or Inf.")
            return action_tensor[0].to(dtype=torch.float32, device="cpu").numpy()

        infer_kwargs = {
            "prompt": prompt,
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            infer_kwargs["num_video_frames"] = int(self._num_video_frames)
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        with torch.no_grad():
            pred = self.model.infer_action(**infer_kwargs)
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0

        action_tensor = pred["action"]  # [T, D]
        action_chunk = self._denormalize_action(action_tensor)[0]  # [T, D]
        return action_chunk

    def _fill_action_queue(
        self,
        task_env,
        observation: Dict[str, Any],
        instruction: str,
    ) -> None:
        action_chunk = self._infer_action_chunk(
            task_env=task_env,
            observation=observation,
            instruction=instruction,
        )
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        if not self.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty "
                    "(replan step for fastwam)."
                )
            instruction = task_env.get_instruction()
            self._fill_action_queue(
                task_env=task_env,
                observation=observation,
                instruction=instruction,
            )

        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type=self.action_type)
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0

    def get_timing_rollout(self) -> Dict[str, float]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
        }

    def reset(self) -> None:
        self.pending_actions.clear()
        self.episode_count += 1
        self.replan_count = 0
        self.step_count = 0
        self.reset_timing_rollout()


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    cfg = _compose_sim_cfg(
        sim_cfg_path=sim_cfg_path,
        sim_cfg_name=sim_cfg_name,
        sim_task=sim_task,
    )

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    finetune_method = str(
        usr_args.get("finetune_method")
        or cfg.get("finetune", {}).get("method", "full")
    ).strip().lower()
    if finetune_method not in {"full", "lora"}:
        raise ValueError(
            f"Unsupported finetune_method={finetune_method!r}. "
            "Expected one of: ['full', 'lora']."
        )

    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = eval_horizon if eval_horizon is not None else int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 8))

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps))

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("seed"))
    text_cfg_scale = float(usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0)))
    negative_prompt = str(usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", "")))
    rand_device = str(usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu")))
    tiled = _parse_bool(usr_args.get("tiled", cfg.EVALUATION.get("tiled", False)))
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
    )
    save_prediction_videos = _parse_bool(
        usr_args.get(
            "save_prediction_videos",
            cfg.EVALUATION.get("save_prediction_videos", False),
        )
    )
    prediction_output_dir = None
    if save_prediction_videos:
        eval_output_dir = usr_args.get("eval_output_dir")
        if _is_none_like(eval_output_dir):
            raise ValueError(
                "`eval_output_dir` is required when saving prediction videos."
            )
        task_config = str(usr_args.get("task_config") or "unknown")
        if Path(task_config).name != task_config:
            raise ValueError(f"Invalid task_config path component: {task_config!r}")
        prediction_output_dir = (
            Path(str(eval_output_dir)).expanduser().resolve()
            / "predictions"
            / task_config
        )

    policy = WorldActionRobotWinPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        finetune_method=finetune_method,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        timing_enabled=timing_enabled,
        num_video_frames=(int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1,
        save_prediction_videos=save_prediction_videos,
        prediction_output_dir=prediction_output_dir,
    )
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()
