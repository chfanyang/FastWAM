import json
import inspect
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image
from tqdm import tqdm

# try:
#     import rootutils

#     rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)
# except ModuleNotFoundError:
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    quat2axisangle,
    save_model_prediction_video,
    save_prediction_video,
    save_rollout_video,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.libero_rgb import build_libero_rgb_canvas
from fastwam.representations.libero_osc import (
    absolute_target_to_normalized_action,
    panda_gripper_qpos_to_open,
)
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from libero.libero import benchmark

try:
    from .action_ensembler import ActionEnsembler, AbsolutePoseActionEnsembler
except ImportError:  # Direct execution: python experiments/libero/eval_libero_single.py
    from action_ensembler import ActionEnsembler, AbsolutePoseActionEnsembler

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


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


def _resolve_eval_device(cfg: DictConfig) -> str:
    eval_device = cfg.EVALUATION.get("device")
    if eval_device is not None:
        return str(eval_device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates: list[Path] = []

    if explicit is not None:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))

    ckpt = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
    for parent in list(ckpt.parents)[:4]:
        candidates.append(parent / "dataset_stats.json")

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    msg = (
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )
    raise FileNotFoundError(msg)


def _load_model_checkpoint(model: torch.nn.Module, ckpt: str) -> None:
    model.load_checkpoint(ckpt)
    logging.info("Loaded checkpoint via model.load_checkpoint: %s", ckpt)


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _normalize_proprio(
    proprio: np.ndarray,
    processor: FastWAMProcessor,
) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged state key in shape_meta['state']."
        )
    state_key = state_meta[0]["key"]

    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def _obs_to_model_input(
    obs: dict,
    cfg: DictConfig,
    processor: FastWAMProcessor,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
):
    imgs = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    if len(image_meta) < int(processor.num_output_cameras):
        raise ValueError(
            f"shape_meta.images has {len(image_meta)} entries, "
            f"but num_output_cameras={processor.num_output_cameras}."
        )

    def _meta_to_hw(meta: dict, camera_idx: int) -> tuple[int, int]:
        shape = meta["shape"]
        if len(shape) != 3:
            raise ValueError(f"shape_meta.images[{camera_idx}].shape must be [C,H,W], got {shape}")
        return int(shape[1]), int(shape[2])

    concatenation = cfg.data.train.get("concat_multi_camera", "horizontal")
    num_cameras = processor.num_output_cameras
    is_visual_action = (
        str(cfg.data.train.get("raymap_representation", ""))
        == "libero_rothko"
    )
    if is_visual_action:
        if num_cameras != 2 or concatenation != "horizontal":
            raise ValueError(
                "LIBERO Rothko eval requires two horizontal cameras, got "
                f"num_cameras={num_cameras}, concat={concatenation!r}."
            )
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        wrist_h, wrist_w = _meta_to_hw(image_meta[1], camera_idx=1)
        if (primary_h, primary_w) != (wrist_h, wrist_w):
            raise ValueError(
                "LIBERO Rothko requires equal per-camera shapes, got "
                f"{(primary_h, primary_w)} and {(wrist_h, wrist_w)}."
            )
        rgb_tensor = build_libero_rgb_canvas(
            imgs["image"],
            imgs["wrist_image"],
            camera_height=primary_h,
            camera_width=primary_w,
        )
        rgb = None
    elif num_cameras == 1:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        rgb = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
    elif num_cameras == 2:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        wrist_h, wrist_w = _meta_to_hw(image_meta[1], camera_idx=1)
        primary = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
        wrist = _center_crop_resize(imgs["wrist_image"], width=wrist_w, height=wrist_h)
        if concatenation == "horizontal":
            rgb = np.concatenate([primary, wrist], axis=1)
        elif concatenation == "vertical":
            rgb = np.concatenate([primary, wrist], axis=0)
        else:
            raise ValueError(f"Invalid concat_multi_camera: {concatenation}")
    else:
        raise ValueError(f"LIBERO eval currently supports num_output_cameras in [1, 2], got {num_cameras}.")

    if is_visual_action:
        actual_h, actual_w = int(rgb_tensor.shape[-2]), int(rgb_tensor.shape[-1])
    else:
        actual_h, actual_w = int(rgb.shape[0]), int(rgb.shape[1])
    expected_h, expected_w = int(height), int(width)
    image_shapes = [meta["shape"] for meta in image_meta]
    assert actual_h == expected_h and actual_w == expected_w, (
        "Input image size mismatch after per-camera resize + concat: "
        f"got (H,W)=({actual_h},{actual_w}), expected (H,W)=({expected_h},{expected_w}) "
        f"from data.train.video_size={[expected_h, expected_w]}; "
        f"shape_meta.images={image_shapes}, concat_multi_camera={concatenation}."
    )

    if is_visual_action:
        x = rgb_tensor.unsqueeze(0).to(device=device, dtype=dtype)
    else:
        x = (
            torch.tensor(rgb)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=device, dtype=dtype)
        )
        x = x * (2.0 / 255.0) - 1.0

    proprio = _normalize_proprio(_extract_sim_state(obs), processor)

    return x, proprio, imgs


def _extract_sim_state(obs: dict) -> np.ndarray:
    """Build simulator state from current observation.

    This is used as proprio input for model inference.
    """
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return state


def _extract_absolute_pose_wxyz(obs: dict) -> torch.Tensor:
    position = torch.as_tensor(obs["robot0_eef_pos"], dtype=torch.float32)
    quaternion_xyzw = torch.as_tensor(
        obs["robot0_eef_quat"], dtype=torch.float32
    )
    quaternion_wxyz = quaternion_xyzw[[3, 0, 1, 2]]
    quaternion_wxyz = quaternion_wxyz / quaternion_wxyz.norm().clamp_min(1e-12)
    return torch.cat((position, quaternion_wxyz), dim=-1)


def _extract_gripper_open(obs: dict) -> torch.Tensor:
    qpos = torch.as_tensor(obs["robot0_gripper_qpos"], dtype=torch.float32)
    return panda_gripper_qpos_to_open(qpos)


def _is_libero_rothko(cfg: DictConfig) -> bool:
    return (
        str(cfg.data.train.get("raymap_representation", ""))
        == "libero_rothko"
    )


def _absolute_target_to_env_action(
    obs: dict,
    target_pose_wxyz: np.ndarray | torch.Tensor,
    gripper_open: np.ndarray | torch.Tensor | float,
    *,
    binarize_gripper: bool,
) -> np.ndarray:
    current_pose = _extract_absolute_pose_wxyz(obs)
    target_pose = torch.as_tensor(target_pose_wxyz, dtype=torch.float32)
    motion = absolute_target_to_normalized_action(
        current_pose, target_pose, clip=True
    )
    open_value = float(torch.as_tensor(gripper_open).reshape(-1)[0])
    if binarize_gripper:
        open_value = float(open_value >= 0.5)
    # LIBERO environment convention: -1=open, +1=close.
    env_gripper = 1.0 - 2.0 * float(np.clip(open_value, 0.0, 1.0))
    return np.concatenate(
        (motion.detach().cpu().numpy(), np.asarray([env_gripper], dtype=np.float32))
    ).astype(np.float32)


def _denormalize_action(action: torch.Tensor, processor: FastWAMProcessor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B, T, D], got {tuple(action.shape)}")

    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged action key in shape_meta['action']."
        )

    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    denorm = normalizer.backward(action)
    return denorm.numpy()


def _get_num_video_frames(cfg: DictConfig) -> int:
    return (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1


def _validate_eval_runtime_cfg(cfg: DictConfig) -> int:
    num_trials = int(cfg.EVALUATION.num_trials)
    if num_trials <= 0:
        raise ValueError(f"EVALUATION.num_trials must be positive, got {num_trials}.")

    configured_horizon = int(cfg.data.train.num_frames) - 1
    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    action_horizon = (
        configured_horizon
        if action_horizon_cfg is None
        else int(action_horizon_cfg)
    )
    if action_horizon <= 0:
        raise ValueError(
            f"EVALUATION.action_horizon must be positive, got {action_horizon}."
        )
    if _is_libero_rothko(cfg) and action_horizon != configured_horizon:
        raise ValueError(
            "A LIBERO visual-action checkpoint has a fixed horizon determined by "
            f"data.train.num_frames ({configured_horizon}). Got "
            f"EVALUATION.action_horizon={action_horizon}; select the matching h16/h32 "
            "task config instead of overriding the horizon at evaluation time."
        )

    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    if replan_steps <= 0:
        raise ValueError(f"EVALUATION.replan_steps must be positive, got {replan_steps}.")
    if replan_steps > action_horizon:
        raise ValueError(
            f"EVALUATION.replan_steps ({replan_steps}) cannot exceed the model "
            f"action horizon ({action_horizon})."
        )
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    if _record_prediction_videos(cfg) and replan_steps % action_video_freq_ratio != 0:
        raise ValueError(
            "Saving predicted videos requires EVALUATION.replan_steps to be "
            "divisible by data.train.action_video_freq_ratio, got "
            f"{replan_steps} and {action_video_freq_ratio}."
        )
    return action_horizon


def _repeat_initial_states(initial_states: Any, num_trials: int) -> list[Any]:
    """Return exactly ``num_trials`` states without mutating LIBERO's tensor."""
    available = len(initial_states)
    if available <= 0:
        raise ValueError("LIBERO returned no initial states for this task.")
    return [initial_states[index % available] for index in range(num_trials)]


def _record_prediction_videos(cfg: DictConfig) -> bool:
    configured = cfg.EVALUATION.get("save_prediction_videos", None)
    save_predictions = (
        _is_libero_rothko(cfg) if configured is None else bool(configured)
    )
    return bool(
        cfg.EVALUATION.get("visualize_future_video", False)
        or save_predictions
    )


def _validate_visualize_future_video_cfg(cfg: DictConfig) -> None:
    if not _record_prediction_videos(cfg):
        return

    action_conditioned = cfg.model.video_dit_config.get("action_conditioned", None)
    if action_conditioned is not False:
        raise ValueError(
            "Saving predicted future video requires "
            "model.video_dit_config.action_conditioned=false."
        )


def _select_predicted_future_frames(pred_video: list[Image.Image], cfg: DictConfig) -> list[Image.Image]:
    if len(pred_video) == 0:
        raise ValueError("`infer_joint` returned an empty predicted video.")

    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    keep_frames = 1 + num_future_frames
    return list(pred_video[:keep_frames])


def _get_future_frame_capture_steps(cfg: DictConfig) -> list[int]:
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    return [step_idx * action_video_freq_ratio for step_idx in range(num_future_frames + 1)]


def _frame_to_rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, dict):
        images = []
        for value in frame.values():
            value_array = np.array(value) if isinstance(value, Image.Image) else np.array(value, copy=True)
            images.append(value_array)
        return np.concatenate(images, axis=1)
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    return np.array(frame, copy=True)


def _compute_clip_mean_psnr(
    gt_frames: list[Any],
    pred_frames: list[Any],
    eps: float = 1e-8,
) -> Optional[float]:
    if len(gt_frames) == 0 or len(pred_frames) == 0:
        return None
    assert len(gt_frames) == len(pred_frames), (
        "Rollout/pred frame count mismatch for PSNR: "
        f"len(gt_frames)={len(gt_frames)} len(pred_frames)={len(pred_frames)}. "
        "This indicates temporal misalignment in future-video capture."
    )
    num_frames = len(gt_frames)

    frame_psnr_values = []
    for gt_frame, pred_frame in zip(gt_frames[:num_frames], pred_frames[:num_frames]):
        gt_image = _frame_to_rgb_array(gt_frame)
        pred_image = _frame_to_rgb_array(pred_frame)
        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            gt_image = np.array(
                Image.fromarray(gt_image).resize((target_w, target_h), resample=Image.BILINEAR)
            )

        gt_f32 = gt_image.astype(np.float32)
        pred_f32 = pred_image.astype(np.float32)
        mse = float(np.mean((pred_f32 - gt_f32) ** 2))
        psnr = 10.0 * np.log10((255.0 * 255.0) / max(mse, eps))
        frame_psnr_values.append(float(psnr))

    if len(frame_psnr_values) == 0:
        return None
    return float(np.mean(frame_psnr_values))


def _predict_action_chunk(
    obs: dict,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    prompt_context: Optional[torch.Tensor] = None,
    prompt_context_mask: Optional[torch.Tensor] = None,
) -> tuple[Any, dict, Optional[list[Image.Image]]]:
    num_inference_steps_cfg = cfg.EVALUATION.get("num_inference_steps", None)
    if num_inference_steps_cfg is None:
        num_inference_steps = int(cfg.get("eval_num_inference_steps", 20))
    else:
        num_inference_steps = int(num_inference_steps_cfg)
    prompt_template = DEFAULT_PROMPT
    prompt = prompt_template.format(task=task_description)

    image, proprio, imgs = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device=model_device,
        dtype=model.torch_dtype,
    )

    infer_kwargs = {
        "prompt": prompt,
        "input_image": image,
        "action_horizon": action_horizon,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": num_inference_steps,
        "proprio": proprio,
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "seed": None if cfg.get("seed") is None else int(cfg.seed),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }
    visualize_future_video = _record_prediction_videos(cfg)
    predicted_future_frames = None
    if _is_libero_rothko(cfg):
        current_pose = _extract_absolute_pose_wxyz(obs)
        current_gripper = _extract_gripper_open(obs)
        input_raymap = model.raymap_codec.encode(
            current_pose.unsqueeze(0),
            current_gripper.unsqueeze(0),
        )[:, 0].unsqueeze(0)
        visual_kwargs = {
            "prompt": prompt if prompt_context is None else None,
            "input_image": image,
            "input_raymap": input_raymap.to(
                device=model_device, dtype=model.torch_dtype
            ),
            "current_endpose": current_pose,
            "proprio": proprio,
            "num_frames": _get_num_video_frames(cfg),
            "num_inference_steps": num_inference_steps,
            "sigma_shift": infer_kwargs["sigma_shift"],
            "seed": infer_kwargs["seed"],
            "rand_device": infer_kwargs["rand_device"],
            "tiled": infer_kwargs["tiled"],
            # The RGB latent trajectory is still predicted. Avoid the unused
            # VAE pixel decode unless prediction artifacts are requested.
            "decode_future_rgb": _record_prediction_videos(cfg),
        }
        if prompt_context is not None:
            if prompt_context_mask is None:
                raise ValueError(
                    "prompt_context_mask is required with prompt_context."
                )
            visual_kwargs["context"] = prompt_context
            visual_kwargs["context_mask"] = prompt_context_mask
        with torch.no_grad():
            pred = model.infer(**visual_kwargs)
        if "pose" not in pred or "gripper" not in pred:
            raise ValueError(
                "LIBERO Rothko model inference did not return pose/gripper."
            )
        predicted_future_frames = None
        if _record_prediction_videos(cfg) and "video" in pred:
            predicted_future_frames = _select_predicted_future_frames(
                pred["video"], cfg
            )
        visual_chunk = {
            "target_pose": pred["pose"][0, 1:].float().cpu().numpy(),
            "gripper_open": pred["gripper"][0, 1:].float().cpu().numpy(),
        }
        if _record_prediction_videos(cfg):
            visual_chunk["predicted_raymap_frames"] = (
                model._video_tensor_to_pil(pred["raymap"][0])
            )
        return visual_chunk, imgs, predicted_future_frames

    if visualize_future_video:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)
    elif "num_video_frames" in inspect.signature(model.infer_action).parameters:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)

    with torch.no_grad():
        if visualize_future_video:
            pred = model.infer_joint(**infer_kwargs)
            predicted_future_frames = _select_predicted_future_frames(pred["video"], cfg)
        else:
            pred = model.infer_action(**infer_kwargs)
    action = pred["action"]  # [T, D]

    action = _denormalize_action(action, processor)[0]  # [T, D]

    # The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    return action, imgs, predicted_future_frames


def _get_max_steps(task_suite_name: str) -> int:
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


def run_single_episode(
    env,
    initial_state,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    episode_idx: int,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    prompt_context: Optional[torch.Tensor] = None,
    prompt_context_mask: Optional[torch.Tensor] = None,
) -> tuple[
    bool,
    list,
    list[dict[str, Any]],
    Optional[float],
    list[dict[str, Any]],
]:
    max_steps = _get_max_steps(cfg.EVALUATION.task_suite_name)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    use_action_ensembler = bool(cfg.EVALUATION.get("use_action_ensembler", False))
    visualize_future_video = _record_prediction_videos(cfg)
    capture_steps = set(_get_future_frame_capture_steps(cfg)[1:])

    env.reset()
    obs = env.set_init_state(initial_state)
    if use_action_ensembler:
        if _is_libero_rothko(cfg):
            ensembler = AbsolutePoseActionEnsembler(
                decay=float(cfg.EVALUATION.get("action_ensemble_decay", 0.01))
            )
        else:
            ensembler = ActionEnsembler()
        ensembler.reset()

    replay_images = []
    predicted_future_video_clips: list[dict[str, Any]] = []
    control_trace: list[dict[str, Any]] = []
    episode_future_clip_psnr: list[float] = []
    pending_actions: list[list[float]] = []
    current_predicted_future_clip: Optional[dict[str, Any]] = None
    current_replan_step = 0
    current_replan_idx = -1
    executed_any_action = False

    t = 0
    done = False
    pbar = tqdm(total=max_steps + num_steps_wait, desc=f"Episode {episode_idx + 1}")
    while t < max_steps + num_steps_wait:
        pbar.update(1)
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue

        if len(pending_actions) == 0:
            action_chunk, imgs, predicted_future_frames = _predict_action_chunk(
                obs=obs,
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
                prompt_context=prompt_context,
                prompt_context_mask=prompt_context_mask,
            )
            current_replan_idx += 1
            predicted_raymap_frames = (
                action_chunk.get("predicted_raymap_frames")
                if isinstance(action_chunk, dict)
                else None
            )
            if (
                predicted_future_frames is not None
                or predicted_raymap_frames is not None
            ):
                current_predicted_future_clip = {
                    "replan_idx": current_replan_idx,
                    "gt_frames": (
                        [imgs.copy()]
                        if predicted_future_frames is not None
                        else []
                    ),
                    "pred_frames": predicted_future_frames or [],
                }
                if predicted_raymap_frames is not None:
                    current_predicted_future_clip["pred_raymap_frames"] = (
                        predicted_raymap_frames
                    )
            else:
                current_predicted_future_clip = None
            current_replan_step = 0
            if isinstance(action_chunk, dict):
                target_pose = action_chunk["target_pose"]
                gripper_open = action_chunk["gripper_open"]
                num_pending_actions = min(replan_steps, len(target_pose))
                if use_action_ensembler:
                    ensembler.add_actions(target_pose, gripper_open, t)
                    pending_actions = [
                        (*ensembler.get_action(t + index), index)
                        for index in range(num_pending_actions)
                    ]
                    ensembler.cleanup(t)
                else:
                    pending_actions = [
                        (target_pose[index], gripper_open[index], index)
                        for index in range(num_pending_actions)
                    ]
            elif use_action_ensembler:
                ensembler.add_actions(action_chunk, t)
                pending_actions = [ensembler.get_action(ts).tolist() for ts in range(t, t + replan_steps)]
            else:
                pending_actions = action_chunk[:replan_steps].tolist()
            replay_images.append(imgs.copy())
        else:
            imgs = get_libero_image(obs)
            replay_images.append(imgs.copy())

        pending = pending_actions.pop(0)
        if _is_libero_rothko(cfg):
            target_pose, gripper_open, target_index = pending
            actual_pose_before = _extract_absolute_pose_wxyz(obs)
            raw_motion = absolute_target_to_normalized_action(
                actual_pose_before,
                torch.as_tensor(target_pose, dtype=torch.float32),
                clip=False,
            )
            action_to_execute = _absolute_target_to_env_action(
                obs,
                target_pose,
                gripper_open,
                binarize_gripper=bool(
                    cfg.EVALUATION.get("binarize_gripper", True)
                ),
            )
            if bool(cfg.EVALUATION.get("save_control_trace", False)):
                control_trace.append(
                    {
                        "episode": int(episode_idx),
                        "env_step": int(t),
                        "replan_index": int(current_replan_idx),
                        "target_index": int(target_index),
                        "predicted_target_xyz": np.asarray(target_pose)[:3].tolist(),
                        "predicted_target_quaternion_wxyz": np.asarray(
                            target_pose
                        )[3:7].tolist(),
                        "actual_xyz_before_action": actual_pose_before[:3].tolist(),
                        "actual_quaternion_wxyz_before_action": actual_pose_before[
                            3:7
                        ].tolist(),
                        "sent_delta_action": action_to_execute.tolist(),
                        "predicted_gripper_open": float(
                            np.asarray(gripper_open).reshape(-1)[0]
                        ),
                        "clipped_dimensions": (
                            raw_motion.abs() > 1.0
                        ).nonzero(as_tuple=False).flatten().tolist(),
                    }
                )
        else:
            action_to_execute = pending
        obs, _, done, _ = env.step(action_to_execute)
        if visualize_future_video and current_predicted_future_clip is not None:
            current_replan_step += 1
            if (
                current_predicted_future_clip["pred_frames"]
                and current_replan_step in capture_steps
            ):
                current_predicted_future_clip["gt_frames"].append(get_libero_image(obs))
            if done or len(pending_actions) == 0:
                expected_frame_count = 1 + sum(
                    1 for capture_step in capture_steps if capture_step <= current_replan_step
                )
                pred_len = len(current_predicted_future_clip["pred_frames"])
                if pred_len:
                    gt_len = len(current_predicted_future_clip["gt_frames"])
                    assert gt_len == expected_frame_count, (
                        "Rollout future frames do not match expected capture count: "
                        f"gt_len={gt_len} expected={expected_frame_count} "
                        f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']} "
                        f"current_replan_step={current_replan_step} capture_steps={sorted(capture_steps)}."
                    )
                    assert pred_len >= expected_frame_count, (
                        "Predicted future frames shorter than expected capture count: "
                        f"pred_len={pred_len} expected={expected_frame_count} "
                        f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                    )
                    if pred_len != expected_frame_count:
                        logging.info(
                            "Align predicted clip length to executed steps: "
                            "episode=%s replan=%s done=%s expected=%s pred_full=%s",
                            episode_idx,
                            current_predicted_future_clip["replan_idx"],
                            done,
                            expected_frame_count,
                            pred_len,
                        )
                    current_predicted_future_clip["pred_frames"] = (
                        current_predicted_future_clip["pred_frames"][
                            :expected_frame_count
                        ]
                    )
                if current_predicted_future_clip.get("pred_raymap_frames") is not None:
                    current_predicted_future_clip["pred_raymap_frames"] = (
                        current_predicted_future_clip["pred_raymap_frames"][:expected_frame_count]
                    )
                if pred_len:
                    assert len(current_predicted_future_clip["gt_frames"]) == len(
                        current_predicted_future_clip["pred_frames"]
                    ), (
                        "Rollout/pred frame count mismatch after alignment: "
                        f"len(gt_frames)={len(current_predicted_future_clip['gt_frames'])} "
                        f"len(pred_frames)={len(current_predicted_future_clip['pred_frames'])} "
                        f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                    )
                    clip_psnr = _compute_clip_mean_psnr(
                        current_predicted_future_clip["gt_frames"],
                        current_predicted_future_clip["pred_frames"],
                    )
                    if clip_psnr is not None:
                        episode_future_clip_psnr.append(clip_psnr)
                predicted_future_video_clips.append(current_predicted_future_clip)
                current_predicted_future_clip = None
        executed_any_action = True
        if done:
            break
        t += 1
    pbar.close()

    # The loop records each pre-action observation. Preserve the final
    # post-action observation as well so rollout videos include the terminal
    # state (success or timeout) instead of ending one control step early.
    if executed_any_action:
        replay_images.append(get_libero_image(obs).copy())

    episode_mean_psnr = (
        float(np.mean(episode_future_clip_psnr)) if len(episode_future_clip_psnr) > 0 else None
    )
    return (
        bool(done),
        replay_images,
        predicted_future_video_clips,
        episode_mean_psnr,
        control_trace,
    )


def run_single_task(
    task,
    initial_states,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    video_dir: Path,
    predicted_video_dir: Path,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> dict:
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))
    model_prompt = DEFAULT_PROMPT.format(task=task_description)
    logging.info("LIBERO task language: %s", task_description)
    logging.info("LIBERO model prompt: %s", model_prompt)
    visualize_future_video = _record_prediction_videos(cfg)
    results = {
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "task_description": task_description,
    }
    if visualize_future_video:
        results["episode_future_video_psnr"] = []
        results["future_video_psnr_mean"] = None

    for trial_idx in range(int(cfg.EVALUATION.num_trials)):
        (
            success,
            replay_images,
            predicted_future_video_clips,
            episode_mean_psnr,
            control_trace,
        ) = run_single_episode(
            env=env,
            initial_state=initial_states[trial_idx],
            task_description=task_description,
            model=model,
            processor=processor,
            cfg=cfg,
            episode_idx=trial_idx,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
        )
        if success:
            results["successes"] += 1
            results["success_episodes"].append(trial_idx)
        else:
            results["failure_episodes"].append(trial_idx)
        if visualize_future_video:
            results["episode_future_video_psnr"].append(episode_mean_psnr)

        save_rollout_video(
            video_dir,
            replay_images,
            f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
            success=success,
            task_description=task_description,
        )
        if bool(cfg.EVALUATION.get("save_control_trace", False)):
            trace_dir = video_dir.parent / "control_traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = trace_dir / (
                f"task{cfg.EVALUATION.task_id}_trial{trial_idx:04d}.jsonl"
            )
            with trace_path.open("w", encoding="utf-8") as handle:
                for row in control_trace:
                    handle.write(json.dumps(row, cls=NumpyEncoder) + "\n")
        if visualize_future_video:
            if len(predicted_future_video_clips) == 0:
                logging.warning(
                    "No predicted future frames collected for task %s trial %s.",
                    cfg.EVALUATION.task_id,
                    trial_idx,
                )
            else:
                all_gt_frames = []
                all_pred_frames = []
                all_pred_raymap_frames = []
                for clip in predicted_future_video_clips:
                    all_gt_frames.extend(clip["gt_frames"])
                    all_pred_frames.extend(clip["pred_frames"])
                    raymap_frames = clip.get("pred_raymap_frames")
                    if raymap_frames:
                        all_pred_raymap_frames.extend(raymap_frames)
                if all_pred_frames:
                    save_prediction_video(
                        predicted_video_dir,
                        all_gt_frames,
                        all_pred_frames,
                        f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                        "all",
                        success=success,
                        task_description=task_description,
                    )
                if all_pred_raymap_frames:
                    save_model_prediction_video(
                        predicted_video_dir,
                        all_pred_raymap_frames,
                        f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                        "all",
                        "rothko",
                        success=success,
                        task_description=task_description,
                    )

    if visualize_future_video:
        valid_episode_psnr = [x for x in results["episode_future_video_psnr"] if x is not None]
        if len(valid_episode_psnr) > 0:
            results["future_video_psnr_mean"] = float(np.mean(valid_episode_psnr))
    return results


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def eval_single_process(cfg: DictConfig):
    start_time = time.time()
    partial_state = PartialState()
    partial_state.config = cfg

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    _validate_visualize_future_video_cfg(cfg)
    action_horizon = _validate_eval_runtime_cfg(cfg)

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError(
            "Only env_num=1 is supported in eval_libero_single.py. "
            "Use run_libero_manager/run_libero_parallel_test.sh for multi-GPU task parallelism."
        )

    if _is_libero_rothko(cfg):
        # sim_libero's standard action-expert defaults skip loading the base
        # DiT. A portable LoRA checkpoint instead requires its configured
        # original Wan base model.
        with open_dict(cfg.model):
            cfg.model.skip_dit_load_from_pretrain = False
            cfg.model.pop("action_dit_pretrained_path", None)
    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    if hasattr(model, "validate_dataset_stats"):
        model.validate_dataset_stats(dataset_stats_path)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])
    concat_multi_camera = cfg.data.train.get("concat_multi_camera", None)
    shape_meta_images = [meta["shape"] for meta in processor.shape_meta["images"]]

    local_log_dir = Path(cfg.EVALUATION.output_dir)
    local_log_dir.mkdir(parents=True, exist_ok=True)
    video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    predicted_video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "predicted_videos"
    if _record_prediction_videos(cfg):
        predicted_video_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.EVALUATION.task_suite_name]()
    task = task_suite.get_task(cfg.EVALUATION.task_id)
    initial_states = task_suite.get_task_init_states(cfg.EVALUATION.task_id)
    initial_states = _repeat_initial_states(
        initial_states,
        int(cfg.EVALUATION.num_trials),
    )

    results = {
        "task_suite": cfg.EVALUATION.task_suite_name,
        "task_id": cfg.EVALUATION.task_id,
        "task_description": None,
        "successes": 0,
        "total_episodes": int(cfg.EVALUATION.num_trials),
        "gpu_id": int(cfg.gpu_id),
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0,
    }

    logging.info("Running LIBERO evaluation with env_num=1")
    task_results = run_single_task(
        task=task,
        initial_states=initial_states,
        model=model,
        processor=processor,
        cfg=cfg,
        video_dir=video_dir,
        predicted_video_dir=predicted_video_dir,
        action_horizon=action_horizon,
        input_w=input_w,
        input_h=input_h,
        model_device=model_device,
    )
    results.update(task_results)
    # Keep both the raw LIBERO instruction and the exact text passed to the
    # model in every per-task result file. All trials and replans for one task
    # share these strings.
    task_description = str(results["task_description"])
    results["language"] = task_description
    results["model_prompt"] = DEFAULT_PROMPT.format(task=task_description)

    results["duration"] = time.time() - start_time
    output_dir = Path(cfg.EVALUATION.output_dir) / cfg.EVALUATION.task_suite_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    print(
        f"Task {cfg.EVALUATION.task_id} completed: "
        f"{results['successes']}/{cfg.EVALUATION.num_trials} successes"
    )
    if results.get("future_video_psnr_mean") is not None:
        print(f"Task {cfg.EVALUATION.task_id} future-video PSNR mean: {results['future_video_psnr_mean']:.4f}")
    print(f"Time taken: {results['duration']:.2f} seconds")
    return results


if __name__ == "__main__":
    eval_single_process()
