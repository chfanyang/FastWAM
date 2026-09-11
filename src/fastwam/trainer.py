import logging
import json
import hashlib
import inspect
import os
import random
import re
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .models.lora import LoRAConfig, mark_only_lora_trainable
from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


def load_fixed_validation_manifest(
    path: str | Path,
    *,
    expected_split_manifest_sha256: str,
    val_dataset_length: int,
) -> dict:
    """Load a fixed loss/visual validation window manifest."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Fixed validation manifest does not exist: {manifest_path}"
        )
    payload = json.loads(manifest_path.read_text())
    if int(payload.get("version", -1)) != 1:
        raise ValueError(
            f"Unsupported fixed validation manifest version in {manifest_path}: "
            f"{payload.get('version')!r}"
        )
    actual_split_sha256 = str(payload.get("split_manifest_sha256", ""))
    if actual_split_sha256 != expected_split_manifest_sha256:
        raise ValueError(
            "Fixed validation manifest split fingerprint mismatch: "
            f"manifest={actual_split_sha256!r}, "
            f"dataset={expected_split_manifest_sha256!r}."
        )
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Fixed validation manifest has no samples: {manifest_path}")
    required = {
        "sample_id",
        "val_dataset_index",
        "episode_index",
        "frame_index",
        "task_index",
        "diffusion_seed",
        "run_visual",
    }
    sample_ids = set()
    dataset_indices = set()
    visual_count = 0
    for offset, record in enumerate(samples):
        if not isinstance(record, dict):
            raise ValueError(f"Sample {offset} in {manifest_path} is not an object.")
        missing = required - set(record)
        if missing:
            raise ValueError(
                f"Sample {offset} in {manifest_path} misses {sorted(missing)}."
            )
        sample_id = str(record["sample_id"])
        dataset_index = int(record["val_dataset_index"])
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate sample_id {sample_id!r} in {manifest_path}.")
        if dataset_index in dataset_indices:
            raise ValueError(
                f"Duplicate val_dataset_index {dataset_index} in {manifest_path}."
            )
        if not 0 <= dataset_index < val_dataset_length:
            raise ValueError(
                f"val_dataset_index {dataset_index} is outside [0,{val_dataset_length}) "
                f"in {manifest_path}."
            )
        sample_ids.add(sample_id)
        dataset_indices.add(dataset_index)
        visual_count += int(bool(record["run_visual"]))
    if visual_count < 1:
        raise ValueError(f"Fixed validation manifest selects no visual samples: {manifest_path}")
    payload["path"] = str(manifest_path.resolve())
    payload["sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return payload


def _decoded_pose_metrics(
    predicted_pose: torch.Tensor,
    target_pose: torch.Tensor,
    predicted_gripper: torch.Tensor,
    target_gripper: torch.Tensor,
) -> dict[str, float]:
    """Compute unit-aware pose metrics with quaternion sign invariance."""
    if predicted_pose.shape != target_pose.shape:
        raise ValueError(
            f"Pose shape mismatch: {tuple(predicted_pose.shape)} vs {tuple(target_pose.shape)}"
        )
    pose_dim = int(predicted_pose.shape[-1])
    if pose_dim % 7 != 0:
        raise ValueError(f"Decoded pose dim must be a multiple of 7, got {pose_dim}.")
    num_arms = pose_dim // 7
    predicted = predicted_pose.reshape(*predicted_pose.shape[:-1], num_arms, 7)
    target = target_pose.reshape(*target_pose.shape[:-1], num_arms, 7)

    position_difference = predicted[..., :3] - target[..., :3]
    predicted_quaternion = torch.nn.functional.normalize(
        predicted[..., 3:7], dim=-1, eps=1e-8
    )
    target_quaternion = torch.nn.functional.normalize(
        target[..., 3:7], dim=-1, eps=1e-8
    )
    # q and -q represent the same rotation, hence abs(dot).
    quaternion_dot = (
        predicted_quaternion * target_quaternion
    ).sum(dim=-1).abs().clamp(0.0, 1.0)
    rotation_error_deg = torch.rad2deg(2.0 * torch.acos(quaternion_dot))

    predicted_gripper = predicted_gripper.float()
    target_gripper = target_gripper.float()
    if predicted_gripper.shape != target_gripper.shape:
        raise ValueError(
            "Gripper shape mismatch: "
            f"{tuple(predicted_gripper.shape)} vs {tuple(target_gripper.shape)}"
        )
    return {
        "decoded_position_mae_m": float(position_difference.abs().mean()),
        "decoded_position_rmse_m": float(position_difference.square().mean().sqrt()),
        "decoded_rotation_geodesic_deg": float(rotation_error_deg.mean()),
        "decoded_gripper_mae": float(
            (predicted_gripper - target_gripper).abs().mean()
        ),
        "decoded_gripper_accuracy": float(
            ((predicted_gripper >= 0.5) == (target_gripper >= 0.5)).float().mean()
        ),
    }


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.warmup_ratio = float(cfg.get("warmup_ratio", 0.05))
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError(
                f"`warmup_ratio` must be in [0, 1), got {self.warmup_ratio}."
            )
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.pin_memory = bool(cfg.get("pin_memory", torch.cuda.is_available()))
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        state_save_every = cfg.get("state_save_every")
        self.state_save_every = (
            self.save_every
            if state_save_every is None
            else int(state_save_every)
        )
        self.save_at_end = bool(cfg.get("save_at_end", True))
        self.eval_every = int(cfg.eval_every)
        self.eval_at_start = bool(cfg.get("eval_at_start", False))
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.eval_sample_index = int(cfg.get("eval_sample_index", 0))
        self.eval_num_samples = int(cfg.get("eval_num_samples", 4))
        self.eval_random_seed = int(cfg.get("eval_random_seed", 42))
        self.eval_sample_manifest_path = cfg.get("eval_sample_manifest")
        if self.eval_sample_manifest_path in (None, "", "null"):
            self.eval_sample_manifest_path = None
        else:
            self.eval_sample_manifest_path = str(self.eval_sample_manifest_path)
        self.fixed_validation_manifest = None
        if self.eval_sample_index < 0:
            raise ValueError(
                f"`eval_sample_index` must be non-negative, got {self.eval_sample_index}."
            )
        if self.eval_num_samples < 1:
            raise ValueError(
                f"`eval_num_samples` must be positive, got {self.eval_num_samples}."
            )
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        finetune_cfg = cfg.get("finetune")
        if isinstance(finetune_cfg, DictConfig):
            finetune_cfg = OmegaConf.to_container(finetune_cfg, resolve=True)
        if finetune_cfg is None:
            finetune_cfg = {}
        if not isinstance(finetune_cfg, dict):
            raise ValueError(
                f"`finetune` must be dict-like, got {type(finetune_cfg)}."
            )
        self.finetune_method = str(
            finetune_cfg.get("method", "full")
        ).strip().lower()
        if self.finetune_method not in {"full", "lora"}:
            raise ValueError(
                f"Unsupported finetune.method={self.finetune_method!r}; "
                "expected `full` or `lora`."
            )
        self.train_proprio_encoder = bool(
            finetune_cfg.get("train_proprio_encoder", True)
        )
        lora_cfg = finetune_cfg.get("lora")
        if isinstance(lora_cfg, DictConfig):
            lora_cfg = OmegaConf.to_container(lora_cfg, resolve=True)
        self.lora_config = (
            LoRAConfig.from_dict(lora_cfg)
            if self.finetune_method == "lora"
            else None
        )
        
        self.resume = cfg.resume
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")
        if self.eval_sample_manifest_path is not None:
            if self.val_dataset is None:
                raise ValueError("`eval_sample_manifest` requires a validation dataset.")
            split_metadata = getattr(self.val_dataset, "episode_split_metadata", None)
            if not split_metadata or split_metadata.get("split") != "val":
                raise ValueError(
                    "`eval_sample_manifest` requires a validation dataset built "
                    "from `episode_split=val`."
                )
            self.fixed_validation_manifest = load_fixed_validation_manifest(
                self.eval_sample_manifest_path,
                expected_split_manifest_sha256=str(split_metadata["sha256"]),
                val_dataset_length=len(self.val_dataset),
            )
            logger.info(
                "Using fixed validation manifest: path=%s sha256=%s "
                "loss_samples=%d visual_samples=%d",
                self.fixed_validation_manifest["path"],
                self.fixed_validation_manifest["sha256"],
                len(self.fixed_validation_manifest["samples"]),
                sum(
                    bool(record["run_visual"])
                    for record in self.fixed_validation_manifest["samples"]
                ),
            )

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)
        trainable_params = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        if not trainable_params:
            raise ValueError(
                f"No trainable parameters for finetune.method={self.finetune_method!r}."
            )
        trainable_count = sum(parameter.numel() for parameter in trainable_params)
        total_count = sum(parameter.numel() for parameter in self.model.parameters())
        logger.info(
            "Fine-tuning mode=%s trainable_parameters=%d total_parameters=%d "
            "trainable_ratio=%.6f%% train_proprio_encoder=%s",
            self.finetune_method,
            trainable_count,
            total_count,
            100.0 * trainable_count / max(total_count, 1),
            self.train_proprio_encoder,
        )
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        # Optional epoch-derived intervals, using the same epoch definition as
        # the training budget. Existing step-based configurations are unchanged.
        for option, attribute in (("save_every_epochs", "save_every"), ("state_save_every_epochs", "state_save_every"), ("eval_every_epochs", "eval_every")):
            interval = cfg.get(option)
            if interval is not None:
                if cfg.get("max_steps") is not None or int(interval) <= 0:
                    raise ValueError(f"{option} requires max_steps=null and a positive epoch interval")
                setattr(self, attribute, (total_train_steps // self.num_epochs) * int(interval))
        warmup_steps = int(total_train_steps * self.warmup_ratio)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        if self.cfg.get("diagnostics_on_train", False):
            payload = {
                (key.replace("eval/", "train_diagnostic/", 1).replace("val_loss", "denoising_loss")
                 if key.startswith("eval/") else key): value
                for key, value in payload.items()
            }
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2.")

    def _set_dit_only_train_mode(self):
        logger.info(
            "Restoring fine-tuning train mode: method=%s.",
            self.finetune_method,
        )
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    def _apply_dit_only_train_mode(self, model):
        model.eval()
        if self.finetune_method == "lora":
            enable_lora = getattr(model, "enable_lora", None)
            if not callable(enable_lora):
                raise TypeError(
                    f"Model {type(model).__name__} does not support LoRA fine-tuning."
                )
            enable_lora(
                self.lora_config,
                train_proprio_encoder=self.train_proprio_encoder,
            )
        model.requires_grad_(False)
        model.dit.train()
        if self.finetune_method == "full":
            model.dit.requires_grad_(True)
        else:
            mark_only_lora_trainable(model.dit)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None and self.train_proprio_encoder:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        output = {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }
        optional_tensor_dims = {
            "raymap": 4,
            "image_is_pad": 1,
            "raymap_is_pad": 1,
            "current_endpose": 1,
            "future_endpose": 2,
            "future_gripper": 2,
        }
        for key, unbatched_dim in optional_tensor_dims.items():
            value = sample.get(key)
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"`sample[{key!r}]` must be a tensor, got {type(value)}")
            if value.ndim == unbatched_dim:
                value = value.unsqueeze(0)
            if value.ndim != unbatched_dim + 1:
                raise ValueError(
                    f"`sample[{key!r}]` must have {unbatched_dim} or "
                    f"{unbatched_dim + 1} dims, got {tuple(value.shape)}"
                )
            output[key] = value
        return output

    def _get_fixed_visual_action_eval_sample(self, index: int, seed: int):
        """Load one eval sample deterministically without perturbing train RNG."""
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        try:
            np.random.seed(seed % (2**32))
            random.seed(seed)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                sample = self.val_dataset[index]
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
        return self._to_batched_eval_sample(sample)

    @torch.no_grad()
    def _evaluate_visual_action(
        self,
        model,
        sample,
        was_dit_training,
        *,
        eval_sample_index: int,
        eval_seed: int,
        compute_val_loss: bool = True,
    ):
        cuda_devices = []
        if self.accelerator.device.type == "cuda":
            cuda_devices = [self.accelerator.device.index]
        # Validation must not consume or perturb training RNG state.  The same
        # fixed sample receives the same timestep/noise at every evaluation.
        val_loss_value = None
        if compute_val_loss:
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(eval_seed)
                if self.accelerator.device.type == "cuda":
                    torch.cuda.manual_seed(eval_seed)
                with self.accelerator.autocast():
                    val_loss, _ = model.training_loss(sample)
                    val_loss_value = float(val_loss.float().item())

        video0 = sample["video"][0]
        raymap0 = sample["raymap"][0]
        proprio0 = sample["proprio"][0, 0]
        infer_kwargs = {
            "prompt": None,
            "input_image": video0[:, 0].unsqueeze(0),
            "input_raymap": raymap0[:, 0].unsqueeze(0),
            "proprio": proprio0,
            "current_endpose": sample["current_endpose"][0],
            "context": sample["context"][0],
            "context_mask": sample["context_mask"][0],
            "num_frames": video0.shape[1],
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": eval_seed,
            "tiled": False,
        }
        prediction = model.infer(**infer_kwargs)
        target_video = (
            (video0.detach().float().cpu().clamp(-1, 1) + 1.0) * 0.5
        ).contiguous()
        rgb_latents = model._encode_video_latents(
            video0.unsqueeze(0).to(model.device, model.torch_dtype)
        )
        vae_video = model._decode_video_tensor(rgb_latents)[0]
        vae_video = ((vae_video.cpu() + 1.0) * 0.5).clamp(0, 1)
        psnr_decode_vs_gt = video_psnr(vae_video, target_video)
        ssim_decode_vs_gt = video_ssim(vae_video, target_video)

        psnr_rollout_vs_gt = ssim_rollout_vs_gt = None
        psnr_rollout_vs_decode = ssim_rollout_vs_decode = None
        if "video" in prediction:
            predicted_video = pil_frames_to_video_tensor(prediction["video"])
            if predicted_video.shape != target_video.shape:
                raise ValueError(
                    f"Visual-action RGB shape mismatch: {predicted_video.shape} "
                    f"vs {target_video.shape}"
                )
            psnr_rollout_vs_gt = video_psnr(predicted_video, target_video)
            ssim_rollout_vs_gt = video_ssim(predicted_video, target_video)
            psnr_rollout_vs_decode = video_psnr(predicted_video, vae_video)
            ssim_rollout_vs_decode = video_ssim(predicted_video, vae_video)
            stitched = torch.cat(
                (predicted_video, vae_video, target_video), dim=2
            ).contiguous()
        else:
            predicted_raymap = (
                (prediction["raymap"][0].detach().float().cpu().clamp(-1, 1) + 1.0)
                * 0.5
            ).contiguous()
            target_raymap = (
                (raymap0.detach().float().cpu().clamp(-1, 1) + 1.0) * 0.5
            ).contiguous()
            if predicted_raymap.shape != target_raymap.shape:
                raise ValueError(
                    "Visual-action Raymap shape mismatch: "
                    f"{predicted_raymap.shape} vs {target_raymap.shape}"
                )
            stitched = torch.cat(
                (predicted_raymap, target_raymap), dim=2
            ).contiguous()

        predicted_pose = prediction.get("pose")
        predicted_gripper = prediction.get("gripper")
        representation_l1 = representation_l2 = None
        pose_metrics = {}
        if predicted_pose is not None and predicted_gripper is not None:
            predicted_future_pose = predicted_pose[:, 1:].float().cpu()
            predicted_future_gripper = predicted_gripper[:, 1:].float().cpu()
            target_future_pose = sample["future_endpose"].float().cpu()
            target_future_gripper = sample["future_gripper"].float().cpu()
            predicted_target = torch.cat(
                (
                    predicted_future_pose,
                    predicted_future_gripper,
                ),
                dim=-1,
            )
            target = torch.cat(
                (
                    target_future_pose,
                    target_future_gripper,
                ),
                dim=-1,
            )
            if predicted_target.shape != target.shape:
                raise ValueError(
                    "Decoded raymap target shape mismatch: "
                    f"{tuple(predicted_target.shape)} vs {tuple(target.shape)}"
                )
            difference = predicted_target - target
            representation_l1 = float(difference.abs().mean())
            representation_l2 = float(difference.square().mean())
            pose_metrics = _decoded_pose_metrics(
                predicted_future_pose,
                target_future_pose,
                predicted_future_gripper,
                target_future_gripper,
            )

        stitched_frames = [
            Image.fromarray(
                (
                    stitched[:, index]
                    .permute(1, 2, 0)
                    .clamp(0, 1)
                    .numpy()
                    * 255
                ).astype(np.uint8)
            )
            for index in range(stitched.shape[1])
        ]
        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}"
            f"_sample_{eval_sample_index:06d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        if was_dit_training:
            self._set_dit_only_train_mode()
        result = {
            "psnr_dg": float(psnr_decode_vs_gt),
            "ssim_dg": float(ssim_decode_vs_gt),
            "video_path": video_path,
        }
        if val_loss_value is not None:
            result["val_loss"] = val_loss_value
        if psnr_rollout_vs_gt is not None:
            result.update(
                {
                    "psnr_rg": float(psnr_rollout_vs_gt),
                    "ssim_rg": float(ssim_rollout_vs_gt),
                    "psnr_rd": float(psnr_rollout_vs_decode),
                    "ssim_rd": float(ssim_rollout_vs_decode),
                }
            )
        if representation_l2 is not None:
            result.update(pose_metrics)
            result["decoded_target_l2"] = representation_l2
            result["decoded_target_l1"] = representation_l1
            if prediction.get("action") is not None:
                # Preserve the existing RoboTwin metric names.
                result["action_l2"] = representation_l2
                result["action_l1"] = representation_l1
        return result

    def _fixed_visual_action_loss(self, model, sample, seed: int) -> float:
        cuda_devices = []
        if self.accelerator.device.type == "cuda":
            cuda_devices = [self.accelerator.device.index]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if self.accelerator.device.type == "cuda":
                torch.cuda.manual_seed(seed)
            with self.accelerator.autocast():
                val_loss, _ = model.training_loss(sample)
        return float(val_loss.float().item())

    def _evaluate_visual_action_manifest(self, model, was_dit_training: bool):
        records = self.fixed_validation_manifest["samples"]
        local_loss_values = []
        for record_offset in range(
            self.accelerator.process_index,
            len(records),
            self.accelerator.num_processes,
        ):
            record = records[record_offset]
            eval_index = int(record["val_dataset_index"])
            eval_seed = int(record["diffusion_seed"])
            sample = self._get_fixed_visual_action_eval_sample(eval_index, eval_seed)
            local_loss_values.append(
                self._fixed_visual_action_loss(model, sample, eval_seed)
            )

        loss_totals = torch.tensor(
            [[sum(local_loss_values), len(local_loss_values)]],
            device=self.accelerator.device,
            dtype=torch.float64,
        )
        gathered_loss = self.accelerator.gather(loss_totals).sum(dim=0)
        loss_count = int(gathered_loss[1])
        if loss_count != len(records):
            raise RuntimeError(
                "Fixed validation loss sample count mismatch: "
                f"gathered={loss_count}, expected={len(records)}."
            )
        result = {"val_loss": float(gathered_loss[0]) / loss_count}

        visual_records = [record for record in records if bool(record["run_visual"])]
        local_visual_results = []
        for visual_offset in range(
            self.accelerator.process_index,
            len(visual_records),
            self.accelerator.num_processes,
        ):
            record = visual_records[visual_offset]
            eval_index = int(record["val_dataset_index"])
            eval_seed = int(record["diffusion_seed"])
            sample = self._get_fixed_visual_action_eval_sample(eval_index, eval_seed)
            local_visual_results.append(
                self._evaluate_visual_action(
                    model,
                    sample,
                    False,
                    eval_sample_index=eval_index,
                    eval_seed=eval_seed,
                    compute_val_loss=False,
                )
            )

        visual_metric_keys = (
            "psnr_rg",
            "ssim_rg",
            "psnr_rd",
            "ssim_rd",
            "psnr_dg",
            "ssim_dg",
            "decoded_target_l2",
            "decoded_target_l1",
            "action_l2",
            "action_l1",
            "decoded_position_mae_m",
            "decoded_position_rmse_m",
            "decoded_rotation_geodesic_deg",
            "decoded_gripper_mae",
            "decoded_gripper_accuracy",
        )
        metric_totals = []
        for key in visual_metric_keys:
            values = [
                float(sample_result[key])
                for sample_result in local_visual_results
                if key in sample_result
            ]
            metric_totals.extend((sum(values), len(values)))
        metric_totals = torch.tensor(
            metric_totals,
            device=self.accelerator.device,
            dtype=torch.float64,
        ).unsqueeze(0)
        gathered_totals = self.accelerator.gather(metric_totals).sum(dim=0)
        for key_index, key in enumerate(visual_metric_keys):
            total = float(gathered_totals[key_index * 2])
            count = int(gathered_totals[key_index * 2 + 1])
            if count:
                result[key] = total / count

        result["eval_sample_indices"] = [
            int(record["val_dataset_index"]) for record in records
        ]
        result["visual_eval_sample_indices"] = [
            int(record["val_dataset_index"]) for record in visual_records
        ]
        result["video_paths"] = [
            os.path.join(
                self.eval_dir,
                f"step_{self.global_step:06d}"
                f"_rank_{offset % self.accelerator.num_processes:03d}"
                f"_sample_{int(record['val_dataset_index']):06d}.mp4",
            )
            for offset, record in enumerate(visual_records)
        ]
        result["video_path"] = result["video_paths"][0]
        if was_dit_training:
            self._set_dit_only_train_mode()
        logger.info(
            "Evaluated fixed manifest: loss_samples=%d visual_samples=%d "
            "across %d rank(s).",
            len(records),
            len(visual_records),
            self.accelerator.num_processes,
        )
        return result

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        if getattr(model, "is_visual_action_model", False):
            if self.fixed_validation_manifest is not None:
                return self._evaluate_visual_action_manifest(model, was_dit_training)
            metric_keys = (
                "val_loss",
                "psnr_rg",
                "ssim_rg",
                "psnr_rd",
                "ssim_rd",
                "psnr_dg",
                "ssim_dg",
                "decoded_target_l2",
                "decoded_target_l1",
                "action_l2",
                "action_l1",
                "decoded_position_mae_m",
                "decoded_position_rmse_m",
                "decoded_rotation_geodesic_deg",
                "decoded_gripper_mae",
                "decoded_gripper_accuracy",
            )
            per_sample_results = []
            local_eval_indices = []
            # `eval_num_samples` is a global group size, independent of GPU
            # count.  Ranks split the fixed group in round-robin order.
            local_offsets = range(
                self.accelerator.process_index,
                self.eval_num_samples,
                self.accelerator.num_processes,
            )
            for sample_offset in local_offsets:
                eval_index = (
                    self.eval_sample_index
                    + sample_offset
                ) % len(self.val_dataset)
                eval_seed = self.eval_random_seed + sample_offset
                local_eval_indices.append(eval_index)
                sample = self._get_fixed_visual_action_eval_sample(
                    eval_index,
                    eval_seed,
                )
                per_sample_results.append(
                    self._evaluate_visual_action(
                        model,
                        sample,
                        False,
                        eval_sample_index=eval_index,
                        eval_seed=eval_seed,
                    )
                )
            if was_dit_training:
                self._set_dit_only_train_mode()
            logger.info(
                "Evaluated local fixed sample indices=%s (base=%d rank=%d/%d).",
                local_eval_indices,
                self.eval_sample_index,
                self.accelerator.process_index,
                self.accelerator.num_processes,
            )

            # Aggregate sums and valid counts once, so ranks may evaluate
            # different numbers of samples without entering per-sample
            # collectives.
            metric_totals = []
            for key in metric_keys:
                values = [
                    float(sample_result[key])
                    for sample_result in per_sample_results
                    if key in sample_result
                ]
                metric_totals.extend((sum(values), len(values)))
            metric_totals = torch.tensor(
                metric_totals,
                device=self.accelerator.device,
                dtype=torch.float64,
            ).unsqueeze(0)
            gathered_totals = self.accelerator.gather(metric_totals).sum(dim=0)

            result = {}
            for key_index, key in enumerate(metric_keys):
                total = float(gathered_totals[key_index * 2])
                count = int(gathered_totals[key_index * 2 + 1])
                if count:
                    result[key] = total / count

            eval_indices = [
                (self.eval_sample_index + offset) % len(self.val_dataset)
                for offset in range(self.eval_num_samples)
            ]
            result["video_paths"] = [
                os.path.join(
                    self.eval_dir,
                    f"step_{self.global_step:06d}"
                    f"_rank_{offset % self.accelerator.num_processes:03d}"
                    f"_sample_{eval_index:06d}.mp4",
                )
                for offset, eval_index in enumerate(eval_indices)
            ]
            # Retain the singular key for callers written before grouped eval.
            result["video_path"] = result["video_paths"][0]
            result["eval_sample_indices"] = eval_indices
            if self.accelerator.is_main_process:
                logger.info(
                    "Aggregated fixed validation group indices=%s seeds=%d..%d "
                    "across %d rank(s).",
                    eval_indices,
                    self.eval_random_seed,
                    self.eval_random_seed + self.eval_num_samples - 1,
                    self.accelerator.num_processes,
                )
            return result

        # Standard FastWAM keeps its existing one-sample evaluation behavior,
        # but uses the configured fixed index instead of changing per step.
        eval_index = (
            self.eval_sample_index + self.accelerator.process_index
        ) % len(self.val_dataset)
        logger.info(
            "Evaluating fixed sample index=%d (base=%d rank=%d).",
            eval_index,
            self.eval_sample_index,
            self.accelerator.process_index,
        )
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
            
            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
            "resume_compatibility": self._resume_compatibility_manifest(),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def _resume_compatibility_manifest(self) -> dict[str, object]:
        """Describe stateful training semantics; operational knobs are omitted."""
        cfg = OmegaConf.to_container(self.cfg, resolve=True)
        assert isinstance(cfg, dict)
        semantic = {
            "model": cfg.get("model"),
            "data": cfg.get("data"),
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "mixed_precision": self.mixed_precision,
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "lr_scheduler_type": cfg.get("lr_scheduler_type"),
            "max_grad_norm": self.max_grad_norm,
            "num_epochs": self.num_epochs,
            "max_steps": self.max_steps,
            "finetune": cfg.get("finetune"),
            "world_size": int(self.accelerator.num_processes),
        }
        encoded = json.dumps(
            semantic,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        ).encode("utf-8")
        return {
            "format_version": 1,
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "semantic_config": semantic,
        }

    def _validate_resume_compatibility(self, saved: object, state_dir: str) -> None:
        if not isinstance(saved, dict) or "sha256" not in saved:
            logger.warning(
                "Training state %s predates resume compatibility metadata; "
                "optimizer/dataloader state will be restored without semantic "
                "configuration authentication.",
                state_dir,
            )
            return
        current = self._resume_compatibility_manifest()
        if saved.get("sha256") == current["sha256"]:
            return
        if bool(self.cfg.get("resume_allow_config_mismatch", False)):
            logger.warning(
                "Resume semantic configuration mismatch was explicitly allowed: "
                "saved=%s current=%s state=%s",
                saved.get("sha256"),
                current["sha256"],
                state_dir,
            )
            return
        saved_semantic = saved.get("semantic_config")
        current_semantic = current["semantic_config"]
        changed = []
        if isinstance(saved_semantic, dict) and isinstance(current_semantic, dict):
            changed = sorted(
                key
                for key in set(saved_semantic) | set(current_semantic)
                if saved_semantic.get(key) != current_semantic.get(key)
            )
        raise ValueError(
            "Refusing incompatible training-state resume. "
            f"Changed semantic fields={changed or 'unknown'}; "
            f"saved_sha256={saved.get('sha256')} current_sha256={current['sha256']}. "
            "Only set resume_allow_config_mismatch=true after manually proving "
            "the optimizer/scheduler/dataloader state is compatible."
        )

    def save_checkpoint(self, *, save_weights: bool = True, save_state: bool = True):
        if not save_weights and not save_state:
            raise ValueError("At least one of `save_weights` or `save_state` must be true.")
        step_tag = f"step_{self.global_step:06d}"

        ckpt_path = None
        if save_weights:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
            self.accelerator.wait_for_everyone()

        state_path = None
        if save_state:
            state_path = os.path.join(self.state_dir, step_tag)
            ensure_dir(state_path)
            self.accelerator.save_state(output_dir=state_path)
            if self.accelerator.is_main_process:
                self._save_trainer_state(state_path)
            self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self._validate_resume_compatibility(
                payload.get("resume_compatibility"), str(state_dir)
            )
            self.accelerator.load_state(input_dir=state_dir)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        self.accelerator.load_state(input_dir=state_dir)

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        if self.eval_at_start:
            metrics = self.evaluate()
            self.accelerator.wait_for_everyone()
            if metrics is not None and self.accelerator.is_main_process:
                logger.info(
                    "[eval/start] step=%d val_loss=%.4f metrics=%s",
                    self.global_step,
                    metrics["val_loss"],
                    {
                        key: value
                        for key, value in metrics.items()
                        if isinstance(value, (int, float))
                    },
                )

        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                            )
                            if "psnr_rd" in metrics:
                                description += " infer_psnr=%.4f infer_ssim=%.4f" % (
                                    metrics["psnr_rd"],
                                    metrics["ssim_rd"],
                                )
                            if "psnr_dg" in metrics:
                                description += " vae_psnr=%.4f vae_ssim=%.4f" % (
                                    metrics["psnr_dg"],
                                    metrics["ssim_dg"],
                                )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            if "decoded_target_l2" in metrics:
                                description += " decoded_target_l2=%.4f" % metrics[
                                    "decoded_target_l2"
                                ]
                            if "decoded_target_l1" in metrics:
                                description += " decoded_target_l1=%.4f" % metrics[
                                    "decoded_target_l1"
                                ]
                            if "decoded_position_mae_m" in metrics:
                                description += " pos_mae_m=%.4f" % metrics[
                                    "decoded_position_mae_m"
                                ]
                            if "decoded_rotation_geodesic_deg" in metrics:
                                description += " rot_deg=%.2f" % metrics[
                                    "decoded_rotation_geodesic_deg"
                                ]
                            logger.info(description)
                            eval_payload = {"eval/val_loss": float(metrics["val_loss"])}
                            for key in (
                                "psnr_rg",
                                "ssim_rg",
                                "psnr_rd",
                                "ssim_rd",
                                "psnr_dg",
                                "ssim_dg",
                            ):
                                if key in metrics:
                                    eval_payload[f"eval/{key}"] = float(metrics[key])
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            if "decoded_target_l2" in metrics:
                                eval_payload["eval/decoded_target_l2"] = float(
                                    metrics["decoded_target_l2"]
                                )
                            if "decoded_target_l1" in metrics:
                                eval_payload["eval/decoded_target_l1"] = float(
                                    metrics["decoded_target_l1"]
                                )
                            for key in (
                                "decoded_position_mae_m",
                                "decoded_position_rmse_m",
                                "decoded_rotation_geodesic_deg",
                                "decoded_gripper_mae",
                                "decoded_gripper_accuracy",
                            ):
                                if key in metrics:
                                    eval_payload[f"eval/{key}"] = float(metrics[key])
                            self._wandb_log(eval_payload)

                    weights_saved_this_step = False
                    state_saved_this_step = False
                    weights_due = (
                        self.save_every > 0
                        and self.global_step % self.save_every == 0
                    )
                    state_due = (
                        self.state_save_every > 0
                        and self.global_step % self.state_save_every == 0
                    )
                    if weights_due or state_due:
                        ckpt_info = self.save_checkpoint(
                            save_weights=weights_due,
                            save_state=state_due,
                        )
                        weights_saved_this_step = weights_due
                        state_saved_this_step = state_due
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.max_steps:
                        if self.save_at_end:
                            if not (weights_saved_this_step and state_saved_this_step):
                                final_ckpt_info = self.save_checkpoint(
                                    save_weights=not weights_saved_this_step,
                                    save_state=not state_saved_this_step,
                                )
                                if weights_saved_this_step:
                                    final_ckpt_info["weights_path"] = ckpt_info["weights_path"]
                                if state_saved_this_step:
                                    final_ckpt_info["state_path"] = ckpt_info["state_path"]
                                ckpt_info = final_ckpt_info
                            if self.accelerator.is_main_process:
                                logger.info(
                                    "[done] max_steps reached step=%d weights=%s state=%s",
                                    self.global_step,
                                    ckpt_info["weights_path"],
                                    ckpt_info["state_path"],
                                )
                        elif self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d; final checkpoint disabled.",
                                self.global_step,
                            )
                        return

        if self.save_at_end:
            ckpt_info = self.save_checkpoint()
            if self.accelerator.is_main_process:
                logger.info(
                    "[done] training finished step=%d weights=%s state=%s",
                    self.global_step,
                    ckpt_info["weights_path"],
                    ckpt_info["state_path"],
                )
        elif self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d; final checkpoint disabled.",
                self.global_step,
            )
        
