from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

from fastwam.representations.rothko import RothkoCodec, RothkoCodecConfig
from fastwam.representations.libero_rothko import (
    LiberoRothkoCodec,
    LiberoRothkoCodecConfig,
)
from fastwam.utils.logging_config import get_logger

from ..lora import (
    LoRAConfig,
    count_lora_parameters,
    inject_lora,
    load_lora_state_dict,
    lora_state_dict,
)
from .helpers.loader import load_wan22_ti2v_5b_components
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


class FastWAMVideoOnlyRaymap(torch.nn.Module):
    """Wan video expert jointly predicting future RGB and Rothko raymaps.

    RGB and Rothko are independently encoded by the same frozen Wan VAE, then
    concatenated as two equally-sized latent-time blocks.  The two frame-zero
    latents are clean conditions; all future latents are jointly denoised by a
    single pretrained Wan video DiT using continuous temporal RoPE positions.
    """
    is_visual_action_model = True
    legacy_video_attention_mask_mode = "condition_frames_causal"

    def __init__(
        self,
        video_expert,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        train_shift: float = 5.0,
        infer_shift: float = 5.0,
        num_train_timesteps: int = 1000,
        loss_lambda_rgb: float = 1.0,
        loss_lambda_raymap: float = 1.0,
        action_horizon: int = 16,
        raymap_representation: str = "rothko",
        rothko_norm_stats: Optional[str] = None,
        rothko_config: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.dit = video_expert
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if text_encoder is None:
                raise ValueError("`text_dim` is required without a loaded text encoder.")
            text_dim = int(text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        self.proprio_encoder = (
            nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
            if self.proprio_dim is not None
            else None
        )

        self.action_horizon = int(action_horizon)
        temporal_factor = int(self.vae.temporal_downsample_factor)
        if self.action_horizon <= 0 or self.action_horizon % temporal_factor != 0:
            raise ValueError(
                "`action_horizon` must be a positive multiple of the VAE temporal "
                f"downsample factor {temporal_factor}, got {self.action_horizon}."
            )
        self.num_pixel_frames = self.action_horizon + 1
        self.num_latent_frames_per_modality = (
            self.num_pixel_frames - 1
        ) // temporal_factor + 1
        self.condition_latent_indices = (
            0,
            self.num_latent_frames_per_modality,
        )
        self.temporal_rope_mode = (
            f"continuous_0_{2 * self.num_latent_frames_per_modality - 1}"
        )

        self.train_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=train_shift,
        )
        self.infer_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=infer_shift,
        )
        self.loss_lambda_rgb = float(loss_lambda_rgb)
        self.loss_lambda_raymap = float(loss_lambda_raymap)

        self.raymap_representation = str(raymap_representation)
        codec_payload = {} if rothko_config is None else dict(rothko_config)
        if self.raymap_representation == "rothko":
            self.raymap_codec = RothkoCodec(
                config=RothkoCodecConfig(**codec_payload),
                norm_stats=rothko_norm_stats,
            )
        elif self.raymap_representation == "libero_rothko":
            self.raymap_codec = LiberoRothkoCodec(
                config=LiberoRothkoCodecConfig(**codec_payload),
                norm_stats=rothko_norm_stats,
                expected_action_horizon=self.action_horizon,
            )
        else:
            raise ValueError(
                "`raymap_representation` must be 'rothko' or 'libero_rothko', "
                f"got {self.raymap_representation!r}."
            )
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.base_model_id: Optional[str] = None
        self.lora_config: Optional[dict[str, Any]] = None
        self.lora_train_proprio_encoder = True
        self.to(self.device)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 128,
        load_text_encoder: bool = False,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        vae_safetensors_path: Optional[str] = None,
        video_dit_config: Optional[dict[str, Any]] = None,
        skip_dit_load_from_pretrain: bool = False,
        train_shift: float = 5.0,
        infer_shift: float = 5.0,
        num_train_timesteps: int = 1000,
        loss_lambda_rgb: float = 1.0,
        loss_lambda_raymap: float = 1.0,
        action_horizon: int = 16,
        raymap_representation: str = "rothko",
        rothko_norm_stats: Optional[str] = None,
        rothko_config: Optional[dict[str, Any]] = None,
    ) -> "FastWAMVideoOnlyRaymap":
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required.")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required.")
        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            vae_safetensors_path=vae_safetensors_path,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )
        model = cls(
            video_expert=components.dit,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            train_shift=train_shift,
            infer_shift=infer_shift,
            num_train_timesteps=num_train_timesteps,
            loss_lambda_rgb=loss_lambda_rgb,
            loss_lambda_raymap=loss_lambda_raymap,
            action_horizon=action_horizon,
            raymap_representation=raymap_representation,
            rothko_norm_stats=rothko_norm_stats,
            rothko_config=rothko_config,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
        }
        model.base_model_id = str(model_id)
        return model

    @torch.no_grad()
    def encode_prompt(
        self, prompt: Union[str, Sequence[str]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires a loaded text encoder/tokenizer; "
                "otherwise provide cached context/context_mask."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        context = self.text_encoder(ids, mask)
        sequence_lengths = mask.gt(0).sum(dim=1).long()
        for index, length in enumerate(sequence_lengths):
            context[index, length:] = 0
        return context.to(device=self.device), torch.ones_like(mask)

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None:
            return context, context_mask
        if proprio is None:
            raise ValueError("`proprio` is required when proprio_dim is enabled.")
        if proprio.ndim != 2 or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"Expected proprio [B,{self.proprio_dim}], got {tuple(proprio.shape)}"
            )
        token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype)
        token_mask = torch.ones(
            (context_mask.shape[0], 1),
            dtype=torch.bool,
            device=context_mask.device,
        )
        return (
            torch.cat((context, token), dim=1),
            torch.cat((context_mask, token_mask), dim=1),
        )

    @torch.no_grad()
    def _encode_video_latents(self, video: torch.Tensor, tiled: bool = False) -> torch.Tensor:
        return self.vae.encode(video, device=self.device, tiled=tiled)

    @torch.no_grad()
    def _decode_video_tensor(
        self, latents: torch.Tensor, tiled: bool = False
    ) -> torch.Tensor:
        return (
            self.vae.decode(latents, device=self.device, tiled=tiled)
            .detach()
            .float()
            .clamp(-1, 1)
        )

    @staticmethod
    def _video_tensor_to_pil(video: torch.Tensor) -> list[Image.Image]:
        if video.ndim != 4 or video.shape[0] != 3:
            raise ValueError(f"Expected [3,T,H,W], got {tuple(video.shape)}")
        uint8 = ((video + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu()
        return [
            Image.fromarray(uint8[:, index].permute(1, 2, 0).numpy())
            for index in range(uint8.shape[1])
        ]

    def build_inputs(self, sample: dict[str, Any], tiled: bool = False) -> dict[str, Any]:
        video = sample.get("video")
        raymap = sample.get("raymap")
        if not isinstance(video, torch.Tensor) or not isinstance(raymap, torch.Tensor):
            raise TypeError("Video-only training requires tensor `video` and `raymap`.")
        if video.ndim != 5 or raymap.ndim != 5:
            raise ValueError(
                f"Expected video/raymap [B,3,T,H,W], got {video.shape} and {raymap.shape}"
            )
        if video.shape != raymap.shape or video.shape[1] != 3:
            raise ValueError(
                f"RGB/Rothko shapes must match and have 3 channels, got "
                f"{tuple(video.shape)} and {tuple(raymap.shape)}"
            )
        if video.shape[2] != self.num_pixel_frames:
            raise ValueError(
                f"Expected {self.num_pixel_frames} pixel frames, got {video.shape[2]}."
            )
        if video.shape[3] % 16 or video.shape[4] % 16:
            raise ValueError("RGB/Rothko spatial dimensions must be multiples of 16.")
        context = sample.get("context")
        context_mask = sample.get("context_mask")
        if not isinstance(context, torch.Tensor) or not isinstance(
            context_mask, torch.Tensor
        ):
            raise ValueError("Cached context and context_mask are required.")
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"Expected context [B,L,D]/[B,L], got "
                f"{tuple(context.shape)} and {tuple(context_mask.shape)}"
            )

        rgb = video.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        raymap = raymap.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        rgb_latents = self._encode_video_latents(rgb, tiled=tiled)
        raymap_latents = self._encode_video_latents(raymap, tiled=tiled)
        if rgb_latents.shape != raymap_latents.shape:
            raise ValueError(
                f"RGB/Rothko latent shapes differ: "
                f"{tuple(rgb_latents.shape)} vs {tuple(raymap_latents.shape)}"
            )
        if rgb_latents.shape[2] != self.num_latent_frames_per_modality:
            raise ValueError(
                f"Expected {self.num_latent_frames_per_modality} latent frames, "
                f"got {rgb_latents.shape[2]}."
            )

        context = context.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        context_mask = context_mask.to(
            device=self.device, dtype=torch.bool, non_blocking=True
        )
        proprio = sample.get("proprio")
        if self.proprio_encoder is not None:
            if not isinstance(proprio, torch.Tensor) or proprio.ndim != 3:
                raise ValueError("Expected proprio [B,T,D].")
            context, context_mask = self._append_proprio_to_context(
                context,
                context_mask,
                proprio[:, 0].to(
                    device=self.device,
                    dtype=self.torch_dtype,
                    non_blocking=True,
                ),
            )
        return {
            "context": context,
            "context_mask": context_mask,
            "rgb_latents": rgb_latents,
            "raymap_latents": raymap_latents,
            "joint_latents": torch.cat((rgb_latents, raymap_latents), dim=2),
            "image_is_pad": self._optional_bool_to_device(
                sample.get("image_is_pad")
            ),
            "raymap_is_pad": self._optional_bool_to_device(
                sample.get("raymap_is_pad")
            ),
        }

    def _optional_bool_to_device(
        self, value: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if value is None:
            return None
        return value.to(device=self.device, dtype=torch.bool, non_blocking=True)

    def _model_fn(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.video_expert(
            x=latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=True,
            condition_latent_indices=self.condition_latent_indices,
        )

    def _latent_future_valid_mask(
        self, pixel_is_pad: Optional[torch.Tensor], batch_size: int
    ) -> torch.Tensor:
        future_latent_frames = self.num_latent_frames_per_modality - 1
        if pixel_is_pad is None:
            return torch.ones(
                (batch_size, future_latent_frames),
                dtype=torch.bool,
                device=self.device,
            )
        if pixel_is_pad.shape != (batch_size, self.num_pixel_frames):
            raise ValueError(
                f"Expected pad mask {(batch_size, self.num_pixel_frames)}, "
                f"got {tuple(pixel_is_pad.shape)}."
            )
        factor = int(self.vae.temporal_downsample_factor)
        latent_future_is_pad = pixel_is_pad[:, 1:].reshape(
            batch_size, future_latent_frames, factor
        ).all(dim=2)
        return ~latent_future_is_pad

    @staticmethod
    def _masked_future_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        token_loss = F.mse_loss(
            prediction.float(), target.float(), reduction="none"
        ).mean(dim=(1, 3, 4))
        valid_float = valid.to(device=token_loss.device, dtype=token_loss.dtype)
        return (token_loss * valid_float).sum(dim=1) / valid_float.sum(
            dim=1
        ).clamp_min(1)

    def training_loss(
        self, sample: dict[str, Any], tiled: bool = False
    ) -> tuple[torch.Tensor, dict[str, float]]:
        inputs = self.build_inputs(sample, tiled=tiled)
        clean = inputs["joint_latents"]
        batch_size = clean.shape[0]
        noise = torch.randn_like(clean)
        timestep = self.train_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=clean.dtype,
        )
        noisy = self.train_scheduler.add_noise(clean, noise, timestep)
        target = self.train_scheduler.training_target(clean, noise, timestep)
        for index in self.condition_latent_indices:
            noisy[:, :, index : index + 1] = clean[:, :, index : index + 1]

        prediction = self._model_fn(
            noisy,
            timestep,
            inputs["context"],
            inputs["context_mask"],
        )
        split = self.num_latent_frames_per_modality
        prediction_rgb = prediction[:, :, 1:split]
        target_rgb = target[:, :, 1:split]
        prediction_raymap = prediction[:, :, split + 1 : 2 * split]
        target_raymap = target[:, :, split + 1 : 2 * split]
        rgb_valid = self._latent_future_valid_mask(
            inputs["image_is_pad"], batch_size
        )
        raymap_valid = self._latent_future_valid_mask(
            inputs["raymap_is_pad"], batch_size
        )
        loss_rgb_per_sample = self._masked_future_loss(
            prediction_rgb, target_rgb, rgb_valid
        )
        loss_raymap_per_sample = self._masked_future_loss(
            prediction_raymap, target_raymap, raymap_valid
        )
        weight = self.train_scheduler.training_weight(timestep).to(
            device=self.device, dtype=loss_rgb_per_sample.dtype
        )
        loss_rgb = (loss_rgb_per_sample * weight).mean()
        loss_raymap = (loss_raymap_per_sample * weight).mean()
        total = (
            self.loss_lambda_rgb * loss_rgb
            + self.loss_lambda_raymap * loss_raymap
        )
        return total, {
            "loss_total": float(total.detach()),
            "loss_rgb_raw": float(loss_rgb.detach()),
            "loss_raymap_raw": float(loss_raymap.detach()),
            "loss_rgb": self.loss_lambda_rgb * float(loss_rgb.detach()),
            "loss_raymap": self.loss_lambda_raymap * float(loss_raymap.detach()),
        }

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        input_raymap: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        current_endpose: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        self.eval()
        if num_frames is None:
            num_frames = self.num_pixel_frames
        if int(num_frames) != self.num_pixel_frames:
            raise ValueError(
                f"Video-only inference requires num_frames={self.num_pixel_frames}."
            )
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_raymap.ndim == 3:
            input_raymap = input_raymap.unsqueeze(0)
        if (
            input_image.ndim != 4
            or input_raymap.ndim != 4
            or input_image.shape != input_raymap.shape
            or input_image.shape[1] != 3
        ):
            raise ValueError(
                f"Expected matching [B,3,H,W] RGB/Rothko conditions, got "
                f"{tuple(input_image.shape)} and {tuple(input_raymap.shape)}"
            )
        if input_image.shape[0] != 1:
            raise ValueError("The first inference implementation supports batch size 1.")
        height, width = input_image.shape[-2:]
        if height % 16 or width % 16:
            raise ValueError("Inference H/W must be multiples of 16.")

        rgb_condition = self._encode_video_latents(
            input_image.to(self.device, self.torch_dtype).unsqueeze(2), tiled=tiled
        )
        raymap_condition = self._encode_video_latents(
            input_raymap.to(self.device, self.torch_dtype).unsqueeze(2), tiled=tiled
        )
        latent_height = height // int(self.vae.upsampling_factor)
        latent_width = width // int(self.vae.upsampling_factor)
        generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )
        latents = torch.randn(
            (
                1,
                int(self.vae.model.z_dim),
                2 * self.num_latent_frames_per_modality,
                latent_height,
                latent_width,
            ),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents[:, :, 0:1] = rgb_condition
        ray_condition_index = self.num_latent_frames_per_modality
        latents[:, :, ray_condition_index : ray_condition_index + 1] = (
            raymap_condition
        )

        if context is None or context_mask is None:
            if prompt is None:
                raise ValueError("Provide either prompt or cached context/context_mask.")
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(self.device, self.torch_dtype)
            context_mask = context_mask.to(self.device, torch.bool)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("proprio is required.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            context, context_mask = self._append_proprio_to_context(
                context, context_mask, proprio
            )

        timesteps, deltas = self.infer_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents.dtype,
            shift_override=sigma_shift,
        )
        for step_t, delta in zip(timesteps, deltas):
            timestep = step_t.reshape(1).to(self.device, latents.dtype)
            prediction = self._model_fn(
                latents, timestep, context, context_mask
            )
            latents = self.infer_scheduler.step(prediction, delta, latents)
            latents[:, :, 0:1] = rgb_condition
            latents[
                :,
                :,
                ray_condition_index : ray_condition_index + 1,
            ] = raymap_condition

        split = self.num_latent_frames_per_modality
        decoded_rgb = self._decode_video_tensor(latents[:, :, :split], tiled=tiled)
        decoded_raymap = self._decode_video_tensor(
            latents[:, :, split:], tiled=tiled
        )
        result: dict[str, Any] = {
            "video": self._video_tensor_to_pil(decoded_rgb[0]),
            "video_tensor": decoded_rgb.cpu(),
            "raymap": decoded_raymap.cpu(),
        }
        if current_endpose is not None:
            if current_endpose.ndim == 1:
                current_endpose = current_endpose.unsqueeze(0)
            pose, gripper = self.raymap_codec.decode(
                decoded_raymap,
                current_endpose.to(
                    device=decoded_raymap.device,
                    dtype=torch.float32,
                ),
            )
            result.update(
                {
                    "pose": pose.cpu(),
                    "gripper": gripper.cpu(),
                }
            )
            if (
                int(getattr(self.raymap_codec, "pose_dim", 14)) == 14
                and int(getattr(self.raymap_codec, "gripper_dim", 2)) == 2
            ):
                ee_action = torch.cat(
                    (
                        pose[:, 1:, :7],
                        gripper[:, 1:, 0:1],
                        pose[:, 1:, 7:14],
                        gripper[:, 1:, 1:2],
                    ),
                    dim=-1,
                )
                result["action"] = ee_action.cpu()
        return result

    @property
    def is_lora_enabled(self) -> bool:
        return self.lora_config is not None

    def enable_lora(
        self,
        config: LoRAConfig | dict[str, Any],
        *,
        train_proprio_encoder: bool = True,
    ) -> dict[str, Any]:
        if not isinstance(config, LoRAConfig):
            config = LoRAConfig.from_dict(config)
        module_names = inject_lora(self.dit, config)
        self.lora_config = config.to_dict()
        self.lora_train_proprio_encoder = bool(train_proprio_encoder)
        parameter_count, module_count = count_lora_parameters(self.dit)
        if module_count != len(module_names):
            raise RuntimeError(
                f"LoRA module count mismatch: {module_count} vs {len(module_names)}."
            )
        logger.info(
            "Enabled LoRA on video DiT: modules=%d parameters=%d rank=%d "
            "alpha=%.4f dropout=%.4f train_proprio_encoder=%s",
            module_count,
            parameter_count,
            config.rank,
            config.alpha,
            config.dropout,
            self.lora_train_proprio_encoder,
        )
        return {
            **config.to_dict(),
            "module_count": module_count,
            "parameter_count": parameter_count,
            "train_proprio_encoder": self.lora_train_proprio_encoder,
        }

    def _visual_action_checkpoint_config(self) -> dict[str, Any]:
        video_attention_mask_mode = str(
            getattr(self.video_expert, "video_attention_mask_mode", "")
        ).strip()
        if not video_attention_mask_mode:
            raise ValueError(
                "Cannot save a video-only checkpoint without "
                "`video_expert.video_attention_mask_mode`."
            )
        return {
            **self.raymap_codec.metadata(),
            "raymap_representation": self.raymap_representation,
            "action_horizon": self.action_horizon,
            "latent_layout": "rgb_then_raymap",
            "temporal_rope_mode": self.temporal_rope_mode,
            "condition_latent_indices": list(self.condition_latent_indices),
            "video_attention_mask_mode": video_attention_mask_mode,
            "loss_weights": {
                "rgb": self.loss_lambda_rgb,
                "raymap": self.loss_lambda_raymap,
            },
        }

    def _validate_visual_action_checkpoint_config(
        self,
        visual_config: dict[str, Any],
        *,
        checkpoint_path: str,
    ) -> None:
        expected_attention_mode = str(
            getattr(self.video_expert, "video_attention_mask_mode", "")
        ).strip()
        if not expected_attention_mode:
            raise ValueError(
                "Cannot load a video-only checkpoint without "
                "`video_expert.video_attention_mask_mode`."
            )

        checkpoint_attention_mode = visual_config.get(
            "video_attention_mask_mode"
        )
        if checkpoint_attention_mode is None:
            run_config_mode = None
            run_config_path = None
            checkpoint = Path(checkpoint_path).resolve()
            for parent in list(checkpoint.parents)[:4]:
                candidate = parent / "config.yaml"
                if not candidate.is_file():
                    continue
                run_config = OmegaConf.load(candidate)
                configured_mode = OmegaConf.select(
                    run_config,
                    "model.video_dit_config.video_attention_mask_mode",
                )
                if configured_mode is not None:
                    run_config_mode = str(configured_mode).strip()
                    run_config_path = candidate
                break

            if run_config_mode:
                checkpoint_attention_mode = run_config_mode
                logger.warning(
                    "Checkpoint %s predates attention-mask metadata; recovered "
                    "video_attention_mask_mode=%s from run config %s.",
                    checkpoint_path,
                    checkpoint_attention_mode,
                    run_config_path,
                )
            else:
                checkpoint_attention_mode = self.legacy_video_attention_mask_mode
                logger.warning(
                    "Checkpoint %s predates attention-mask metadata and has no "
                    "usable run config; treating it as legacy "
                    "video_attention_mask_mode=%s.",
                    checkpoint_path,
                    checkpoint_attention_mode,
                )
        checkpoint_attention_mode = str(checkpoint_attention_mode).strip()
        if checkpoint_attention_mode != expected_attention_mode:
            raise ValueError(
                "Checkpoint attention-mask semantics mismatch: "
                f"checkpoint={checkpoint_attention_mode!r}, "
                f"model={expected_attention_mode!r}. Legacy checkpoints without "
                "`video_attention_mask_mode` were trained with "
                f"{self.legacy_video_attention_mask_mode!r}. Use the matching "
                "model.video_dit_config.video_attention_mask_mode override, or "
                "start a fresh training run."
            )

        expected_metadata = {
            "action_horizon": self.action_horizon,
            "latent_layout": "rgb_then_raymap",
            "temporal_rope_mode": self.temporal_rope_mode,
            "condition_latent_indices": list(self.condition_latent_indices),
        }
        for key, expected in expected_metadata.items():
            actual = visual_config.get(key)
            if actual != expected:
                raise ValueError(
                    f"Checkpoint metadata mismatch for {key}: "
                    f"checkpoint={actual!r}, model={expected!r}."
                )
        checkpoint_representation = visual_config.get("raymap_representation")
        if checkpoint_representation is None:
            # Video-only checkpoints created before LIBERO support are
            # unambiguously dual-arm RoboTwin Rothko checkpoints.
            checkpoint_representation = "rothko"
        if str(checkpoint_representation) != self.raymap_representation:
            raise ValueError(
                "Checkpoint raymap representation mismatch: "
                f"checkpoint={checkpoint_representation!r}, "
                f"model={self.raymap_representation!r}."
            )

    def save_checkpoint(self, path, optimizer=None, step=None) -> None:
        payload: dict[str, Any] = {
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "visual_action_config": self._visual_action_checkpoint_config(),
        }
        if self.is_lora_enabled:
            adapter_state = lora_state_dict(self.dit)
            parameter_count, module_count = count_lora_parameters(self.dit)
            payload.update(
                {
                    "checkpoint_type": "lora_adapter",
                    "lora": adapter_state,
                    "fine_tuning": {
                        "method": "lora",
                        "base_model_id": self.base_model_id,
                        "lora": {
                            **dict(self.lora_config or {}),
                            "module_count": module_count,
                            "parameter_count": parameter_count,
                        },
                        "train_proprio_encoder": self.lora_train_proprio_encoder,
                    },
                }
            )
        else:
            payload["checkpoint_type"] = "full"
            payload["dit"] = self.dit.state_dict()
        if self.proprio_encoder is not None:
            proprio_state = self.proprio_encoder.state_dict()
            if self.is_lora_enabled:
                proprio_state = {
                    key: value.detach().cpu().clone()
                    for key, value in proprio_state.items()
                }
            payload["proprio_encoder"] = proprio_state
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None) -> dict[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        visual_config = payload.get("visual_action_config")
        if not isinstance(visual_config, dict):
            raise ValueError(
                "Checkpoint is not a video-only Rothko checkpoint because "
                f"`visual_action_config` is missing: {path}"
            )
        self._validate_visual_action_checkpoint_config(
            visual_config,
            checkpoint_path=str(path),
        )

        checkpoint_type = payload.get("checkpoint_type")
        if "lora" in payload:
            if checkpoint_type not in (None, "lora_adapter"):
                raise ValueError(
                    f"LoRA payload has invalid checkpoint_type={checkpoint_type!r}."
                )
            fine_tuning = payload.get("fine_tuning")
            if not isinstance(fine_tuning, dict):
                raise ValueError(
                    f"LoRA checkpoint is missing `fine_tuning` metadata: {path}"
                )
            if fine_tuning.get("method") != "lora":
                raise ValueError(
                    f"Unsupported fine_tuning metadata in checkpoint: {fine_tuning}"
                )
            lora_config_payload = fine_tuning.get("lora")
            if not isinstance(lora_config_payload, dict):
                raise ValueError(
                    f"LoRA checkpoint is missing its adapter config: {path}"
                )
            checkpoint_base_model_id = fine_tuning.get("base_model_id")
            if (
                checkpoint_base_model_id
                and self.base_model_id
                and str(checkpoint_base_model_id) != str(self.base_model_id)
            ):
                raise ValueError(
                    "LoRA base model mismatch: "
                    f"checkpoint={checkpoint_base_model_id!r}, "
                    f"model={self.base_model_id!r}."
                )
            self.enable_lora(
                lora_config_payload,
                train_proprio_encoder=bool(
                    fine_tuning.get("train_proprio_encoder", True)
                ),
            )
            parameter_count, module_count = count_lora_parameters(self.dit)
            expected_module_count = lora_config_payload.get("module_count")
            expected_parameter_count = lora_config_payload.get("parameter_count")
            if (
                expected_module_count is not None
                and int(expected_module_count) != module_count
            ):
                raise ValueError(
                    "LoRA module count mismatch: "
                    f"checkpoint={expected_module_count}, model={module_count}."
                )
            if (
                expected_parameter_count is not None
                and int(expected_parameter_count) != parameter_count
            ):
                raise ValueError(
                    "LoRA parameter count mismatch: "
                    f"checkpoint={expected_parameter_count}, "
                    f"model={parameter_count}."
                )
            load_lora_state_dict(self.dit, payload["lora"])
        elif "dit" in payload:
            if checkpoint_type not in (None, "full"):
                raise ValueError(
                    f"Full DiT payload has invalid checkpoint_type={checkpoint_type!r}."
                )
            if self.is_lora_enabled:
                raise ValueError(
                    "Cannot load a full fine-tuned DiT checkpoint into an "
                    "adapter-only LoRA run. The resulting adapter would depend "
                    "on that external fine-tuned base and would not be portable. "
                    "Start LoRA from the configured original Wan2.2 base, or "
                    "resume from a LoRA checkpoint/state directory."
                )
            self.dit.load_state_dict(payload["dit"], strict=True)
        else:
            raise ValueError(
                f"Video-only checkpoint is missing both `dit` and `lora`: {path}"
            )

        if self.proprio_encoder is not None:
            if "proprio_encoder" not in payload:
                raise ValueError(
                    f"Checkpoint is missing required `proprio_encoder`: {path}"
                )
            self.proprio_encoder.load_state_dict(
                payload["proprio_encoder"], strict=True
            )
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
