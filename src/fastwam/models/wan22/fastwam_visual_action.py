from __future__ import annotations

import hashlib
import json
import math
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
from .helpers.loader import load_wan_video_components
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .wan_video_dit import (
    FRAME_ROLE_RAYMAP_CONDITION,
    FRAME_ROLE_RAYMAP_FUTURE,
    FRAME_ROLE_RGB_CONDITION,
    FRAME_ROLE_RGB_FUTURE,
)

logger = get_logger(__name__)


LATENT_LAYOUT_RGB_THEN_RAYMAP = "rgb_then_raymap"
LATENT_LAYOUT_RGB_RAYMAP_CHANNEL = "rgb_raymap_channel"
SUPPORTED_LATENT_LAYOUTS = {
    LATENT_LAYOUT_RGB_THEN_RAYMAP,
    LATENT_LAYOUT_RGB_RAYMAP_CHANNEL,
}
CHANNEL_LAYOUT_IO_INIT_DUPLICATE_SQRT2 = (
    "duplicate_sqrt2_input_duplicate_output"
)
CHANNEL_LAYOUT_IO_INIT_RGB_PRESERVE = (
    "rgb_preserve_zero_ray_input_duplicate_output"
)
SUPPORTED_CHANNEL_LAYOUT_IO_INITS = {
    CHANNEL_LAYOUT_IO_INIT_DUPLICATE_SQRT2,
    CHANNEL_LAYOUT_IO_INIT_RGB_PRESERVE,
}
FUTURE_RGB_MODE_JOINT = "joint"
FUTURE_RGB_MODE_TRAIN_ONLY_AUXILIARY = "train_only_auxiliary"
SUPPORTED_FUTURE_RGB_MODES = {
    FUTURE_RGB_MODE_JOINT,
    FUTURE_RGB_MODE_TRAIN_ONLY_AUXILIARY,
}
INFERENCE_TEMPORAL_POSITION_FULL_JOINT = "full_joint_contiguous"
INFERENCE_TEMPORAL_POSITION_PRUNED_PRESERVE = (
    "pruned_preserve_training_positions"
)


@torch.no_grad()
def _expand_video_dit_io_for_rgb_raymap_channels(
    dit: nn.Module,
    *,
    target_channels: int,
    init_mode: str = CHANNEL_LAYOUT_IO_INIT_DUPLICATE_SQRT2,
) -> None:
    """Expand a pretrained Wan DiT's latent-facing layers by a factor of two.

    ``duplicate_sqrt2_input_duplicate_output`` duplicates the input convolution
    across the RGB/Rothko halves and scales both copies by ``1/sqrt(2)``.

    ``rgb_preserve_zero_ray_input_duplicate_output`` copies the pretrained
    input convolution into the RGB half and initializes the Rothko half to
    zero. This exactly preserves the pretrained RGB patch features at step
    zero while leaving both halves trainable.

    Both modes duplicate output rows *within every spatiotemporal patch
    position* so unpatchify retains the expected
    ``[patch_t, patch_h, patch_w, channel]`` ordering.
    """
    init_mode = str(init_mode)
    if init_mode not in SUPPORTED_CHANNEL_LAYOUT_IO_INITS:
        raise ValueError(
            f"Unsupported channel-layout I/O init_mode={init_mode!r}; "
            f"expected one of {sorted(SUPPORTED_CHANNEL_LAYOUT_IO_INITS)}."
        )
    patch_embedding = getattr(dit, "patch_embedding", None)
    head = getattr(getattr(dit, "head", None), "head", None)
    patch_size = tuple(int(value) for value in getattr(dit, "patch_size", ()))
    if not isinstance(patch_embedding, nn.Conv3d) or not isinstance(
        head, nn.Linear
    ):
        raise TypeError(
            "Channel layout requires Conv3d patch embedding and Linear output head."
        )
    if len(patch_size) != 3:
        raise ValueError(f"Invalid DiT patch size for channel expansion: {patch_size}")

    source_channels = int(patch_embedding.in_channels)
    target_channels = int(target_channels)
    if target_channels != 2 * source_channels:
        raise ValueError(
            "RGB/Rothko channel layout must double the pretrained DiT latent "
            f"channels, got source={source_channels}, target={target_channels}."
        )
    patch_volume = math.prod(patch_size)
    if head.out_features != source_channels * patch_volume:
        raise ValueError(
            "Pretrained DiT output head is inconsistent with its input channels: "
            f"out_features={head.out_features}, source_channels={source_channels}, "
            f"patch_volume={patch_volume}."
        )

    expanded_patch = nn.Conv3d(
        in_channels=target_channels,
        out_channels=patch_embedding.out_channels,
        kernel_size=patch_embedding.kernel_size,
        stride=patch_embedding.stride,
        padding=patch_embedding.padding,
        dilation=patch_embedding.dilation,
        groups=patch_embedding.groups,
        bias=patch_embedding.bias is not None,
        padding_mode=patch_embedding.padding_mode,
        device=patch_embedding.weight.device,
        dtype=patch_embedding.weight.dtype,
    )
    if init_mode == CHANNEL_LAYOUT_IO_INIT_DUPLICATE_SQRT2:
        scaled_weight = patch_embedding.weight / math.sqrt(2.0)
        expanded_patch.weight[:, :source_channels].copy_(scaled_weight)
        expanded_patch.weight[:, source_channels:].copy_(scaled_weight)
    else:
        expanded_patch.weight[:, :source_channels].copy_(patch_embedding.weight)
        expanded_patch.weight[:, source_channels:].zero_()
    if patch_embedding.bias is not None:
        expanded_patch.bias.copy_(patch_embedding.bias)

    expanded_head = nn.Linear(
        in_features=head.in_features,
        out_features=target_channels * patch_volume,
        bias=head.bias is not None,
        device=head.weight.device,
        dtype=head.weight.dtype,
    )
    source_weight = head.weight.reshape(
        patch_volume, source_channels, head.in_features
    )
    expanded_weight = torch.cat((source_weight, source_weight), dim=1)
    expanded_head.weight.copy_(expanded_weight.reshape_as(expanded_head.weight))
    if head.bias is not None:
        source_bias = head.bias.reshape(patch_volume, source_channels)
        expanded_bias = torch.cat((source_bias, source_bias), dim=1)
        expanded_head.bias.copy_(expanded_bias.reshape_as(expanded_head.bias))

    dit.patch_embedding = expanded_patch
    dit.head.head = expanded_head
    dit.in_dim = target_channels
    logger.info(
        "Expanded pretrained video DiT latent I/O for RGB/Rothko channel layout: "
        "channels=%d->%d init=%s",
        source_channels,
        target_channels,
        init_mode,
    )


class FastWAMVideoOnlyRaymap(torch.nn.Module):
    """Wan video expert jointly predicting future RGB and Rothko raymaps.

    RGB and Rothko are independently encoded by the same frozen Wan VAE. The
    legacy/default layout concatenates them as two latent-time blocks; the
    opt-in channel layout pairs both modalities at each latent time step. Clean
    frame-zero conditions and all noisy future latents are jointly processed by
    a single pretrained Wan video DiT.
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
        rothko_decode_mode: str = "legacy",
        rothko_decode_anchor_alpha: float = 0.0,
        rothko_decode_block_grid: int = 4,
        latent_layout: str = LATENT_LAYOUT_RGB_THEN_RAYMAP,
        channel_io_init: str = CHANNEL_LAYOUT_IO_INIT_DUPLICATE_SQRT2,
        future_rgb_mode: str = FUTURE_RGB_MODE_JOINT,
        inference_predict_future_rgb: bool = False,
        allow_vae_mismatch: bool = False,
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
        self.vae_latent_channels = int(self.vae.model.z_dim)
        self.latent_layout = str(latent_layout)
        if self.latent_layout not in SUPPORTED_LATENT_LAYOUTS:
            raise ValueError(
                f"Unsupported latent_layout={self.latent_layout!r}; "
                f"expected one of {sorted(SUPPORTED_LATENT_LAYOUTS)}."
            )
        if self.latent_layout == LATENT_LAYOUT_RGB_THEN_RAYMAP:
            self.condition_latent_indices = (
                0,
                self.num_latent_frames_per_modality,
            )
            joint_latent_frames = 2 * self.num_latent_frames_per_modality
            expected_dit_channels = self.vae_latent_channels
            self.pretrained_io_expansion = None
        else:
            channel_io_init = str(channel_io_init)
            if channel_io_init not in SUPPORTED_CHANNEL_LAYOUT_IO_INITS:
                raise ValueError(
                    f"Unsupported channel_io_init={channel_io_init!r}; expected "
                    f"one of {sorted(SUPPORTED_CHANNEL_LAYOUT_IO_INITS)}."
                )
            self.condition_latent_indices = (0,)
            joint_latent_frames = self.num_latent_frames_per_modality
            expected_dit_channels = 2 * self.vae_latent_channels
            self.pretrained_io_expansion = channel_io_init
            attention_mode = str(
                getattr(self.video_expert, "video_attention_mask_mode", "")
            )
            if attention_mode != "bidirectional":
                raise ValueError(
                    "The first RGB/Rothko channel-layout implementation requires "
                    "video_attention_mask_mode='bidirectional', got "
                    f"{attention_mode!r}."
                )
        self.temporal_rope_mode = f"continuous_0_{joint_latent_frames - 1}"
        self.future_rgb_mode = str(future_rgb_mode)
        if self.future_rgb_mode not in SUPPORTED_FUTURE_RGB_MODES:
            raise ValueError(
                f"Unsupported future_rgb_mode={self.future_rgb_mode!r}; expected "
                f"one of {sorted(SUPPORTED_FUTURE_RGB_MODES)}."
            )
        attention_mode = str(
            getattr(self.video_expert, "video_attention_mask_mode", "")
        )
        if self.future_rgb_mode == FUTURE_RGB_MODE_TRAIN_ONLY_AUXILIARY:
            if self.latent_layout != LATENT_LAYOUT_RGB_THEN_RAYMAP:
                raise ValueError(
                    "Training-only future RGB requires the legacy RGB-then-Raymap "
                    "time layout; channel concatenation cannot prune RGB at inference."
                )
            if attention_mode != "independent_rgb_aux_ray":
                raise ValueError(
                    "Training-only future RGB requires "
                    "video_attention_mask_mode='independent_rgb_aux_ray', got "
                    f"{attention_mode!r}."
                )
            if int(self.video_expert.patch_size[0]) != 1:
                raise ValueError(
                    "Ray-only inference pruning requires temporal DiT patch size 1, "
                    f"got patch_size={self.video_expert.patch_size}."
                )
            self.inference_temporal_position_mode = (
                INFERENCE_TEMPORAL_POSITION_PRUNED_PRESERVE
            )
        else:
            if attention_mode == "independent_rgb_aux_ray":
                raise ValueError(
                    "video_attention_mask_mode='independent_rgb_aux_ray' is only "
                    "valid with future_rgb_mode='train_only_auxiliary'."
                )
            self.inference_temporal_position_mode = (
                INFERENCE_TEMPORAL_POSITION_FULL_JOINT
            )
        self.inference_predict_future_rgb = bool(inference_predict_future_rgb)
        if (
            self.inference_predict_future_rgb
            and self.future_rgb_mode != FUTURE_RGB_MODE_TRAIN_ONLY_AUXILIARY
        ):
            raise ValueError(
                "`inference_predict_future_rgb=true` is only meaningful when "
                "future_rgb_mode='train_only_auxiliary'."
            )

        actual_dit_in = int(getattr(self.video_expert, "in_dim", -1))
        patch_volume = math.prod(
            tuple(int(value) for value in self.video_expert.patch_size)
        )
        actual_dit_out = int(self.video_expert.head.head.out_features) // patch_volume
        if (
            actual_dit_in != expected_dit_channels
            or actual_dit_out != expected_dit_channels
        ):
            raise ValueError(
                "DiT latent I/O channels do not match the selected layout: "
                f"layout={self.latent_layout}, expected={expected_dit_channels}, "
                f"in={actual_dit_in}, out={actual_dit_out}."
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
                decode_mode=rothko_decode_mode,
                decode_anchor_alpha=rothko_decode_anchor_alpha,
                decode_block_grid=rothko_decode_block_grid,
            )
        elif self.raymap_representation == "libero_rothko":
            self.raymap_codec = LiberoRothkoCodec(
                config=LiberoRothkoCodecConfig(**codec_payload),
                norm_stats=rothko_norm_stats,
                expected_action_horizon=self.action_horizon,
                decode_mode=rothko_decode_mode,
                decode_anchor_alpha=rothko_decode_anchor_alpha,
                decode_block_grid=rothko_decode_block_grid,
            )
        else:
            raise ValueError(
                "`raymap_representation` must be 'rothko' or 'libero_rothko', "
                f"got {self.raymap_representation!r}."
            )
        # Decoding is an inference-time policy choice, not part of the learned
        # representation geometry or checkpoint compatibility contract.
        self.rothko_decode_mode = self.raymap_codec.decode_mode
        self.rothko_decode_anchor_alpha = self.raymap_codec.decode_anchor_alpha
        self.rothko_decode_block_grid = self.raymap_codec.decode_block_grid
        logger.info(
            "Rothko inference decoder: mode=%s anchor_alpha=%.3f block_grid=%d",
            self.rothko_decode_mode,
            self.rothko_decode_anchor_alpha,
            self.rothko_decode_block_grid,
        )
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.base_model_id: Optional[str] = None
        self.model_variant: Optional[str] = None
        self.vae_safetensors_path_requested: Optional[str] = None
        self.allow_vae_mismatch = bool(allow_vae_mismatch)
        self._vae_identity_cache: Optional[dict[str, Any]] = None
        self.dataset_stats_sha256: Optional[str] = None
        self.checkpoint_dataset_stats_sha256: Optional[str] = None
        self.lora_config: Optional[dict[str, Any]] = None
        self.lora_train_proprio_encoder = True
        self.to(self.device)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        model_variant: Optional[str] = None,
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
        rothko_decode_mode: str = "legacy",
        rothko_decode_anchor_alpha: float = 0.0,
        rothko_decode_block_grid: int = 4,
        latent_layout: str = LATENT_LAYOUT_RGB_THEN_RAYMAP,
        channel_io_init: str = CHANNEL_LAYOUT_IO_INIT_DUPLICATE_SQRT2,
        future_rgb_mode: str = FUTURE_RGB_MODE_JOINT,
        inference_predict_future_rgb: bool = False,
        allow_vae_mismatch: bool = False,
    ) -> "FastWAMVideoOnlyRaymap":
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required.")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required.")
        latent_layout = str(latent_layout)
        if latent_layout not in SUPPORTED_LATENT_LAYOUTS:
            raise ValueError(
                f"Unsupported latent_layout={latent_layout!r}; "
                f"expected one of {sorted(SUPPORTED_LATENT_LAYOUTS)}."
            )
        channel_io_init = str(channel_io_init)
        if channel_io_init not in SUPPORTED_CHANNEL_LAYOUT_IO_INITS:
            raise ValueError(
                f"Unsupported channel_io_init={channel_io_init!r}; expected one "
                f"of {sorted(SUPPORTED_CHANNEL_LAYOUT_IO_INITS)}."
            )
        load_dit_config = dict(video_dit_config)
        if (
            latent_layout == LATENT_LAYOUT_RGB_RAYMAP_CHANNEL
            and not skip_dit_load_from_pretrain
        ):
            target_in = int(load_dit_config["in_dim"])
            target_out = int(load_dit_config["out_dim"])
            if target_in % 2 or target_out % 2:
                raise ValueError(
                    "Channel-layout DiT in_dim/out_dim must be even so the original "
                    f"Wan I/O can be loaded first, got {target_in}/{target_out}."
                )
            load_dit_config["in_dim"] = target_in // 2
            load_dit_config["out_dim"] = target_out // 2

        components = load_wan_video_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            model_variant=model_variant,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            vae_safetensors_path=vae_safetensors_path,
            dit_config=load_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )
        if (
            latent_layout == LATENT_LAYOUT_RGB_RAYMAP_CHANNEL
            and not skip_dit_load_from_pretrain
        ):
            _expand_video_dit_io_for_rgb_raymap_channels(
                components.dit,
                target_channels=int(video_dit_config["in_dim"]),
                init_mode=channel_io_init,
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
            rothko_decode_mode=rothko_decode_mode,
            rothko_decode_anchor_alpha=rothko_decode_anchor_alpha,
            rothko_decode_block_grid=rothko_decode_block_grid,
            latent_layout=latent_layout,
            channel_io_init=channel_io_init,
            future_rgb_mode=future_rgb_mode,
            inference_predict_future_rgb=inference_predict_future_rgb,
            allow_vae_mismatch=allow_vae_mismatch,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
        }
        model.base_model_id = str(model_id)
        model.model_variant = components.model_variant
        model.vae_safetensors_path_requested = (
            None if vae_safetensors_path is None else str(vae_safetensors_path)
        )
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

    @torch.no_grad()
    def _ensure_rothko_decode_anchor_template(self, *, tiled: bool) -> None:
        if self.rothko_decode_anchor_alpha == 0.0:
            return
        if getattr(self.raymap_codec, "_decode_anchor_raw", None) is not None:
            return
        time = self.num_pixel_frames
        if self.raymap_representation == "libero_rothko":
            pose = torch.zeros(1, time, 7, device=self.device, dtype=torch.float32)
            pose[..., 3] = 1.0
            gripper = torch.full(
                (1, time, 1), 0.5, device=self.device, dtype=torch.float32
            )
        else:
            pose = torch.zeros(1, time, 14, device=self.device, dtype=torch.float32)
            pose[..., 3] = 1.0
            pose[..., 10] = 1.0
            gripper = torch.full(
                (1, time, 2), 0.5, device=self.device, dtype=torch.float32
            )
        ideal = self.raymap_codec.encode(pose, gripper).to(
            device=self.device, dtype=self.torch_dtype
        )
        reconstructed = self._decode_video_tensor(
            self._encode_video_latents(ideal, tiled=tiled), tiled=tiled
        )
        self.raymap_codec.set_decode_anchor_video(reconstructed)
        logger.info(
            "Cached a VAE-calibrated zero-motion Rothko anchor template "
            "for anchor_alpha=%.3f.",
            self.rothko_decode_anchor_alpha,
        )

    def _join_rgb_raymap_latents(
        self,
        rgb_latents: torch.Tensor,
        raymap_latents: torch.Tensor,
    ) -> torch.Tensor:
        if rgb_latents.shape != raymap_latents.shape:
            raise ValueError(
                "RGB/Rothko latent shapes must match before joining, got "
                f"{tuple(rgb_latents.shape)} and {tuple(raymap_latents.shape)}."
            )
        if self.latent_layout == LATENT_LAYOUT_RGB_THEN_RAYMAP:
            return torch.cat((rgb_latents, raymap_latents), dim=2)
        return torch.cat((rgb_latents, raymap_latents), dim=1)

    def _split_rgb_raymap_latents(
        self,
        joint_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.latent_layout == LATENT_LAYOUT_RGB_THEN_RAYMAP:
            expected_frames = 2 * self.num_latent_frames_per_modality
            if joint_latents.shape[2] != expected_frames:
                raise ValueError(
                    f"Expected {expected_frames} joint latent frames, got "
                    f"{joint_latents.shape[2]}."
                )
            return joint_latents.split(self.num_latent_frames_per_modality, dim=2)
        expected_channels = 2 * self.vae_latent_channels
        if joint_latents.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} joint latent channels, got "
                f"{joint_latents.shape[1]}."
            )
        return joint_latents.split(self.vae_latent_channels, dim=1)

    def _set_clean_conditions(
        self,
        joint_latents: torch.Tensor,
        rgb_condition: torch.Tensor,
        raymap_condition: torch.Tensor,
    ) -> None:
        if self.latent_layout == LATENT_LAYOUT_RGB_THEN_RAYMAP:
            joint_latents[:, :, 0:1] = rgb_condition
            ray_condition_index = self.num_latent_frames_per_modality
            joint_latents[
                :, :, ray_condition_index : ray_condition_index + 1
            ] = raymap_condition
            return
        joint_condition = torch.cat((rgb_condition, raymap_condition), dim=1)
        joint_latents[:, :, 0:1] = joint_condition

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
        cached_rgb_latents = sample.get("rgb_latents")
        cached_raymap_latents = sample.get("raymap_latents")
        if (cached_rgb_latents is None) != (cached_raymap_latents is None):
            raise ValueError(
                "Cached `rgb_latents` and `raymap_latents` must be provided together."
            )
        has_cached_latents = cached_rgb_latents is not None
        if not has_cached_latents:
            if not isinstance(video, torch.Tensor) or not isinstance(
                raymap, torch.Tensor
            ):
                raise TypeError(
                    "Video-only training without a latent cache requires tensor "
                    "`video` and `raymap`."
                )
            if video.ndim != 5 or raymap.ndim != 5:
                raise ValueError(
                    "Expected video/raymap [B,3,T,H,W], got "
                    f"{video.shape} and {raymap.shape}"
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
                raise ValueError(
                    "RGB/Rothko spatial dimensions must be multiples of 16."
                )
        elif (video is None) != (raymap is None):
            raise ValueError(
                "When raw pixels accompany cached latents, `video` and `raymap` "
                "must be provided together."
            )
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

        if not has_cached_latents:
            rgb = video.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
            raymap_device = raymap.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
            rgb_latents = self._encode_video_latents(rgb, tiled=tiled)
            raymap_latents = self._encode_video_latents(
                raymap_device, tiled=tiled
            )
        else:
            if not isinstance(cached_rgb_latents, torch.Tensor) or not isinstance(
                cached_raymap_latents, torch.Tensor
            ):
                raise TypeError("Cached RGB/Rothko latents must be tensors.")
            image_height = int(self.raymap_codec.config.image_height)
            image_width = int(self.raymap_codec.config.image_width)
            expected_latent_shape = (
                cached_rgb_latents.shape[0],
                self.vae_latent_channels,
                self.num_latent_frames_per_modality,
                image_height // int(self.vae.upsampling_factor),
                image_width // int(self.vae.upsampling_factor),
            )
            if (
                tuple(cached_rgb_latents.shape) != expected_latent_shape
                or tuple(cached_raymap_latents.shape) != expected_latent_shape
            ):
                raise ValueError(
                    "Cached RGB/Rothko latent shape mismatch: "
                    f"rgb={tuple(cached_rgb_latents.shape)}, "
                    f"raymap={tuple(cached_raymap_latents.shape)}, "
                    f"expected={expected_latent_shape}."
                )
            rgb_latents = cached_rgb_latents.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
            raymap_latents = cached_raymap_latents.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
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
        if context.shape[0] != rgb_latents.shape[0] or context_mask.shape[0] != rgb_latents.shape[0]:
            raise ValueError(
                "Cached context/latent batch mismatch: "
                f"context={context.shape[0]}, mask={context_mask.shape[0]}, "
                f"latents={rgb_latents.shape[0]}."
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
            "joint_latents": self._join_rgb_raymap_latents(
                rgb_latents, raymap_latents
            ),
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
        *,
        condition_latent_indices: Optional[Sequence[int]] = None,
        latent_frame_roles: Optional[Sequence[str]] = None,
        temporal_position_indices: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        if condition_latent_indices is None:
            condition_latent_indices = self.condition_latent_indices
        if (
            self.future_rgb_mode == FUTURE_RGB_MODE_TRAIN_ONLY_AUXILIARY
            and latent_frame_roles is None
        ):
            latent_frame_roles = self._training_latent_frame_roles()
            temporal_position_indices = tuple(
                range(2 * self.num_latent_frames_per_modality)
            )
        return self.video_expert(
            x=latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=True,
            condition_latent_indices=condition_latent_indices,
            latent_frame_roles=latent_frame_roles,
            temporal_position_indices=temporal_position_indices,
        )

    def _training_latent_frame_roles(self) -> tuple[str, ...]:
        future_count = self.num_latent_frames_per_modality - 1
        return (
            FRAME_ROLE_RGB_CONDITION,
            *(FRAME_ROLE_RGB_FUTURE for _ in range(future_count)),
            FRAME_ROLE_RAYMAP_CONDITION,
            *(FRAME_ROLE_RAYMAP_FUTURE for _ in range(future_count)),
        )

    def _ray_only_inference_frame_roles(self) -> tuple[str, ...]:
        future_count = self.num_latent_frames_per_modality - 1
        return (
            FRAME_ROLE_RGB_CONDITION,
            FRAME_ROLE_RAYMAP_CONDITION,
            *(FRAME_ROLE_RAYMAP_FUTURE for _ in range(future_count)),
        )

    def _ray_only_inference_temporal_positions(self) -> tuple[int, ...]:
        ray_start = self.num_latent_frames_per_modality
        return (0, *range(ray_start, 2 * ray_start))

    @property
    def predicts_future_rgb(self) -> bool:
        return (
            self.future_rgb_mode == FUTURE_RGB_MODE_JOINT
            or self.inference_predict_future_rgb
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
        prediction_rgb_all, prediction_raymap_all = self._split_rgb_raymap_latents(
            prediction
        )
        target_rgb_all, target_raymap_all = self._split_rgb_raymap_latents(target)
        prediction_rgb = prediction_rgb_all[:, :, 1:]
        target_rgb = target_rgb_all[:, :, 1:]
        prediction_raymap = prediction_raymap_all[:, :, 1:]
        target_raymap = target_raymap_all[:, :, 1:]
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
        decode_future_rgb: bool = True,
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

        self._ensure_rothko_decode_anchor_template(tiled=tiled)

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
        ray_only_inference = (
            self.future_rgb_mode == FUTURE_RGB_MODE_TRAIN_ONLY_AUXILIARY
            and not self.inference_predict_future_rgb
        )
        if ray_only_inference:
            joint_channels = self.vae_latent_channels
            joint_frames = 1 + self.num_latent_frames_per_modality
            inference_condition_indices = (0, 1)
            inference_frame_roles = self._ray_only_inference_frame_roles()
            inference_temporal_positions = (
                self._ray_only_inference_temporal_positions()
            )
        else:
            joint_channels = (
                self.vae_latent_channels
                if self.latent_layout == LATENT_LAYOUT_RGB_THEN_RAYMAP
                else 2 * self.vae_latent_channels
            )
            joint_frames = (
                2 * self.num_latent_frames_per_modality
                if self.latent_layout == LATENT_LAYOUT_RGB_THEN_RAYMAP
                else self.num_latent_frames_per_modality
            )
            inference_condition_indices = self.condition_latent_indices
            inference_frame_roles = None
            inference_temporal_positions = None
        latents = torch.randn(
            (1, joint_channels, joint_frames, latent_height, latent_width),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        if ray_only_inference:
            latents[:, :, 0:1] = rgb_condition
            latents[:, :, 1:2] = raymap_condition
        else:
            self._set_clean_conditions(latents, rgb_condition, raymap_condition)

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
                latents,
                timestep,
                context,
                context_mask,
                condition_latent_indices=inference_condition_indices,
                latent_frame_roles=inference_frame_roles,
                temporal_position_indices=inference_temporal_positions,
            )
            latents = self.infer_scheduler.step(prediction, delta, latents)
            if ray_only_inference:
                latents[:, :, 0:1] = rgb_condition
                latents[:, :, 1:2] = raymap_condition
            else:
                self._set_clean_conditions(
                    latents, rgb_condition, raymap_condition
                )

        if ray_only_inference:
            raymap_latents = latents[:, :, 1:]
            rgb_latents = None
        else:
            rgb_latents, raymap_latents = self._split_rgb_raymap_latents(latents)
        decoded_raymap = self._decode_video_tensor(raymap_latents, tiled=tiled)
        result: dict[str, Any] = {"raymap": decoded_raymap.cpu()}
        if rgb_latents is not None and decode_future_rgb:
            decoded_rgb = self._decode_video_tensor(rgb_latents, tiled=tiled)
            result.update(
                {
                    "video": self._video_tensor_to_pil(decoded_rgb[0]),
                    "video_tensor": decoded_rgb.cpu(),
                }
            )
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
        norm_stats = getattr(self.raymap_codec, "norm_stats", None)
        return {
            **self.raymap_codec.metadata(),
            "base_model_id": self.base_model_id,
            "model_variant": self.model_variant,
            "norm_stats_sha256": (
                None if norm_stats is None else norm_stats.fingerprint()
            ),
            "vae_identity": self._current_vae_identity(),
            "dataset_stats_sha256": self.dataset_stats_sha256,
            "raymap_representation": self.raymap_representation,
            "action_horizon": self.action_horizon,
            "latent_layout": self.latent_layout,
            "pretrained_io_expansion": self.pretrained_io_expansion,
            "future_rgb_mode": self.future_rgb_mode,
            "inference_temporal_position_mode": (
                self.inference_temporal_position_mode
            ),
            "temporal_rope_mode": self.temporal_rope_mode,
            "condition_latent_indices": list(self.condition_latent_indices),
            "video_attention_mask_mode": video_attention_mask_mode,
            "loss_weights": {
                "rgb": self.loss_lambda_rgb,
                "raymap": self.loss_lambda_raymap,
            },
        }

    @staticmethod
    def _dataset_stats_fingerprint(path: str | Path) -> str:
        stats_path = Path(path).expanduser().resolve()
        with stats_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def set_training_dataset_stats(self, path: str | Path) -> None:
        self.dataset_stats_sha256 = self._dataset_stats_fingerprint(path)

    def validate_dataset_stats(self, path: str | Path) -> None:
        current = self._dataset_stats_fingerprint(path)
        expected = self.checkpoint_dataset_stats_sha256
        if expected is None:
            logger.warning(
                "Checkpoint has no recoverable dataset-stats identity; %s cannot "
                "be authenticated.",
                path,
            )
            return
        if current != expected:
            raise ValueError(
                "Checkpoint dataset normalization stats mismatch: "
                f"checkpoint_sha256={expected}, current_sha256={current}, "
                f"path={path}."
            )

    def _current_vae_identity(self) -> Optional[dict[str, Any]]:
        if self._vae_identity_cache is not None:
            return dict(self._vae_identity_cache)
        model_paths = getattr(self, "model_paths", None)
        path_value = model_paths.get("vae") if isinstance(model_paths, dict) else None
        if path_value in (None, "", "null"):
            return None
        path = Path(str(path_value)).expanduser().resolve()
        original_kind = (
            "original_wan21"
            if self.model_variant == "wan2.1-t2v-1.3b"
            else "original_wan22"
        )
        identity: dict[str, Any] = {
            "kind": (
                "custom"
                if self.vae_safetensors_path_requested not in (None, "", "null")
                else original_kind
            ),
            "filename": path.name,
        }
        if path.is_file():
            identity.update(
                {
                    "size_bytes": path.stat().st_size,
                    "sha256": self._sha256_file(path),
                }
            )
        else:
            logger.warning(
                "VAE source %s is not a file; checkpoint VAE identity is limited.",
                path,
            )
        self._vae_identity_cache = identity
        return dict(identity)

    def validate_latent_cache_metadata(self, metadata: dict[str, Any]) -> None:
        """Reject a cache produced by a different VAE/model precision contract."""
        cached_variant = metadata.get("model_variant")
        if cached_variant != self.model_variant:
            raise ValueError(
                "Latent cache Wan variant mismatch: "
                f"cache={cached_variant!r}, model={self.model_variant!r}."
            )
        cached_dtype = metadata.get("encoding_torch_dtype")
        expected_dtype = str(self.torch_dtype)
        if cached_dtype != expected_dtype:
            raise ValueError(
                "Latent cache encoding dtype mismatch: "
                f"cache={cached_dtype!r}, model={expected_dtype!r}."
            )
        cached_vae_identity = metadata.get("vae_identity")
        current_vae_identity = self._current_vae_identity()
        if cached_vae_identity != current_vae_identity:
            raise ValueError(
                "Latent cache VAE identity mismatch: "
                f"cache={cached_vae_identity}, current={current_vae_identity}."
            )
        cached_shape = tuple(int(value) for value in metadata.get("latent_shape", ()))
        expected_shape = (
            self.vae_latent_channels,
            self.num_latent_frames_per_modality,
            int(self.raymap_codec.config.image_height)
            // int(self.vae.upsampling_factor),
            int(self.raymap_codec.config.image_width)
            // int(self.vae.upsampling_factor),
        )
        if cached_shape != expected_shape:
            raise ValueError(
                "Latent cache shape/model mismatch: "
                f"cache={cached_shape}, expected={expected_shape}."
            )

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _validate_vae_identity(
        self,
        checkpoint_identity: Optional[dict[str, Any]],
        *,
        run_config: Optional[DictConfig],
        checkpoint_path: str,
    ) -> None:
        current = self._current_vae_identity()
        mismatch_reason: Optional[str] = None
        if isinstance(checkpoint_identity, dict):
            if current is None:
                mismatch_reason = "checkpoint identifies a VAE but the model does not"
            elif checkpoint_identity.get("sha256") and current.get("sha256"):
                if checkpoint_identity["sha256"] != current["sha256"]:
                    mismatch_reason = (
                        f"checkpoint_sha256={checkpoint_identity['sha256']} "
                        f"model_sha256={current['sha256']}"
                    )
            elif checkpoint_identity.get("kind") != current.get("kind"):
                mismatch_reason = (
                    f"checkpoint_kind={checkpoint_identity.get('kind')} "
                    f"model_kind={current.get('kind')}"
                )
        elif run_config is not None:
            trained_custom = OmegaConf.select(
                run_config, "model.vae_safetensors_path"
            )
            trained_model_variant = OmegaConf.select(
                run_config, "model.model_variant"
            )
            if trained_model_variant in (None, "", "null"):
                trained_model_id = OmegaConf.select(run_config, "model.model_id")
                trained_model_variant = (
                    "wan2.1-t2v-1.3b"
                    if str(trained_model_id).rstrip("/").lower()
                    == "wan-ai/wan2.1-t2v-1.3b"
                    else "wan2.2-ti2v-5b"
                )
            trained_original_kind = (
                "original_wan21"
                if str(trained_model_variant) == "wan2.1-t2v-1.3b"
                else "original_wan22"
            )
            trained_kind = (
                "custom"
                if trained_custom not in (None, "", "null")
                else trained_original_kind
            )
            current_kind = None if current is None else current.get("kind")
            if trained_kind != current_kind:
                mismatch_reason = (
                    f"legacy run config used {trained_kind}, model uses {current_kind}"
                )
            elif trained_kind == "custom" and current is not None:
                trained_path = Path(str(trained_custom)).expanduser()
                if not trained_path.is_absolute():
                    trained_path = Path.cwd() / trained_path
                if trained_path.is_file() and current.get("sha256"):
                    trained_sha256 = self._sha256_file(trained_path.resolve())
                    if trained_sha256 != current["sha256"]:
                        mismatch_reason = (
                            f"legacy_run_vae_sha256={trained_sha256} "
                            f"model_sha256={current['sha256']}"
                        )
                else:
                    logger.warning(
                        "Checkpoint %s predates embedded VAE identity and the "
                        "training custom VAE path %s cannot be hashed.",
                        checkpoint_path,
                        trained_path,
                    )
            else:
                logger.warning(
                    "Checkpoint %s predates embedded VAE identity; recovered VAE "
                    "kind=%s from run config, but exact bytes are unauthenticated.",
                    checkpoint_path,
                    trained_kind,
                )
        else:
            logger.warning(
                "Checkpoint %s has no VAE identity and no run config; VAE bytes "
                "cannot be authenticated.",
                checkpoint_path,
            )
        if mismatch_reason is None:
            return
        message = (
            "Checkpoint VAE mismatch: "
            f"{mismatch_reason}. The frozen VAE is not stored in the DiT checkpoint."
        )
        if self.allow_vae_mismatch:
            logger.warning("%s Proceeding because allow_vae_mismatch=true.", message)
            return
        raise ValueError(
            message
            + " Use the training VAE, or explicitly set "
            "model.allow_vae_mismatch=true for an intentional VAE ablation."
        )

    @staticmethod
    def _find_run_config(
        checkpoint_path: str,
    ) -> tuple[Optional[DictConfig], Optional[Path]]:
        checkpoint = Path(checkpoint_path).resolve()
        for parent in list(checkpoint.parents)[:5]:
            candidate = parent / "config.yaml"
            if candidate.is_file():
                return OmegaConf.load(candidate), candidate
        return None, None

    @staticmethod
    def _metadata_values_match(actual: Any, expected: Any) -> bool:
        if isinstance(expected, float):
            try:
                return abs(float(actual) - expected) <= 1e-8
            except (TypeError, ValueError):
                return False
        if isinstance(expected, tuple):
            expected = list(expected)
        if isinstance(actual, tuple):
            actual = list(actual)
        return actual == expected

    def _recover_legacy_norm_stats_fingerprint(
        self,
        *,
        run_config: Optional[DictConfig],
        run_config_path: Optional[Path],
    ) -> Optional[str]:
        if run_config is None:
            return None
        configured_path = OmegaConf.select(run_config, "model.rothko_norm_stats")
        if configured_path in (None, "", "null"):
            configured_path = OmegaConf.select(
                run_config, "data.train.rothko_norm_stats"
            )
        if configured_path in (None, "", "null"):
            return None
        raw_path = Path(str(configured_path)).expanduser()
        candidates = [raw_path]
        if not raw_path.is_absolute():
            candidates = [Path.cwd() / raw_path]
            if run_config_path is not None:
                candidates.append(run_config_path.parent / raw_path)
        stats_path = next((path for path in candidates if path.is_file()), None)
        if stats_path is None:
            logger.warning(
                "Could not resolve legacy checkpoint Rothko stats path %r from %s.",
                configured_path,
                run_config_path,
            )
            return None
        from fastwam.representations.rothko import RothkoNormStats

        return RothkoNormStats.load(stats_path).fingerprint()

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

        run_config, run_config_path = self._find_run_config(checkpoint_path)
        checkpoint_attention_mode = visual_config.get(
            "video_attention_mask_mode"
        )
        if checkpoint_attention_mode is None:
            configured_mode = (
                None
                if run_config is None
                else OmegaConf.select(
                    run_config,
                    "model.video_dit_config.video_attention_mask_mode",
                )
            )
            run_config_mode = (
                None if configured_mode is None else str(configured_mode).strip()
            )

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

        checkpoint_base_model_id = visual_config.get("base_model_id")
        if (
            checkpoint_base_model_id is not None
            and self.base_model_id is not None
            and str(checkpoint_base_model_id) != str(self.base_model_id)
        ):
            raise ValueError(
                "Checkpoint base model mismatch: "
                f"checkpoint={checkpoint_base_model_id!r}, "
                f"model={self.base_model_id!r}."
            )
        checkpoint_model_variant = visual_config.get("model_variant")
        if (
            checkpoint_model_variant is not None
            and self.model_variant is not None
            and str(checkpoint_model_variant) != str(self.model_variant)
        ):
            raise ValueError(
                "Checkpoint Wan variant mismatch: "
                f"checkpoint={checkpoint_model_variant!r}, "
                f"model={self.model_variant!r}."
            )

        expected_metadata = {
            "action_horizon": self.action_horizon,
            "latent_layout": self.latent_layout,
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
        checkpoint_io_expansion = visual_config.get("pretrained_io_expansion")
        if checkpoint_io_expansion != self.pretrained_io_expansion:
            raise ValueError(
                "Checkpoint pretrained I/O expansion mismatch: "
                f"checkpoint={checkpoint_io_expansion!r}, "
                f"model={self.pretrained_io_expansion!r}."
            )
        checkpoint_future_rgb_mode = visual_config.get(
            "future_rgb_mode", FUTURE_RGB_MODE_JOINT
        )
        if checkpoint_future_rgb_mode != self.future_rgb_mode:
            raise ValueError(
                "Checkpoint future RGB mode mismatch: "
                f"checkpoint={checkpoint_future_rgb_mode!r}, "
                f"model={self.future_rgb_mode!r}."
            )
        checkpoint_inference_positions = visual_config.get(
            "inference_temporal_position_mode",
            INFERENCE_TEMPORAL_POSITION_FULL_JOINT,
        )
        if checkpoint_inference_positions != self.inference_temporal_position_mode:
            raise ValueError(
                "Checkpoint inference temporal-position mode mismatch: "
                f"checkpoint={checkpoint_inference_positions!r}, "
                f"model={self.inference_temporal_position_mode!r}."
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

        expected_codec_metadata = self.raymap_codec.metadata()
        # Old checkpoints unambiguously use relative frame zero. Never allow
        # the new absolute conditioning to be silently enabled on old weights
        # (or an absolute checkpoint to be deployed with the legacy codec).
        if visual_config.get("frame0_pose_mode", "relative") != expected_codec_metadata.get("frame0_pose_mode", "relative"):
            raise ValueError("Checkpoint Rothko frame0_pose_mode mismatch")
        missing_codec_keys: list[str] = []
        for key, expected in expected_codec_metadata.items():
            if key == "raymap_representation":
                continue
            if key not in visual_config:
                if key in {"absolute_position_min", "absolute_position_max"}:
                    raise ValueError(f"Absolute RAY0 checkpoint missing {key}")
                missing_codec_keys.append(key)
                continue
            actual = visual_config[key]
            if not self._metadata_values_match(actual, expected):
                raise ValueError(
                    f"Checkpoint Rothko codec mismatch for {key}: "
                    f"checkpoint={actual!r}, model={expected!r}. Use the exact "
                    "training Rothko configuration for evaluation/resume."
                )
        if missing_codec_keys:
            logger.warning(
                "Checkpoint %s predates complete Rothko codec metadata; missing "
                "keys=%s. Geometry could not be fully authenticated.",
                checkpoint_path,
                missing_codec_keys,
            )

        current_stats = getattr(self.raymap_codec, "norm_stats", None)
        expected_stats_fingerprint = (
            None if current_stats is None else current_stats.fingerprint()
        )
        checkpoint_stats_fingerprint = visual_config.get("norm_stats_sha256")
        if checkpoint_stats_fingerprint is None:
            checkpoint_stats_fingerprint = self._recover_legacy_norm_stats_fingerprint(
                run_config=run_config,
                run_config_path=run_config_path,
            )
            if checkpoint_stats_fingerprint is not None:
                logger.warning(
                    "Checkpoint %s predates embedded norm-stats identity; recovered "
                    "it from %s.",
                    checkpoint_path,
                    run_config_path,
                )
            elif expected_stats_fingerprint is not None:
                logger.warning(
                    "Checkpoint %s has no recoverable norm-stats identity. The "
                    "current stats cannot be authenticated against this legacy "
                    "checkpoint.",
                    checkpoint_path,
                )
        if (
            checkpoint_stats_fingerprint is not None
            and expected_stats_fingerprint != checkpoint_stats_fingerprint
        ):
            raise ValueError(
                "Checkpoint Rothko normalization stats mismatch: "
                f"checkpoint_sha256={checkpoint_stats_fingerprint}, "
                f"model_sha256={expected_stats_fingerprint}. Use the exact stats "
                "used during training."
            )

        self._validate_vae_identity(
            visual_config.get("vae_identity"),
            run_config=run_config,
            checkpoint_path=checkpoint_path,
        )

        checkpoint_dataset_stats = visual_config.get("dataset_stats_sha256")
        if checkpoint_dataset_stats is None:
            checkpoint = Path(checkpoint_path).resolve()
            for parent in list(checkpoint.parents)[:5]:
                candidate = parent / "dataset_stats.json"
                if candidate.is_file():
                    checkpoint_dataset_stats = self._dataset_stats_fingerprint(
                        candidate
                    )
                    logger.warning(
                        "Checkpoint %s predates embedded dataset-stats identity; "
                        "recovered it from %s.",
                        checkpoint_path,
                        candidate,
                    )
                    break
        self.checkpoint_dataset_stats_sha256 = checkpoint_dataset_stats

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
                    "Start LoRA from the configured original Wan base, or "
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
