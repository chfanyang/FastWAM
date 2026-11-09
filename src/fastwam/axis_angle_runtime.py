"""Opt-in factory; shared FastWAM/Raymap runtime is unchanged."""
import torch
from omegaconf import DictConfig, OmegaConf

def create_axis_angle_raymap(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    model_variant: str | None = None,
    tokenizer_max_len: int = 128,
    load_text_encoder: bool = False,
    proprio_dim: int | None = None,
    redirect_common_files: bool = True,
    vae_safetensors_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    scheduler=None,
    loss=None,
    action_horizon: int = 16,
    raymap_representation: str = "rothko",
    rothko_norm_stats: str | None = None,
    rothko_config=None,
    rothko_decode_mode: str = "legacy",
    rothko_decode_anchor_alpha: float = 0.0,
    rothko_decode_block_grid: int = 4,
    latent_layout: str = "rgb_then_raymap",
    channel_io_init: str = "duplicate_sqrt2_input_duplicate_output",
    future_rgb_mode: str = "joint",
    inference_predict_future_rgb: bool = False,
    allow_vae_mismatch: bool = False,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.fastwam_axis_angle import FastWAMAxisAngle

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(
            f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}"
        )
    if isinstance(scheduler, DictConfig):
        scheduler = OmegaConf.to_container(scheduler, resolve=True)
    scheduler = {} if scheduler is None else scheduler
    if not isinstance(scheduler, dict):
        raise ValueError(f"`scheduler` must be dict-like, got {type(scheduler)}")
    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    loss = {} if loss is None else loss
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")
    if isinstance(rothko_config, DictConfig):
        rothko_config = OmegaConf.to_container(rothko_config, resolve=True)
    rothko_config = {} if rothko_config is None else rothko_config
    if not isinstance(rothko_config, dict):
        raise ValueError(
            f"`rothko_config` must be dict-like, got {type(rothko_config)}"
        )

    return FastWAMAxisAngle.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        model_variant=model_variant,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        redirect_common_files=bool(redirect_common_files),
        vae_safetensors_path=vae_safetensors_path,
        video_dit_config=video_dit_config,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        train_shift=float(scheduler.get("train_shift", 5.0)),
        infer_shift=float(scheduler.get("infer_shift", 5.0)),
        num_train_timesteps=int(scheduler.get("num_train_timesteps", 1000)),
        loss_lambda_rgb=float(loss.get("lambda_rgb", 1.0)),
        loss_lambda_raymap=float(loss.get("lambda_raymap", 1.0)),
        action_horizon=int(action_horizon),
        raymap_representation=str(raymap_representation),
        rothko_norm_stats=rothko_norm_stats,
        rothko_config=rothko_config,
        rothko_decode_mode=str(rothko_decode_mode),
        rothko_decode_anchor_alpha=float(rothko_decode_anchor_alpha),
        rothko_decode_block_grid=int(rothko_decode_block_grid),
        latent_layout=str(latent_layout),
        channel_io_init=str(channel_io_init),
        future_rgb_mode=str(future_rgb_mode),
        inference_predict_future_rgb=bool(inference_predict_future_rgb),
        allow_vae_mismatch=bool(allow_vae_mismatch),
    )
