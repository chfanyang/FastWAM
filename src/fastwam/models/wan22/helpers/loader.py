from dataclasses import dataclass
import inspect
import os
from pathlib import Path
from typing import Any

import torch
import time

from .io import ModelConfig, hash_model_file, load_state_dict
from .state_dict_converters import (
    wan_video_vae_state_dict_converter,
)
from ..wan_video_dit import WanVideoDiT
from ..wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder
from ..wan_video_vae import WanVideoVAE, WanVideoVAE38
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)
SKIPPED_PRETRAIN_SENTINEL = "SKIPPED_PRETRAIN"


@dataclass(frozen=True)
class WanModelSpec:
    name: str
    model_id: str
    vae_filename: str
    vae_class: type[WanVideoVAE]


WAN_MODEL_SPECS = {
    "wan2.1-t2v-1.3b": WanModelSpec(
        name="wan2.1-t2v-1.3b",
        model_id="Wan-AI/Wan2.1-T2V-1.3B",
        vae_filename="Wan2.1_VAE.pth",
        vae_class=WanVideoVAE,
    ),
    "wan2.2-ti2v-5b": WanModelSpec(
        name="wan2.2-ti2v-5b",
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        vae_filename="Wan2.2_VAE.pth",
        vae_class=WanVideoVAE38,
    ),
}


def resolve_wan_model_spec(
    model_id: str,
    model_variant: str | None = None,
) -> WanModelSpec:
    if model_variant is not None:
        key = str(model_variant).strip().lower()
        if key not in WAN_MODEL_SPECS:
            raise ValueError(
                f"Unsupported Wan model_variant={model_variant!r}. "
                f"Supported variants: {sorted(WAN_MODEL_SPECS)}"
            )
        spec = WAN_MODEL_SPECS[key]
        if str(model_id).rstrip("/").lower() != spec.model_id.lower():
            raise ValueError(
                "Wan model_id/model_variant mismatch: "
                f"model_id={model_id!r}, model_variant={model_variant!r}, "
                f"expected_model_id={spec.model_id!r}."
            )
        return spec

    normalized = str(model_id).rstrip("/").lower()
    matches = [
        spec for spec in WAN_MODEL_SPECS.values()
        if normalized == spec.model_id.lower()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Cannot infer a supported Wan variant from model_id={model_id!r}. "
            "Set `model_variant` explicitly. Supported official model IDs: "
            f"{[spec.model_id for spec in WAN_MODEL_SPECS.values()]}"
        )
    return matches[0]


@dataclass
class WanLoadedComponents:
    dit: WanVideoDiT
    vae: WanVideoVAE
    text_encoder: WanTextEncoder | None
    tokenizer: HuggingfaceTokenizer | None
    dit_path: str
    vae_path: str
    text_encoder_path: str | None
    tokenizer_path: str | None
    model_variant: str


# Backward-compatible public name used by the original Wan2.2-only code.
Wan22LoadedComponents = WanLoadedComponents


WAN22_MODEL_REGISTRY = [
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth")
        "model_hash": "9c8818c2cbea55eca56c7b447df170da",
        "model_name": "wan_video_text_encoder",
        "model_class": WanTextEncoder,
    },
    {
        # ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "9269f8db9040a9d860eaca435be61814",
        "model_name": "wan_video_dit",
        "model_class": WanVideoDiT,
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "1f5ab7703c6fc803fdded85ff040c316",
        "model_name": "wan_video_dit",
        "model_class": WanVideoDiT,
    },
    {
        # ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth")
        "model_hash": "ccc42284ea13e1ad04693284c7a09be6",
        "model_name": "wan_video_vae",
        "model_class": WanVideoVAE,
        "state_dict_converter": wan_video_vae_state_dict_converter,
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth")
        "model_hash": "e1de6c02cdac79f8b739f4d3698cd216",
        "model_name": "wan_video_vae",
        "model_class": WanVideoVAE38,
        "state_dict_converter": wan_video_vae_state_dict_converter,
    },
]


def _validate_dit_config(dit_config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(dit_config, dict):
        raise ValueError(f"`dit_config` must be a dict, got {type(dit_config)}")

    validated = dict(dit_config)

    signature = inspect.signature(WanVideoDiT.__init__)
    allowed_keys = set()
    required_keys = set()
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        allowed_keys.add(name)
        if param.default is inspect.Signature.empty:
            required_keys.add(name)

    unknown_keys = sorted(set(validated) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Unknown keys in `dit_config`: {unknown_keys}. "
            f"Allowed keys: {sorted(allowed_keys)}"
        )

    missing_keys = sorted(required_keys - set(validated))
    if missing_keys:
        raise ValueError(
            f"Missing required keys in `dit_config`: {missing_keys}. "
            "Please specify all required WanVideoDiT constructor args."
        )

    return validated


def _load_registered_model(
    path,
    model_name: str,
    torch_dtype: torch.dtype,
    device: str,
    model_kwargs_override: dict[str, Any] | None = None,
):
    model_hash = hash_model_file(path)

    matched_config = None
    for config in WAN22_MODEL_REGISTRY:
        if config["model_hash"] == model_hash and config["model_name"] == model_name:
            matched_config = config
            break
    if matched_config is None:
        raise ValueError(
            f"Cannot detect model type for {model_name}. File: {path}. "
            f"Model hash: {model_hash}. This standalone package follows DiffSynth hash-based loading."
        )

    model_class = matched_config["model_class"]
    model_kwargs = dict(matched_config.get("extra_kwargs", {}))
    if model_kwargs_override is not None:
        model_kwargs.update(model_kwargs_override)
    state_dict_converter = matched_config.get("state_dict_converter")

    model = model_class(**model_kwargs)
    state_dict = load_state_dict(path, torch_dtype=torch_dtype, device="cpu")
    if state_dict_converter is not None:
        state_dict = state_dict_converter(state_dict)

    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            f"Registered {model_name} weights do not exactly match "
            f"{model_class.__name__}: missing={incompatible.missing_keys[:8]}, "
            f"unexpected={incompatible.unexpected_keys[:8]}, file={path}."
        )
    model = model.to(device=device, dtype=torch_dtype)
    return model


def _resolve_optional_safetensors_path(
    path: str | os.PathLike[str] | None,
) -> Path | None:
    if path is None:
        return None
    path_text = str(path).strip()
    if path_text.lower() in {"", "none", "null"}:
        return None
    resolved = Path(os.path.expandvars(path_text)).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Custom VAE safetensors not found: {resolved}")
    if resolved.suffix.lower() != ".safetensors":
        raise ValueError(
            "Custom VAE override must be a `.safetensors` file, got: "
            f"{resolved}"
        )
    return resolved


def _load_custom_wan_vae(
    path: Path,
    *,
    vae_class: type[WanVideoVAE],
    torch_dtype: torch.dtype,
    device: str,
) -> WanVideoVAE:
    vae = vae_class()
    state = load_state_dict(
        str(path),
        torch_dtype=torch_dtype,
        device="cpu",
    )
    expected_keys = set(vae.state_dict())
    provided_keys = set(state)
    if provided_keys == expected_keys:
        mapped_state = state
    elif {f"model.{key}" for key in provided_keys} == expected_keys:
        mapped_state = {f"model.{key}": value for key, value in state.items()}
    else:
        missing = sorted(expected_keys - provided_keys)
        unexpected = sorted(provided_keys - expected_keys)
        prefixed_keys = {f"model.{key}" for key in provided_keys}
        if len(expected_keys - prefixed_keys) < len(missing):
            missing = sorted(expected_keys - prefixed_keys)
            unexpected = sorted(prefixed_keys - expected_keys)
        raise ValueError(
            f"Custom VAE safetensors must contain a complete {vae_class.__name__} "
            "state dict. "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}, path={path}"
        )
    vae.load_state_dict(mapped_state, strict=True)
    vae = vae.eval().requires_grad_(False).to(
        device=device,
        dtype=torch_dtype,
    )
    logger.info("Loaded complete custom Wan VAE from %s", path)
    return vae


def _resolve_configs(
    model_id: str,
    tokenizer_model_id: str,
    redirect_common_files: bool = True,
    model_variant: str | None = None,
):
    spec = resolve_wan_model_spec(model_id, model_variant)
    dit_config = ModelConfig(model_id=model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors")
    text_config = ModelConfig(model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth")
    vae_config = ModelConfig(model_id=model_id, origin_file_pattern=spec.vae_filename)
    tokenizer_config = ModelConfig(model_id=tokenizer_model_id, origin_file_pattern="google/umt5-xxl/")

    if redirect_common_files:
        text_config.model_id = "DiffSynth-Studio/Wan-Series-Converted-Safetensors"
        text_config.origin_file_pattern = "models_t5_umt5-xxl-enc-bf16.safetensors"
        if spec.name == "wan2.2-ti2v-5b":
            vae_config.model_id = "DiffSynth-Studio/Wan-Series-Converted-Safetensors"
            vae_config.origin_file_pattern = "Wan2.2_VAE.safetensors"
    return dit_config, text_config, vae_config, tokenizer_config


def load_wan_video_components(
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
    model_variant: str | None = None,
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
    tokenizer_max_len: int = 512,
    redirect_common_files: bool = True,
    dit_config: dict[str, Any] | None = None,
    skip_dit_load_from_pretrain: bool = False,
    load_text_encoder: bool = True,
    vae_safetensors_path: str | os.PathLike[str] | None = None,
) -> WanLoadedComponents:
    spec = resolve_wan_model_spec(model_id, model_variant)
    logger.info("Loading %s components...", spec.name)
    start = time.time()

    if dit_config is None:
        raise ValueError(f"`dit_config` is required for {spec.name} loading.")
    validated_dit_config = _validate_dit_config(dit_config)

    dit_model_config, text_config, vae_config, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
        model_variant=spec.name,
    )

    custom_vae_path = _resolve_optional_safetensors_path(vae_safetensors_path)
    if custom_vae_path is None:
        vae_config.download_if_necessary()
    else:
        vae_config.path = str(custom_vae_path)
        logger.info("Using custom Wan VAE safetensors: %s", custom_vae_path)
    if load_text_encoder:
        text_config.download_if_necessary()
        tokenizer_config.download_if_necessary()

    if skip_dit_load_from_pretrain:
        logger.info(
            "Skipping pretrained video DiT load (`skip_dit_load_from_pretrain=True`); "
            "initializing video expert randomly and expecting checkpoint override."
        )
        dit: WanVideoDiT = WanVideoDiT(**validated_dit_config).to(device=device, dtype=torch_dtype)
        dit_path = SKIPPED_PRETRAIN_SENTINEL
    else:
        dit_model_config.download_if_necessary()
        dit = _load_registered_model(
            dit_model_config.path,
            "wan_video_dit",
            torch_dtype=torch_dtype,
            device=device,
            model_kwargs_override=validated_dit_config,
        )
        dit_path = str(dit_model_config.path)

    text_encoder: WanTextEncoder | None = None
    tokenizer: HuggingfaceTokenizer | None = None
    text_encoder_path: str | None = None
    tokenizer_path: str | None = None
    if load_text_encoder:
        text_encoder = _load_registered_model(
            text_config.path,
            "wan_video_text_encoder",
            torch_dtype=torch_dtype,
            device=device,
        )
        tokenizer = HuggingfaceTokenizer(
            name=tokenizer_config.path,
            seq_len=int(tokenizer_max_len),
            clean="whitespace",
        )
        text_encoder_path = str(text_config.path)
        tokenizer_path = str(tokenizer_config.path)
    else:
        logger.info(
            "Skipping pretrained text encoder/tokenizer load (`load_text_encoder=False`); "
            "training must provide cached `context/context_mask`."
        )

    if custom_vae_path is None:
        vae: WanVideoVAE = _load_registered_model(
            vae_config.path,
            "wan_video_vae",
            torch_dtype=torch_dtype,
            device=device,
        )
        if not isinstance(vae, spec.vae_class):
            raise TypeError(
                f"Loaded VAE type mismatch for {spec.name}: "
                f"expected {spec.vae_class.__name__}, got {type(vae).__name__}."
            )
    else:
        vae = _load_custom_wan_vae(
            custom_vae_path,
            vae_class=spec.vae_class,
            torch_dtype=torch_dtype,
            device=device,
        )
    logger.info(
        "Finished loading %s components in %.2f seconds.",
        spec.name,
        time.time() - start,
    )
    return WanLoadedComponents(
        dit=dit,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        dit_path=dit_path,
        vae_path=str(vae_config.path),
        text_encoder_path=text_encoder_path,
        tokenizer_path=tokenizer_path,
        model_variant=spec.name,
    )


def load_wan22_ti2v_5b_components(
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
    tokenizer_max_len: int = 512,
    redirect_common_files: bool = True,
    dit_config: dict[str, Any] | None = None,
    skip_dit_load_from_pretrain: bool = False,
    load_text_encoder: bool = True,
    vae_safetensors_path: str | os.PathLike[str] | None = None,
):
    return load_wan_video_components(
        device=device,
        torch_dtype=torch_dtype,
        model_id=model_id,
        model_variant="wan2.2-ti2v-5b",
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=tokenizer_max_len,
        redirect_common_files=redirect_common_files,
        dit_config=dit_config,
        skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
        load_text_encoder=load_text_encoder,
        vae_safetensors_path=vae_safetensors_path,
    )
