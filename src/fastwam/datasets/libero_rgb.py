"""Shared RGB canvas construction for LIBERO training and deployment."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _to_chw_float(image: Any) -> torch.Tensor:
    tensor = image if isinstance(image, torch.Tensor) else torch.as_tensor(np.asarray(image))
    if tensor.ndim < 3:
        raise ValueError(f"LIBERO image must have at least 3 dims, got {tensor.shape}")
    if tensor.shape[-1] == 3 and tensor.shape[-3] != 3:
        tensor = tensor.movedim(-1, -3)
    if tensor.shape[-3] != 3:
        raise ValueError(
            "LIBERO image channel dimension must be last or third-from-last, "
            f"got {tensor.shape}"
        )
    original_dtype = tensor.dtype
    tensor = tensor.float()
    if original_dtype == torch.uint8 or (
        tensor.numel() and float(tensor.detach().amax()) > 1.5
    ):
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0)


def _resize(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    leading = image.shape[:-3]
    flattened = image.reshape(-1, 3, image.shape[-2], image.shape[-1])
    resized = F.interpolate(
        flattened,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return resized.reshape(*leading, 3, height, width)


def build_libero_rgb_canvas(
    agentview: Any,
    wrist: Any,
    *,
    camera_height: int = 224,
    camera_width: int = 224,
    normalize: bool = True,
) -> torch.Tensor:
    """Build ``[agentview | wrist]`` with identical train/eval interpolation.

    Inputs may be HWC NumPy arrays or tensors with arbitrary matching leading
    dimensions and either HWC/CHW image layout.  Output is a tensor with shape
    ``[...,3,camera_height,2*camera_width]``.
    """
    agentview_tensor = _to_chw_float(agentview)
    wrist_tensor = _to_chw_float(wrist)
    if agentview_tensor.shape[:-3] != wrist_tensor.shape[:-3]:
        raise ValueError(
            "LIBERO camera leading shapes differ: "
            f"{agentview_tensor.shape} vs {wrist_tensor.shape}"
        )
    agentview_tensor = _resize(agentview_tensor, camera_height, camera_width)
    wrist_tensor = _resize(wrist_tensor, camera_height, camera_width)
    canvas = torch.cat((agentview_tensor, wrist_tensor), dim=-1)
    return canvas.mul(2.0).sub(1.0) if normalize else canvas
