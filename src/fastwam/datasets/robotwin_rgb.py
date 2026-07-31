"""Shared RoboTwin RGB preprocessing for training and deployment."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F


RGBInput: TypeAlias = np.ndarray | torch.Tensor

_PROCESSOR_SIZE = [240, 320]
_HEAD_SIZE = [256, 320]
_WRIST_SIZE = [128, 160]
_INTERPOLATION = transforms_F.InterpolationMode.BILINEAR


def _as_float_chw(image: RGBInput, *, name: str) -> torch.Tensor:
    if isinstance(image, np.ndarray):
        tensor = torch.from_numpy(np.array(image, copy=True, order="C"))
    elif isinstance(image, torch.Tensor):
        tensor = image
    else:
        raise TypeError(
            f"`{name}` must be a numpy array or torch tensor, got {type(image)}."
        )

    if tensor.ndim < 3:
        raise ValueError(
            f"`{name}` must have at least 3 dimensions, got {tuple(tensor.shape)}."
        )
    if tensor.shape[-3] == 3:
        pass
    elif tensor.shape[-1] == 3:
        tensor = tensor.movedim(-1, -3)
    else:
        raise ValueError(
            f"`{name}` must use CHW or HWC RGB layout, got {tuple(tensor.shape)}."
        )

    if tensor.dtype == torch.uint8:
        tensor = tensor.to(dtype=torch.float32).div_(255.0)
    elif tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float32)
    else:
        raise TypeError(
            f"`{name}` must have uint8 or floating-point dtype, got {tensor.dtype}."
        )
    return tensor


def _resize(image: torch.Tensor, size: list[int]) -> torch.Tensor:
    return transforms_F.resize(
        image,
        size=size,
        interpolation=_INTERPOLATION,
        antialias=True,
    )


def build_robotwin_rgb_canvas(
    head: RGBInput,
    left_wrist: RGBInput,
    right_wrist: RGBInput,
) -> torch.Tensor:
    """Build the normalized 384x320 RoboTwin three-camera canvas.

    Inputs may be uint8 numpy arrays/tensors in HWC layout or uint8/float
    tensors in CHW layout, with optional matching leading dimensions such as
    time. Floating-point inputs must already be in [0, 1].

    This intentionally reproduces the preprocessing used by the existing
    training runs: every camera first goes through 240x320, then the head is
    resized to 256x320 and each wrist to 128x160. The output is float32 in
    CHW layout and normalized to [-1, 1].
    """

    cameras = [
        _as_float_chw(head, name="head"),
        _as_float_chw(left_wrist, name="left_wrist"),
        _as_float_chw(right_wrist, name="right_wrist"),
    ]
    expected_prefix = cameras[0].shape[:-3]
    expected_device = cameras[0].device
    for name, camera in zip(
        ("head", "left_wrist", "right_wrist"),
        cameras,
        strict=True,
    ):
        if camera.shape[:-3] != expected_prefix:
            raise ValueError(
                "RoboTwin camera leading dimensions must match: "
                f"expected {tuple(expected_prefix)}, got {tuple(camera.shape[:-3])} "
                f"for `{name}`."
            )
        if camera.device != expected_device:
            raise ValueError(
                "RoboTwin cameras must be on the same device: "
                f"expected {expected_device}, got {camera.device} for `{name}`."
            )

    head_tensor, left_tensor, right_tensor = [
        _resize(camera, _PROCESSOR_SIZE) for camera in cameras
    ]
    head_tensor = _resize(head_tensor, _HEAD_SIZE)
    left_tensor = _resize(left_tensor, _WRIST_SIZE)
    right_tensor = _resize(right_tensor, _WRIST_SIZE)

    bottom = torch.cat((left_tensor, right_tensor), dim=-1)
    canvas = torch.cat((head_tensor, bottom), dim=-2)
    return canvas.mul(2.0).sub(1.0)
