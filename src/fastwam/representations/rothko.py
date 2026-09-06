"""Rothko raymap representation for dual-arm RoboTwin end-effector actions.

The implementation is intentionally pure PyTorch.  It follows the geometry in
``rothko_raymap_vae_sim_replay.py`` without requiring PyTorch3D:

* each arm is represented by a 192 x 160, 3-channel raymap;
* the center rectangle stores translation relative to frame zero;
* the periphery stores the rotated canonical ray directions;
* left/right arm maps are concatenated horizontally;
* the 192 x 320 result is duplicated vertically to produce 384 x 320;
* normalized gripper values are stored in the decoder-ignored 8-pixel border.

Quaternions use the RoboTwin convention ``wxyz`` throughout.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


ROTHKO_DECODE_MODE_LEGACY = "legacy"
ROTHKO_DECODE_MODE_ROBUST_JOINT = "robust_joint"
ROTHKO_DECODE_MODE_ROBUST_TILEWISE = "robust_tilewise"
ROTHKO_DECODE_MODE_ROBUST_BLOCK_CONSENSUS = "robust_block_consensus"
SUPPORTED_ROTHKO_DECODE_MODES = {
    ROTHKO_DECODE_MODE_LEGACY,
    ROTHKO_DECODE_MODE_ROBUST_JOINT,
    ROTHKO_DECODE_MODE_ROBUST_TILEWISE,
    ROTHKO_DECODE_MODE_ROBUST_BLOCK_CONSENSUS,
}


def _validate_decode_mode(decode_mode: str) -> str:
    decode_mode = str(decode_mode)
    if decode_mode not in SUPPORTED_ROTHKO_DECODE_MODES:
        raise ValueError(
            f"Unsupported Rothko decode_mode={decode_mode!r}; expected one of "
            f"{sorted(SUPPORTED_ROTHKO_DECODE_MODES)}."
        )
    return decode_mode


def _validate_decode_anchor_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "Rothko decode_anchor_alpha must be in [0,1], got "
            f"{alpha}."
        )
    return alpha


def _validate_decode_block_grid(block_grid: int) -> int:
    block_grid = int(block_grid)
    if block_grid < 2:
        raise ValueError(
            "Rothko decode_block_grid must be at least 2, got "
            f"{block_grid}."
        )
    return block_grid


@dataclass(frozen=True)
class RothkoCodecConfig:
    image_height: int = 384
    image_width: int = 320
    arm_height: int = 192
    arm_width: int = 160
    focal: float = 0.2
    center_scale: float = 1.0
    dir_scale: float = 1.0
    center_frac: float = 0.5
    boundary_margin: int = 8
    outer_margin: int = 8
    duplicate_vertical: bool = True

    def validate(self) -> None:
        if self.image_width != 2 * self.arm_width:
            raise ValueError(
                "Rothko image width must equal two arm widths, got "
                f"{self.image_width} and {self.arm_width}."
            )
        expected_height = 2 * self.arm_height if self.duplicate_vertical else self.arm_height
        if self.image_height != expected_height:
            raise ValueError(
                f"Rothko image height must be {expected_height}, got {self.image_height}."
            )
        if self.focal <= 0 or self.center_scale <= 0 or self.dir_scale <= 0:
            raise ValueError("focal, center_scale, and dir_scale must be positive.")
        if not 0.0 < self.center_frac < 1.0:
            raise ValueError(f"center_frac must be in (0,1), got {self.center_frac}.")
        if min(self.boundary_margin, self.outer_margin) < 0:
            raise ValueError("boundary_margin and outer_margin must be non-negative.")
        if 2 * self.outer_margin >= min(self.arm_height, self.arm_width):
            raise ValueError("outer_margin is too large for an arm tile.")


@dataclass
class RothkoNormStats:
    lo: torch.Tensor
    hi: torch.Tensor
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        self.lo = torch.as_tensor(self.lo, dtype=torch.float32).contiguous()
        self.hi = torch.as_tensor(self.hi, dtype=torch.float32).contiguous()
        if self.lo.shape != self.hi.shape:
            raise ValueError(
                "Rothko normalization lo/hi shape mismatch: "
                f"{tuple(self.lo.shape)} vs {tuple(self.hi.shape)}."
            )
        if self.lo.ndim != 4 or self.lo.shape[0] != 1 or self.lo.shape[1] != 3:
            raise ValueError(
                "Rothko normalization tensors must have shape [1,3,H,W], got "
                f"{tuple(self.lo.shape)}."
            )
        if not torch.isfinite(self.lo).all() or not torch.isfinite(self.hi).all():
            raise ValueError("Rothko normalization tensors contain NaN or Inf.")
        if not torch.all(self.hi > self.lo):
            invalid = int((self.hi <= self.lo).sum())
            raise ValueError(
                "Rothko normalization requires hi > lo at every pixel; "
                f"found {invalid} invalid values."
            )

    def fingerprint(self) -> str:
        """Content identity independent of the stats filename and metadata paths."""
        digest = hashlib.sha256()
        for name, tensor in (("lo", self.lo), ("hi", self.hi)):
            value = tensor.detach().cpu().to(torch.float32).contiguous()
            digest.update(name.encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes(order="C"))
        return digest.hexdigest()

    @classmethod
    def load(cls, path: str | Path) -> "RothkoNormStats":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "lo" not in payload or "hi" not in payload:
            raise ValueError(f"Invalid Rothko normalization checkpoint: {path}")
        result = cls(
            lo=torch.as_tensor(payload["lo"], dtype=torch.float32),
            hi=torch.as_tensor(payload["hi"], dtype=torch.float32),
            metadata=dict(payload.get("metadata") or {}),
        )
        version = result.metadata.get("stats_format_version")
        if version is not None and int(version) > 2:
            raise ValueError(
                f"Unsupported Rothko stats format version {version} in {path}."
            )
        return result


def quaternion_wxyz_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    if quaternion.shape[-1] != 4:
        raise ValueError(f"Expected quaternion [...,4], got {tuple(quaternion.shape)}")
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def matrix_to_quaternion_wxyz(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to normalized real-first quaternions.

    This branch-free candidate selection is stable near 180-degree rotations
    and matches the sign-insensitive semantics of a rotation quaternion.
    """
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrix [...,3,3], got {tuple(matrix.shape)}")
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    candidates = torch.stack(
        (
            torch.stack((1 + m00 + m11 + m22, m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, 1 + m00 - m11 - m22, m01 + m10, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m01 + m10, 1 - m00 + m11 - m22, m12 + m21), dim=-1),
            torch.stack((m10 - m01, m02 + m20, m12 + m21, 1 - m00 - m11 + m22), dim=-1),
        ),
        dim=-2,
    )
    denominators = (candidates[..., :, 0].clamp_min(0.0).sqrt() * 2.0).clamp_min(1e-8)
    candidates = candidates / denominators.unsqueeze(-1)
    best = denominators.argmax(dim=-1, keepdim=True)
    gather_index = best[..., None].expand(best.shape + (4,))
    quaternion = torch.gather(candidates, dim=-2, index=gather_index).squeeze(-2)
    return quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _center_and_read_masks(
    height: int,
    width: int,
    *,
    center_frac: float,
    boundary_margin: int,
    outer_margin: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    center_h = max(1, int(round(height * center_frac)))
    center_w = max(1, int(round(width * center_frac)))
    y0, x0 = (height - center_h) // 2, (width - center_w) // 2
    y1, x1 = y0 + center_h, x0 + center_w

    center = torch.zeros(height, width, dtype=torch.bool, device=device)
    center[y0:y1, x0:x1] = True

    origin = torch.zeros_like(center)
    iy0, iy1 = max(y0 + boundary_margin, 0), min(y1 - boundary_margin, height)
    ix0, ix1 = max(x0 + boundary_margin, 0), min(x1 - boundary_margin, width)
    if iy1 > iy0 and ix1 > ix0:
        origin[iy0:iy1, ix0:ix1] = True
    else:
        origin.copy_(center)

    expanded_center = torch.zeros_like(center)
    expanded_center[
        max(y0 - boundary_margin, 0) : min(y1 + boundary_margin, height),
        max(x0 - boundary_margin, 0) : min(x1 + boundary_margin, width),
    ] = True
    direction = ~expanded_center
    if outer_margin:
        valid = torch.zeros_like(center)
        valid[
            outer_margin : height - outer_margin,
            outer_margin : width - outer_margin,
        ] = True
        direction &= valid
    if not origin.any() or not direction.any():
        raise ValueError("Rothko decode mask is empty.")
    return center, origin, direction


def _border_mask(height: int, width: int, margin: int, device: torch.device) -> torch.Tensor:
    mask = torch.ones(height, width, dtype=torch.bool, device=device)
    if margin:
        mask[margin : height - margin, margin : width - margin] = False
    else:
        mask.zero_()
    return mask


def _huber_location(
    values: torch.Tensor,
    *,
    sample_dim: int,
    num_iterations: int = 2,
    tuning: float = 1.5,
) -> torch.Tensor:
    """Return a component-wise robust location without rejecting any sample."""
    estimate = _midpoint_median(values, dim=sample_dim)
    eps = torch.finfo(values.dtype).eps
    for _ in range(num_iterations):
        residual = values - estimate.unsqueeze(sample_dim)
        mad = _midpoint_median(residual.abs(), dim=sample_dim)
        delta = (float(tuning) * 1.4826 * mad).clamp_min(1e-6)
        weights = torch.clamp(
            delta.unsqueeze(sample_dim) / residual.abs().clamp_min(eps),
            max=1.0,
        )
        estimate = (weights * values).sum(dim=sample_dim) / weights.sum(
            dim=sample_dim
        ).clamp_min(eps)
    return estimate


def _midpoint_median(values: torch.Tensor, *, dim: int) -> torch.Tensor:
    """Median whose even-sample value is the midpoint of both central values."""
    lower = values.median(dim=dim).values
    if values.shape[dim] % 2:
        return lower
    upper = -(-values).median(dim=dim).values
    return (lower + upper) * 0.5


def _proper_rotation_from_correlation(correlation: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(correlation)
    determinant = torch.linalg.det(u @ vh)
    correction = torch.eye(
        3, dtype=correlation.dtype, device=correlation.device
    ).expand(correlation.shape[:-2] + (3, 3)).clone()
    correction[..., 2, 2] = torch.where(determinant >= 0, 1.0, -1.0)
    return u @ correction @ vh


def _robust_relative_rotation(
    direction: torch.Tensor,
    reference: torch.Tensor,
    *,
    num_iterations: int = 2,
    tuning: float = 1.5,
) -> torch.Tensor:
    """Joint iteratively reweighted Kabsch fit over all duplicate-tile rays."""
    eps = torch.finfo(direction.dtype).eps
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    reference = reference / reference.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    weights = torch.ones(
        direction.shape[:-1], dtype=direction.dtype, device=direction.device
    )
    rotation = None
    for iteration in range(num_iterations + 1):
        correlation = torch.einsum(
            "btni,btnj,btn->btij", direction, reference, weights
        )
        correlation = correlation / weights.sum(dim=-1, keepdim=True).clamp_min(
            eps
        ).unsqueeze(-1)
        rotation = _proper_rotation_from_correlation(correlation)
        if iteration == num_iterations:
            break
        fitted = torch.einsum("btij,btnj->btni", rotation, reference)
        residual = (direction - fitted).norm(dim=-1)
        median = _midpoint_median(residual, dim=-1).unsqueeze(-1)
        mad = _midpoint_median((residual - median).abs(), dim=-1).unsqueeze(-1)
        delta = (median + float(tuning) * 1.4826 * mad).clamp_min(1e-6)
        weights = torch.clamp(delta / residual.clamp_min(eps), max=1.0)
    assert rotation is not None
    return rotation


def _blend_direction_reference(
    predicted: torch.Tensor,
    canonical: torch.Tensor,
    *,
    anchor_alpha: float,
) -> torch.Tensor:
    """Blend a predicted frame-zero ray field with the analytic template."""
    predicted = predicted / predicted.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    canonical = canonical / canonical.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    alpha = float(anchor_alpha)
    if alpha == 0.0:
        return predicted
    reference = predicted * (1.0 - alpha) + canonical * alpha
    return reference / reference.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def _robust_rotation_average(
    rotations: torch.Tensor,
    *,
    sample_dim: int,
    num_iterations: int = 2,
    tuning: float = 1.5,
) -> torch.Tensor:
    """Return an iteratively reweighted chordal mean on SO(3)."""
    rotations = rotations.movedim(sample_dim, -3)
    eps = torch.finfo(rotations.dtype).eps
    weights = torch.ones(
        rotations.shape[:-2], dtype=rotations.dtype, device=rotations.device
    )
    mean = None
    for iteration in range(num_iterations + 1):
        correlation = (rotations * weights[..., None, None]).sum(dim=-3)
        correlation = correlation / weights.sum(dim=-1, keepdim=True).clamp_min(
            eps
        ).unsqueeze(-1)
        mean = _proper_rotation_from_correlation(correlation)
        if iteration == num_iterations:
            break
        relative = rotations @ mean.unsqueeze(-3).transpose(-1, -2)
        cosine = (
            (relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5
        ).clamp(-1.0, 1.0)
        residual = torch.acos(cosine)
        median = _midpoint_median(residual, dim=-1).unsqueeze(-1)
        mad = _midpoint_median((residual - median).abs(), dim=-1).unsqueeze(-1)
        delta = (median + float(tuning) * 1.4826 * mad).clamp_min(1e-6)
        weights = torch.clamp(delta / residual.clamp_min(eps), max=1.0)
    assert mean is not None
    return mean


def _absolute_pose_from_relative(
    relative_position: torch.Tensor,
    relative_rotation: torch.Tensor,
    base_pose7: torch.Tensor,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    relative_position = relative_position.clone()
    relative_rotation = relative_rotation.clone()
    relative_position[:, 0] = 0
    relative_rotation[:, 0] = torch.eye(
        3, dtype=relative_rotation.dtype, device=relative_rotation.device
    )
    base_position = base_pose7[:, None, :3].to(relative_position.dtype)
    base_rotation = quaternion_wxyz_to_matrix(
        base_pose7[:, 3:7].to(relative_rotation.dtype)
    )[:, None]
    absolute_position = base_position + torch.einsum(
        "btij,btj->bti", base_rotation, relative_position
    )
    absolute_rotation = base_rotation @ relative_rotation
    quaternion = matrix_to_quaternion_wxyz(absolute_rotation)
    return torch.cat((absolute_position, quaternion), dim=-1).to(output_dtype)


def _decode_pose_tiles_robust_joint(
    tiles: torch.Tensor,
    base_pose7: torch.Tensor,
    *,
    center_frac: float,
    boundary_margin: int,
    outer_margin: int,
    center_scale: float,
    canonical_directions: torch.Tensor | None = None,
    anchor_alpha: float = 0.0,
    anchor_tiles: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode one pose from ``[B,T,K,3,H,W]`` duplicate Rothko tiles."""
    if tiles.ndim != 6 or tiles.shape[3] != 3:
        raise ValueError(
            "Expected robust Rothko tiles [B,T,K,3,H,W], got "
            f"{tuple(tiles.shape)}."
        )
    batch, time, copies, _, height, width = tiles.shape
    _, origin_mask, direction_mask = _center_and_read_masks(
        height,
        width,
        center_frac=center_frac,
        boundary_margin=boundary_margin,
        outer_margin=outer_margin,
        device=tiles.device,
    )
    values = tiles.permute(0, 1, 2, 4, 5, 3).to(torch.float64)
    values = values.reshape(batch, time, copies, height * width, 3)

    origin_values = values[:, :, :, origin_mask.reshape(-1), :]
    origin_values = origin_values.reshape(batch, time, -1, 3)
    origin_code = _huber_location(origin_values, sample_dim=2)
    predicted_origin_reference = origin_code[:, :1].expand(-1, time, -1)
    anchor_origin = torch.zeros_like(predicted_origin_reference)
    anchor_direction = None
    if anchor_tiles is not None:
        if anchor_tiles.ndim != 6 or anchor_tiles.shape[2:] != tiles.shape[2:]:
            raise ValueError(
                "Rothko anchor tile shape mismatch: "
                f"anchor={tuple(anchor_tiles.shape)} tiles={tuple(tiles.shape)}."
            )
        if anchor_tiles.shape[0] not in (1, batch) or anchor_tiles.shape[1] != time:
            raise ValueError(
                "Rothko anchor tiles must have batch 1 or the decode batch and "
                f"matching time, got {tuple(anchor_tiles.shape)}."
            )
        anchor_values = anchor_tiles.permute(0, 1, 2, 4, 5, 3).to(
            device=tiles.device, dtype=torch.float64
        )
        anchor_values = anchor_values.reshape(
            anchor_values.shape[0], time, copies, height * width, 3
        )
        anchor_origin_values = anchor_values[
            :, :, :, origin_mask.reshape(-1), :
        ].reshape(anchor_values.shape[0], time, -1, 3)
        anchor_origin = _huber_location(anchor_origin_values, sample_dim=2)
        anchor_origin = anchor_origin.expand(batch, -1, -1)
        anchor_direction = anchor_values[
            :, :, :, direction_mask.reshape(-1), :
        ].reshape(anchor_values.shape[0], time, -1, 3)
        anchor_direction = anchor_direction.expand(batch, -1, -1, -1)
    origin_reference = (
        predicted_origin_reference * (1.0 - float(anchor_alpha))
        + anchor_origin * float(anchor_alpha)
    )
    relative_position = (origin_code - origin_reference) / float(center_scale)

    direction = values[:, :, :, direction_mask.reshape(-1), :]
    direction = direction.reshape(batch, time, -1, 3)
    predicted_reference = direction[:, :1].expand(-1, time, -1, -1)
    if float(anchor_alpha) != 0.0:
        if anchor_direction is None and canonical_directions is None:
            raise ValueError(
                "canonical_directions or anchor_tiles are required when "
                "anchor_alpha is nonzero."
            )
        if anchor_direction is None:
            canonical = canonical_directions.to(
                device=tiles.device, dtype=torch.float64
            )[direction_mask]
            canonical = canonical.reshape(1, 1, 1, -1, 3).expand(
                batch, time, copies, -1, -1
            )
            anchor_direction = canonical.reshape(batch, time, -1, 3)
        predicted_reference = _blend_direction_reference(
            predicted_reference,
            anchor_direction,
            anchor_alpha=anchor_alpha,
        )
    relative_rotation = _robust_relative_rotation(direction, predicted_reference)
    return _absolute_pose_from_relative(
        relative_position,
        relative_rotation,
        base_pose7,
        output_dtype=tiles.dtype,
    )


def _decode_pose_tiles_robust_tilewise(
    tiles: torch.Tensor,
    base_pose7: torch.Tensor,
    *,
    center_frac: float,
    boundary_margin: int,
    outer_margin: int,
    center_scale: float,
    canonical_directions: torch.Tensor | None = None,
    anchor_alpha: float = 0.0,
    anchor_tiles: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode each duplicate tile independently, then fuse poses on SE(3)."""
    if tiles.ndim != 6 or tiles.shape[3] != 3:
        raise ValueError(
            "Expected robust Rothko tiles [B,T,K,3,H,W], got "
            f"{tuple(tiles.shape)}."
        )
    candidates = []
    for index in range(tiles.shape[2]):
        candidates.append(
            _decode_pose_tiles_robust_joint(
                tiles[:, :, index : index + 1],
                base_pose7,
                center_frac=center_frac,
                boundary_margin=boundary_margin,
                outer_margin=outer_margin,
                center_scale=center_scale,
                canonical_directions=canonical_directions,
                anchor_alpha=anchor_alpha,
                anchor_tiles=(
                    None
                    if anchor_tiles is None
                    else anchor_tiles[:, :, index : index + 1]
                ),
            )
        )
    poses = torch.stack(candidates, dim=2).to(torch.float64)
    position = _huber_location(poses[..., :3], sample_dim=2)
    rotations = quaternion_wxyz_to_matrix(poses[..., 3:7])
    rotation = _robust_rotation_average(rotations, sample_dim=2)
    quaternion = matrix_to_quaternion_wxyz(rotation)
    return torch.cat((position, quaternion), dim=-1).to(tiles.dtype)


def _decode_pose_tiles_robust_block_consensus(
    tiles: torch.Tensor,
    base_pose7: torch.Tensor,
    *,
    center_frac: float,
    boundary_margin: int,
    outer_margin: int,
    center_scale: float,
    canonical_directions: torch.Tensor,
    anchor_alpha: float = 0.0,
    block_grid: int = 4,
    anchor_tiles: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse block-level translation and rotation estimates across duplicates."""
    if tiles.ndim != 6 or tiles.shape[3] != 3:
        raise ValueError(
            "Expected robust Rothko tiles [B,T,K,3,H,W], got "
            f"{tuple(tiles.shape)}."
        )
    batch, time, copies, _, height, width = tiles.shape
    _, origin_mask, direction_mask = _center_and_read_masks(
        height,
        width,
        center_frac=center_frac,
        boundary_margin=boundary_margin,
        outer_margin=outer_margin,
        device=tiles.device,
    )
    values = tiles.permute(0, 1, 2, 4, 5, 3).to(torch.float64)
    canonical = canonical_directions.to(
        device=tiles.device, dtype=torch.float64
    )
    if canonical.shape != (height, width, 3):
        raise ValueError(
            "Canonical Rothko direction shape mismatch: "
            f"{tuple(canonical.shape)} vs {(height, width, 3)}."
        )
    anchor_values = None
    if anchor_tiles is not None:
        if anchor_tiles.ndim != 6 or anchor_tiles.shape[2:] != tiles.shape[2:]:
            raise ValueError(
                "Rothko anchor tile shape mismatch: "
                f"anchor={tuple(anchor_tiles.shape)} tiles={tuple(tiles.shape)}."
            )
        if anchor_tiles.shape[0] not in (1, batch) or anchor_tiles.shape[1] != time:
            raise ValueError(
                "Rothko anchor tiles must have batch 1 or the decode batch and "
                f"matching time, got {tuple(anchor_tiles.shape)}."
            )
        anchor_values = anchor_tiles.permute(0, 1, 2, 4, 5, 3).to(
            device=tiles.device, dtype=torch.float64
        )
        anchor_values = anchor_values.expand(batch, -1, -1, -1, -1, -1)

    position_candidates: list[torch.Tensor] = []
    rotation_candidates: list[torch.Tensor] = []
    y_edges = torch.linspace(0, height, block_grid + 1).round().to(torch.int64)
    x_edges = torch.linspace(0, width, block_grid + 1).round().to(torch.int64)
    for copy_index in range(copies):
        tile = values[:, :, copy_index]
        for y_index in range(block_grid):
            y0, y1 = int(y_edges[y_index]), int(y_edges[y_index + 1])
            for x_index in range(block_grid):
                x0, x1 = int(x_edges[x_index]), int(x_edges[x_index + 1])
                origin_block = origin_mask[y0:y1, x0:x1]
                if int(origin_block.sum()) >= 4:
                    samples = tile[:, :, y0:y1, x0:x1][..., origin_block, :]
                    location = _huber_location(samples, sample_dim=2)
                    predicted_reference = location[:, :1].expand(-1, time, -1)
                    if anchor_values is None:
                        anchor_reference = torch.zeros_like(predicted_reference)
                    else:
                        anchor_samples = anchor_values[
                            :, :, copy_index, y0:y1, x0:x1
                        ][..., origin_block, :]
                        anchor_reference = _huber_location(
                            anchor_samples, sample_dim=2
                        )
                    reference = (
                        predicted_reference * (1.0 - float(anchor_alpha))
                        + anchor_reference * float(anchor_alpha)
                    )
                    position_candidates.append(
                        (location - reference) / float(center_scale)
                    )

                direction_block = direction_mask[y0:y1, x0:x1]
                if int(direction_block.sum()) >= 8:
                    samples = tile[:, :, y0:y1, x0:x1][..., direction_block, :]
                    predicted_reference = samples[:, :1].expand(-1, time, -1, -1)
                    if anchor_values is None:
                        anchor_reference = canonical[y0:y1, x0:x1][direction_block]
                        anchor_reference = anchor_reference.reshape(
                            1, 1, -1, 3
                        ).expand(batch, time, -1, -1)
                    else:
                        anchor_reference = anchor_values[
                            :, :, copy_index, y0:y1, x0:x1
                        ][..., direction_block, :]
                    reference = _blend_direction_reference(
                        predicted_reference,
                        anchor_reference,
                        anchor_alpha=anchor_alpha,
                    )
                    rotation_candidates.append(
                        _robust_relative_rotation(samples, reference)
                    )

    if not position_candidates or not rotation_candidates:
        raise ValueError(
            "Rothko block consensus produced no valid position or rotation blocks."
        )
    relative_position = _huber_location(
        torch.stack(position_candidates, dim=2), sample_dim=2
    )
    relative_rotation = _robust_rotation_average(
        torch.stack(rotation_candidates, dim=2), sample_dim=2
    )
    return _absolute_pose_from_relative(
        relative_position,
        relative_rotation,
        base_pose7,
        output_dtype=tiles.dtype,
    )


class RothkoCodec:
    def __init__(
        self,
        config: RothkoCodecConfig | None = None,
        norm_stats: RothkoNormStats | str | Path | None = None,
        decode_mode: str = ROTHKO_DECODE_MODE_LEGACY,
        decode_anchor_alpha: float = 0.0,
        decode_block_grid: int = 4,
    ):
        self.config = config or RothkoCodecConfig()
        self.config.validate()
        self.decode_mode = _validate_decode_mode(decode_mode)
        self.decode_anchor_alpha = _validate_decode_anchor_alpha(
            decode_anchor_alpha
        )
        self.decode_block_grid = _validate_decode_block_grid(decode_block_grid)
        self._decode_anchor_raw: torch.Tensor | None = None
        if isinstance(norm_stats, (str, Path)):
            norm_stats = RothkoNormStats.load(norm_stats)
        self.norm_stats = norm_stats
        if self.norm_stats is not None:
            expected_shape = (1, 3, self.config.image_height, self.config.image_width)
            if tuple(self.norm_stats.lo.shape) != expected_shape:
                raise ValueError(
                    "Rothko stats tensor shape mismatch: "
                    f"stats={tuple(self.norm_stats.lo.shape)} codec={expected_shape}."
                )
            self._validate_stats_metadata(self.norm_stats.metadata)

    def set_decode_anchor_video(self, normalized_video: torch.Tensor) -> None:
        """Cache a VAE-reconstructed zero-motion template for pose anchoring."""
        if normalized_video.ndim != 5 or normalized_video.shape[0] != 1:
            raise ValueError(
                "Rothko anchor video must be [1,3,T,H,W], got "
                f"{tuple(normalized_video.shape)}."
            )
        if normalized_video.shape[1] != 3 or normalized_video.shape[-2:] != (
            self.config.image_height,
            self.config.image_width,
        ):
            raise ValueError(
                "Rothko anchor video channel/spatial shape mismatch: "
                f"{tuple(normalized_video.shape)}."
            )
        normalized = normalized_video.permute(0, 2, 1, 3, 4).contiguous()
        self._decode_anchor_raw = self.denormalize(normalized).detach()

    def metadata(self) -> dict[str, Any]:
        return {
            "representation": "rothko",
            **asdict(self.config),
            "gripper_encoding": "normalized_outer_border_2g_minus_1",
            "quaternion_order": "wxyz",
        }

    def _validate_stats_metadata(self, metadata: dict[str, Any]) -> None:
        expected = {
            "representation": "rothko",
            "active_shape": [self.config.image_height, self.config.image_width],
            "focal": self.config.focal,
            "center_scale": self.config.center_scale,
            "dir_scale": self.config.dir_scale,
            "center_frac": self.config.center_frac,
            "boundary_margin": self.config.boundary_margin,
            "outer_margin": self.config.outer_margin,
            "duplicate_vertical": self.config.duplicate_vertical,
        }
        strict = int(metadata.get("stats_format_version", 1)) >= 2
        for key, value in expected.items():
            if key not in metadata:
                if strict:
                    raise ValueError(
                        f"Rothko stats v2 metadata is missing required key {key!r}."
                    )
                continue
            actual = metadata[key]
            matches = (
                abs(float(actual) - float(value)) <= 1e-8
                if isinstance(value, float)
                else actual == value
            )
            if not matches:
                raise ValueError(
                    f"Rothko stats metadata mismatch for {key}: "
                    f"stats={actual!r} codec={value!r}"
                )

    @staticmethod
    def _ensure_batched_pose(pose: torch.Tensor) -> tuple[torch.Tensor, bool]:
        squeeze = pose.ndim == 2
        if squeeze:
            pose = pose.unsqueeze(0)
        if pose.ndim != 3 or pose.shape[-1] != 14:
            raise ValueError(f"Expected pose [B,T,14] or [T,14], got {tuple(pose.shape)}")
        return pose, squeeze

    def _canonical_directions(
        self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        cfg = self.config
        dx, dy = 1.0 / cfg.arm_width, 1.0 / cfg.arm_height
        y, x = torch.meshgrid(
            torch.linspace(1 - dy, -(1 - dy), cfg.arm_height, device=device, dtype=dtype),
            torch.linspace(1 - dx, -(1 - dx), cfg.arm_width, device=device, dtype=dtype),
            indexing="ij",
        )
        directions = torch.stack((x / cfg.focal, y / cfg.focal, torch.ones_like(x)), dim=-1)
        return directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def _encode_arm_raw(self, pose7: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        position = pose7[..., :3]
        rotation = quaternion_wxyz_to_matrix(pose7[..., 3:7])
        base_position = position[:, :1]
        base_rotation = rotation[:, :1]
        relative_position = torch.einsum(
            "btij,btj->bti",
            base_rotation.transpose(-1, -2),
            position - base_position,
        )
        relative_rotation = base_rotation.transpose(-1, -2) @ rotation

        canonical = self._canonical_directions(pose7.device, pose7.dtype)
        directions = torch.einsum(
            "btij,hwj->btihw", relative_rotation, canonical
        )
        directions = directions * cfg.dir_scale
        center, _, _ = _center_and_read_masks(
            cfg.arm_height,
            cfg.arm_width,
            center_frac=cfg.center_frac,
            boundary_margin=0,
            outer_margin=0,
            device=pose7.device,
        )
        output = directions.clone()
        center_values = (relative_position * cfg.center_scale)[..., :, None]
        output[..., center] = center_values
        return output

    def encode_raw(self, pose14: torch.Tensor) -> torch.Tensor:
        """Encode absolute dual-arm poses to raw raymaps.

        Args:
            pose14: ``[B,T,14]`` or ``[T,14]`` absolute EE pose sequence.

        Returns:
            ``[B,T,3,384,320]`` (or ``[T,3,384,320]`` for unbatched input).
        """
        pose14, squeeze = self._ensure_batched_pose(pose14)
        left = self._encode_arm_raw(pose14[..., :7])
        right = self._encode_arm_raw(pose14[..., 7:14])
        top = torch.cat((left, right), dim=-1)
        output = torch.cat((top, top), dim=-2) if self.config.duplicate_vertical else top
        return output.squeeze(0) if squeeze else output

    def _expanded_stats(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.norm_stats is None:
            raise ValueError("Rothko normalization stats are required for normalized encode/decode.")
        lo = self.norm_stats.lo.to(device=device, dtype=dtype)
        hi = self.norm_stats.hi.to(device=device, dtype=dtype)
        if lo.ndim != 4 or hi.shape != lo.shape or lo.shape[0:2] != (1, 3):
            raise ValueError(
                f"Expected Rothko stats [1,3,H,W], got {tuple(lo.shape)} and {tuple(hi.shape)}"
            )
        if lo.shape[-2:] == (self.config.arm_height, self.config.image_width):
            lo = torch.cat((lo, lo), dim=-2)
            hi = torch.cat((hi, hi), dim=-2)
        if lo.shape[-2:] != (self.config.image_height, self.config.image_width):
            raise ValueError(
                "Rothko stats spatial shape mismatch: "
                f"{tuple(lo.shape[-2:])} vs "
                f"{(self.config.image_height, self.config.image_width)}"
            )
        return lo, hi

    def normalize_raw(self, raw: torch.Tensor) -> torch.Tensor:
        squeeze = raw.ndim == 4
        if squeeze:
            raw = raw.unsqueeze(0)
        if raw.ndim != 5 or raw.shape[2:] != (
            3,
            self.config.image_height,
            self.config.image_width,
        ):
            raise ValueError(f"Expected raw Rothko [B,T,3,H,W], got {tuple(raw.shape)}")
        lo, hi = self._expanded_stats(device=raw.device, dtype=raw.dtype)
        span = (hi - lo).clamp_min(1e-6)
        normalized = (2.0 * (raw - lo.unsqueeze(1)) / span.unsqueeze(1) - 1.0).clamp(-1, 1)
        return normalized.squeeze(0) if squeeze else normalized

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        squeeze = normalized.ndim == 4
        if squeeze:
            normalized = normalized.unsqueeze(0)
        if normalized.ndim != 5:
            raise ValueError(
                f"Expected normalized Rothko [B,T,3,H,W], got {tuple(normalized.shape)}"
            )
        lo, hi = self._expanded_stats(device=normalized.device, dtype=normalized.dtype)
        raw = (normalized + 1.0) * 0.5 * (hi - lo).unsqueeze(1) + lo.unsqueeze(1)
        return raw.squeeze(0) if squeeze else raw

    def write_gripper(self, normalized: torch.Tensor, gripper: torch.Tensor) -> torch.Tensor:
        squeeze = normalized.ndim == 4
        if squeeze:
            normalized = normalized.unsqueeze(0)
            gripper = gripper.unsqueeze(0) if gripper.ndim == 2 else gripper
        if gripper.ndim != 3 or gripper.shape[-1] != 2:
            raise ValueError(f"Expected gripper [B,T,2], got {tuple(gripper.shape)}")
        if normalized.shape[:2] != gripper.shape[:2]:
            raise ValueError(
                f"Rothko/gripper time shape mismatch: {normalized.shape[:2]} vs {gripper.shape[:2]}"
            )
        output = normalized.clone()
        border = _border_mask(
            self.config.arm_height,
            self.config.arm_width,
            self.config.outer_margin,
            normalized.device,
        )
        code = gripper.to(device=normalized.device, dtype=normalized.dtype).clamp(0, 1) * 2 - 1
        half_offsets = (0, self.config.arm_height) if self.config.duplicate_vertical else (0,)
        for y_offset in half_offsets:
            for arm_index, x_offset in enumerate((0, self.config.arm_width)):
                tile = output[
                    :,
                    :,
                    :,
                    y_offset : y_offset + self.config.arm_height,
                    x_offset : x_offset + self.config.arm_width,
                ]
                tile[..., border] = code[..., arm_index, None, None].expand(
                    -1, -1, 3, int(border.sum().item())
                )
        return output.squeeze(0) if squeeze else output

    def read_gripper(self, normalized: torch.Tensor) -> torch.Tensor:
        squeeze = normalized.ndim == 4
        if squeeze:
            normalized = normalized.unsqueeze(0)
        border = _border_mask(
            self.config.arm_height,
            self.config.arm_width,
            self.config.outer_margin,
            normalized.device,
        )
        decoded = []
        half_offsets = (0, self.config.arm_height) if self.config.duplicate_vertical else (0,)
        for arm_index, x_offset in enumerate((0, self.config.arm_width)):
            observations = []
            for y_offset in half_offsets:
                tile = normalized[
                    :,
                    :,
                    :,
                    y_offset : y_offset + self.config.arm_height,
                    x_offset : x_offset + self.config.arm_width,
                ]
                observations.append(tile[..., border].reshape(*tile.shape[:2], -1))
            code = torch.cat(observations, dim=-1).median(dim=-1).values
            decoded.append(((code + 1.0) * 0.5).clamp(0, 1))
        gripper = torch.stack(decoded, dim=-1)
        return gripper.squeeze(0) if squeeze else gripper

    def encode(self, pose14: torch.Tensor, gripper: torch.Tensor) -> torch.Tensor:
        """Return VAE-ready normalized video in ``[B,3,T,H,W]`` format."""
        pose14_batched, squeeze = self._ensure_batched_pose(pose14)
        if gripper.ndim == 2:
            gripper = gripper.unsqueeze(0)
        normalized = self.normalize_raw(self.encode_raw(pose14_batched))
        normalized = self.write_gripper(normalized, gripper)
        video = normalized.permute(0, 2, 1, 3, 4).contiguous()
        return video.squeeze(0) if squeeze else video

    def _decode_arm_raw(
        self, arm: torch.Tensor, base_pose7: torch.Tensor
    ) -> torch.Tensor:
        cfg = self.config
        _, origin_mask, direction_mask = _center_and_read_masks(
            cfg.arm_height,
            cfg.arm_width,
            center_frac=cfg.center_frac,
            boundary_margin=cfg.boundary_margin,
            outer_margin=cfg.outer_margin,
            device=arm.device,
        )
        values = arm.permute(0, 1, 3, 4, 2).to(torch.float64)
        origin_code = values[:, :, origin_mask].median(dim=2).values
        relative_position = (origin_code - origin_code[:, :1]) / cfg.center_scale

        direction = values[:, :, direction_mask]
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        reference = direction[:, 0]
        correlation = torch.einsum("btpi,bpj->btij", direction, reference)
        correlation = correlation / direction.shape[2]
        u, _, vh = torch.linalg.svd(correlation)
        determinant = torch.linalg.det(u @ vh)
        correction = torch.eye(3, dtype=torch.float64, device=arm.device).repeat(
            arm.shape[0], arm.shape[1], 1, 1
        )
        correction[..., 2, 2] = torch.where(determinant >= 0, 1.0, -1.0)
        relative_rotation = u @ correction @ vh
        relative_position[:, 0] = 0
        relative_rotation[:, 0] = torch.eye(
            3, dtype=torch.float64, device=arm.device
        )

        base_position = base_pose7[:, None, :3].to(torch.float64)
        base_rotation = quaternion_wxyz_to_matrix(
            base_pose7[:, 3:7].to(torch.float64)
        )[:, None]
        absolute_position = base_position + torch.einsum(
            "btij,btj->bti", base_rotation, relative_position
        )
        absolute_rotation = base_rotation @ relative_rotation
        quaternion = matrix_to_quaternion_wxyz(absolute_rotation)
        return torch.cat((absolute_position, quaternion), dim=-1).to(arm.dtype)

    def decode(
        self, video: torch.Tensor, current_pose14: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode a normalized video into absolute EE poses and grippers.

        Args:
            video: ``[B,3,T,384,320]`` or ``[3,T,384,320]``.
            current_pose14: absolute frame-zero EE pose, ``[B,14]`` or ``[14]``.
        """
        squeeze = video.ndim == 4
        if squeeze:
            video = video.unsqueeze(0)
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"Expected Rothko video [B,3,T,H,W], got {tuple(video.shape)}")
        normalized = video.permute(0, 2, 1, 3, 4).contiguous()
        gripper = self.read_gripper(normalized)
        raw = self.denormalize(normalized)
        top = raw[..., : self.config.arm_height, :]
        if self.decode_mode == ROTHKO_DECODE_MODE_LEGACY and self.config.duplicate_vertical:
            bottom = raw[
                ...,
                self.config.arm_height : 2 * self.config.arm_height,
                :,
            ]
            combined = (top + bottom) * 0.5
        elif self.decode_mode == ROTHKO_DECODE_MODE_LEGACY:
            combined = top

        if current_pose14.ndim == 1:
            current_pose14 = current_pose14.unsqueeze(0)
        if current_pose14.shape != (video.shape[0], 14):
            raise ValueError(
                f"Expected current_pose14 {(video.shape[0], 14)}, got {tuple(current_pose14.shape)}"
            )
        if self.decode_mode == ROTHKO_DECODE_MODE_LEGACY:
            left = combined[..., : self.config.arm_width]
            right = combined[..., self.config.arm_width :]
            pose = torch.cat(
                (
                    self._decode_arm_raw(left, current_pose14[:, :7]),
                    self._decode_arm_raw(right, current_pose14[:, 7:14]),
                ),
                dim=-1,
            )
        else:
            halves = [top]
            if self.config.duplicate_vertical:
                halves.append(
                    raw[
                        ...,
                        self.config.arm_height : 2 * self.config.arm_height,
                        :,
                    ]
                )
            left_tiles = torch.stack(
                [half[..., : self.config.arm_width] for half in halves], dim=2
            )
            right_tiles = torch.stack(
                [half[..., self.config.arm_width :] for half in halves], dim=2
            )
            cfg = self.config
            decoder = {
                ROTHKO_DECODE_MODE_ROBUST_JOINT: _decode_pose_tiles_robust_joint,
                ROTHKO_DECODE_MODE_ROBUST_TILEWISE: _decode_pose_tiles_robust_tilewise,
                ROTHKO_DECODE_MODE_ROBUST_BLOCK_CONSENSUS: (
                    _decode_pose_tiles_robust_block_consensus
                ),
            }[self.decode_mode]
            decoder_kwargs: dict[str, Any] = {
                "center_frac": cfg.center_frac,
                "boundary_margin": cfg.boundary_margin,
                "outer_margin": cfg.outer_margin,
                "center_scale": cfg.center_scale,
                "canonical_directions": self._canonical_directions(
                    raw.device, torch.float64
                ),
                "anchor_alpha": self.decode_anchor_alpha,
            }
            if self.decode_mode == ROTHKO_DECODE_MODE_ROBUST_BLOCK_CONSENSUS:
                decoder_kwargs["block_grid"] = self.decode_block_grid
            left_kwargs = dict(decoder_kwargs)
            right_kwargs = dict(decoder_kwargs)
            if self._decode_anchor_raw is not None:
                anchor_raw = self._decode_anchor_raw.to(
                    device=raw.device, dtype=raw.dtype
                )
                anchor_halves = [anchor_raw[..., : self.config.arm_height, :]]
                if self.config.duplicate_vertical:
                    anchor_halves.append(
                        anchor_raw[
                            ...,
                            self.config.arm_height : 2 * self.config.arm_height,
                            :,
                        ]
                    )
                left_kwargs["anchor_tiles"] = torch.stack(
                    [
                        half[..., : self.config.arm_width]
                        for half in anchor_halves
                    ],
                    dim=2,
                )
                right_kwargs["anchor_tiles"] = torch.stack(
                    [
                        half[..., self.config.arm_width :]
                        for half in anchor_halves
                    ],
                    dim=2,
                )
            pose = torch.cat(
                (
                    decoder(
                        left_tiles,
                        current_pose14[:, :7],
                        **left_kwargs,
                    ),
                    decoder(
                        right_tiles,
                        current_pose14[:, 7:14],
                        **right_kwargs,
                    ),
                ),
                dim=-1,
            )
        if squeeze:
            return pose.squeeze(0), gripper.squeeze(0)
        return pose, gripper
