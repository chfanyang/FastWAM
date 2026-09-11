"""Single-arm Rothko raymap representation for LIBERO.

One 224x224 single-arm map is duplicated horizontally to match LIBERO's
``[agentview | wrist]`` 224x448 RGB canvas.  Geometry is identical in both
tiles; averaging them during decode makes the duplication explicit rather than
pretending they represent two arms.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .rothko import (
    ROTHKO_DECODE_MODE_BLOCK_POSITION_JOINT_ROTATION,
    ROTHKO_DECODE_MODE_ROBUST_BLOCK_WEIGHTED_JOINT,
    _decode_pose_tiles_robust_block_weighted_joint,
    _decode_pose_tiles_block_position_joint_rotation,
    ROTHKO_DECODE_MODE_LEGACY,
    ROTHKO_DECODE_MODE_ROBUST_BLOCK_CONSENSUS,
    ROTHKO_DECODE_MODE_ROBUST_JOINT,
    ROTHKO_DECODE_MODE_ROBUST_TILEWISE,
    RothkoNormStats,
    _border_mask,
    _center_and_read_masks,
    _decode_pose_tiles_robust_block_consensus,
    _decode_pose_tiles_robust_joint,
    _decode_pose_tiles_robust_tilewise,
    _validate_decode_anchor_alpha,
    _validate_decode_block_grid,
    _validate_decode_mode,
    matrix_to_quaternion_wxyz,
    quaternion_wxyz_to_matrix,
)


@dataclass(frozen=True)
class LiberoRothkoCodecConfig:
    image_height: int = 224
    image_width: int = 448
    tile_height: int = 224
    tile_width: int = 224
    focal: float = 0.2
    center_scale: float = 1.0
    dir_scale: float = 1.0
    center_frac: float = 0.5
    boundary_margin: int = 8
    outer_margin: int = 8
    duplicate_horizontal: bool = True
    # Opt-in: only frame zero carries world-frame pose. Future frames retain
    # the original chunk-local representation and normalization.
    frame0_pose_mode: str = "relative"
    absolute_position_min: tuple[float, float, float] | None = None
    absolute_position_max: tuple[float, float, float] | None = None

    def validate(self) -> None:
        if self.frame0_pose_mode not in {"relative", "absolute"}:
            raise ValueError("frame0_pose_mode must be relative or absolute")
        if self.frame0_pose_mode == "absolute":
            lo, hi = self.absolute_position_min, self.absolute_position_max
            if lo is None or hi is None or len(lo) != 3 or len(hi) != 3:
                raise ValueError("Absolute RAY0 requires three-axis position bounds")
            bounds = torch.tensor([lo, hi], dtype=torch.float64)
            if not torch.isfinite(bounds).all() or not (bounds[1] > bounds[0]).all():
                raise ValueError("Absolute RAY0 position bounds must be finite and ordered")
        expected_width = (
            2 * self.tile_width if self.duplicate_horizontal else self.tile_width
        )
        if self.image_height != self.tile_height or self.image_width != expected_width:
            raise ValueError(
                "LIBERO Rothko image/tile mismatch: "
                f"image={(self.image_height, self.image_width)} "
                f"tile={(self.tile_height, self.tile_width)} "
                f"duplicate_horizontal={self.duplicate_horizontal}."
            )
        if self.focal <= 0 or self.center_scale <= 0 or self.dir_scale <= 0:
            raise ValueError("focal, center_scale, and dir_scale must be positive.")
        if not 0.0 < self.center_frac < 1.0:
            raise ValueError(f"center_frac must be in (0,1), got {self.center_frac}.")
        if min(self.boundary_margin, self.outer_margin) < 0:
            raise ValueError("boundary_margin and outer_margin must be non-negative.")
        if 2 * self.outer_margin >= min(self.tile_height, self.tile_width):
            raise ValueError("outer_margin is too large for the LIBERO Rothko tile.")


class LiberoRothkoCodec:
    environment = "libero"
    representation = "libero_rothko"
    pose_dim = 7
    gripper_dim = 1
    num_arms = 1
    layout = "single_arm_duplicated_horizontal"

    def __init__(
        self,
        config: LiberoRothkoCodecConfig | None = None,
        norm_stats: RothkoNormStats | str | Path | None = None,
        expected_action_horizon: int | None = None,
        decode_mode: str = ROTHKO_DECODE_MODE_LEGACY,
        decode_anchor_alpha: float = 0.0,
        decode_block_grid: int = 4,
    ):
        self.config = config or LiberoRothkoCodecConfig()
        self.config.validate()
        self.decode_mode = _validate_decode_mode(decode_mode)
        self.decode_anchor_alpha = _validate_decode_anchor_alpha(
            decode_anchor_alpha
        )
        self.decode_block_grid = _validate_decode_block_grid(decode_block_grid)
        if self.config.frame0_pose_mode == "absolute" and (
            self.decode_mode != ROTHKO_DECODE_MODE_LEGACY or self.decode_anchor_alpha != 0
        ):
            raise ValueError("Absolute RAY0 currently requires legacy decoding and anchor_alpha=0")
        self._decode_anchor_raw: torch.Tensor | None = None
        self.expected_action_horizon = (
            None if expected_action_horizon is None else int(expected_action_horizon)
        )
        if self.expected_action_horizon is not None and self.expected_action_horizon <= 0:
            raise ValueError(
                "expected_action_horizon must be positive, got "
                f"{self.expected_action_horizon}."
            )
        if isinstance(norm_stats, (str, Path)):
            norm_stats = RothkoNormStats.load(norm_stats)
        self.norm_stats = norm_stats
        if self.norm_stats is not None:
            expected_shape = (1, 3, self.config.image_height, self.config.image_width)
            if tuple(self.norm_stats.lo.shape) != expected_shape:
                raise ValueError(
                    "LIBERO Rothko stats tensor shape mismatch: "
                    f"stats={tuple(self.norm_stats.lo.shape)} codec={expected_shape}."
                )
            self._validate_stats_metadata(self.norm_stats.metadata)

    def set_decode_anchor_video(self, normalized_video: torch.Tensor) -> None:
        """Cache a VAE-reconstructed zero-motion template for pose anchoring."""
        if normalized_video.ndim != 5 or normalized_video.shape[0] != 1:
            raise ValueError(
                "LIBERO Rothko anchor video must be [1,3,T,H,W], got "
                f"{tuple(normalized_video.shape)}."
            )
        if normalized_video.shape[1] != 3 or normalized_video.shape[-2:] != (
            self.config.image_height,
            self.config.image_width,
        ):
            raise ValueError(
                "LIBERO Rothko anchor video channel/spatial shape mismatch: "
                f"{tuple(normalized_video.shape)}."
            )
        normalized = normalized_video.permute(0, 2, 1, 3, 4).contiguous()
        self._decode_anchor_raw = self.denormalize(normalized).detach()

    def metadata(self) -> dict[str, Any]:
        config_metadata = asdict(self.config)
        # Preserve old metadata/cache fingerprints exactly for legacy configs.
        if self.config.frame0_pose_mode == "relative":
            for key in ("frame0_pose_mode", "absolute_position_min", "absolute_position_max"):
                config_metadata.pop(key)
        else:
            config_metadata["absolute_position_min"] = list(self.config.absolute_position_min)
            config_metadata["absolute_position_max"] = list(self.config.absolute_position_max)
        return {
            "environment": self.environment,
            "representation": "rothko",
            "raymap_representation": self.representation,
            "layout": self.layout,
            "pose_dim": self.pose_dim,
            "gripper_dim": self.gripper_dim,
            "num_arms": self.num_arms,
            **config_metadata,
            "gripper_encoding": "normalized_outer_border_2g_minus_1",
            "quaternion_order": "wxyz",
        }

    def _validate_stats_metadata(self, metadata: dict[str, Any]) -> None:
        expected: dict[str, Any] = {
            "environment": self.environment,
            "representation": "rothko",
            "raymap_representation": self.representation,
            "layout": self.layout,
            "image_size": [self.config.image_height, self.config.image_width],
            "tile_size": [self.config.tile_height, self.config.tile_width],
            "focal": self.config.focal,
            "center_scale": self.config.center_scale,
            "dir_scale": self.config.dir_scale,
            "center_frac": self.config.center_frac,
            "boundary_margin": self.config.boundary_margin,
            "outer_margin": self.config.outer_margin,
        }
        if self.expected_action_horizon is not None:
            expected["action_horizon"] = self.expected_action_horizon
            expected["pixel_frames"] = self.expected_action_horizon + 1
        strict = int(metadata.get("stats_format_version", 1)) >= 2
        for key, value in expected.items():
            if key not in metadata:
                if strict:
                    raise ValueError(
                        "LIBERO Rothko stats v2 metadata is missing required key "
                        f"{key!r}."
                    )
                continue
            actual = metadata[key]
            if isinstance(value, float):
                matches = abs(float(actual) - value) <= 1e-8
            else:
                matches = actual == value
            if not matches:
                raise ValueError(
                    f"LIBERO Rothko stats metadata mismatch for {key}: "
                    f"stats={actual!r} codec={value!r}"
                )

    @staticmethod
    def _ensure_batched_pose(pose: torch.Tensor) -> tuple[torch.Tensor, bool]:
        squeeze = pose.ndim == 2
        if squeeze:
            pose = pose.unsqueeze(0)
        if pose.ndim != 3 or pose.shape[-1] != 7:
            raise ValueError(f"Expected pose [B,T,7] or [T,7], got {tuple(pose.shape)}")
        return pose, squeeze

    def _canonical_directions(
        self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        cfg = self.config
        dx, dy = 1.0 / cfg.tile_width, 1.0 / cfg.tile_height
        y, x = torch.meshgrid(
            torch.linspace(
                1 - dy,
                -(1 - dy),
                cfg.tile_height,
                device=device,
                dtype=dtype,
            ),
            torch.linspace(
                1 - dx,
                -(1 - dx),
                cfg.tile_width,
                device=device,
                dtype=dtype,
            ),
            indexing="ij",
        )
        directions = torch.stack(
            (x / cfg.focal, y / cfg.focal, torch.ones_like(x)), dim=-1
        )
        return directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def _encode_tile_raw(self, pose7: torch.Tensor) -> torch.Tensor:
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
        output = torch.einsum(
            "btij,hwj->btihw", relative_rotation, canonical
        ) * cfg.dir_scale
        center, _, _ = _center_and_read_masks(
            cfg.tile_height,
            cfg.tile_width,
            center_frac=cfg.center_frac,
            boundary_margin=0,
            outer_margin=0,
            device=pose7.device,
        )
        output[..., center] = (
            relative_position * cfg.center_scale
        )[..., :, None]
        if cfg.frame0_pose_mode == "absolute":
            output[:, 0] = torch.einsum(
                "bij,hwj->bihw", rotation[:, 0], canonical
            ) * cfg.dir_scale
            output[:, 0, :, center] = (position[:, 0] * cfg.center_scale)[..., None]
        return output

    def encode_raw(self, pose7: torch.Tensor) -> torch.Tensor:
        pose7, squeeze = self._ensure_batched_pose(pose7)
        tile = self._encode_tile_raw(pose7)
        output = (
            torch.cat((tile, tile), dim=-1)
            if self.config.duplicate_horizontal
            else tile
        )
        return output.squeeze(0) if squeeze else output

    def _expanded_stats(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.norm_stats is None:
            raise ValueError(
                "LIBERO Rothko normalization stats are required for encode/decode."
            )
        lo = self.norm_stats.lo.to(device=device, dtype=dtype)
        hi = self.norm_stats.hi.to(device=device, dtype=dtype)
        if lo.ndim != 4 or hi.shape != lo.shape or lo.shape[:2] != (1, 3):
            raise ValueError(
                f"Expected LIBERO Rothko stats [1,3,H,W], got {lo.shape} and {hi.shape}"
            )
        if lo.shape[-2:] == (self.config.tile_height, self.config.tile_width):
            if not self.config.duplicate_horizontal:
                pass
            else:
                lo = torch.cat((lo, lo), dim=-1)
                hi = torch.cat((hi, hi), dim=-1)
        if lo.shape[-2:] != (self.config.image_height, self.config.image_width):
            raise ValueError(
                "LIBERO Rothko stats spatial shape mismatch: "
                f"{tuple(lo.shape[-2:])} vs "
                f"{(self.config.image_height, self.config.image_width)}"
            )
        return lo, hi

    def normalize_raw(self, raw: torch.Tensor) -> torch.Tensor:
        squeeze = raw.ndim == 4
        if squeeze:
            raw = raw.unsqueeze(0)
        expected = (
            3,
            self.config.image_height,
            self.config.image_width,
        )
        if raw.ndim != 5 or raw.shape[2:] != expected:
            raise ValueError(f"Expected raw LIBERO Rothko [B,T,3,H,W], got {raw.shape}")
        lo, hi = self._expanded_stats(device=raw.device, dtype=raw.dtype)
        normalized = (
            2.0
            * (raw - lo.unsqueeze(1))
            / (hi - lo).clamp_min(1e-6).unsqueeze(1)
            - 1.0
        ).clamp(-1.0, 1.0)
        if self.config.frame0_pose_mode == "absolute":
            abs_lo, abs_hi = self._absolute_frame_bounds(raw)
            normalized[:, 0] = (
                2 * (raw[:, 0] - abs_lo) / (abs_hi - abs_lo) - 1
            ).clamp(-1, 1)
        return normalized.squeeze(0) if squeeze else normalized

    def _absolute_frame_bounds(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        lo = tensor.new_full((1, 3, cfg.tile_height, cfg.tile_width), -cfg.dir_scale)
        hi = torch.full_like(lo, cfg.dir_scale)
        center, _, _ = _center_and_read_masks(
            cfg.tile_height, cfg.tile_width, center_frac=cfg.center_frac,
            boundary_margin=0, outer_margin=0, device=tensor.device,
        )
        lo[..., center] = tensor.new_tensor(cfg.absolute_position_min)[None, :, None] * cfg.center_scale
        hi[..., center] = tensor.new_tensor(cfg.absolute_position_max)[None, :, None] * cfg.center_scale
        if cfg.duplicate_horizontal:
            lo, hi = torch.cat((lo, lo), -1), torch.cat((hi, hi), -1)
        return lo, hi

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        squeeze = normalized.ndim == 4
        if squeeze:
            normalized = normalized.unsqueeze(0)
        if normalized.ndim != 5:
            raise ValueError(
                f"Expected normalized LIBERO Rothko [B,T,3,H,W], got {normalized.shape}"
            )
        lo, hi = self._expanded_stats(device=normalized.device, dtype=normalized.dtype)
        raw = (normalized + 1.0) * 0.5 * (hi - lo).unsqueeze(1) + lo.unsqueeze(1)
        if self.config.frame0_pose_mode == "absolute":
            abs_lo, abs_hi = self._absolute_frame_bounds(normalized)
            raw[:, 0] = (normalized[:, 0] + 1) * 0.5 * (abs_hi - abs_lo) + abs_lo
        return raw.squeeze(0) if squeeze else raw

    def write_gripper(
        self, normalized: torch.Tensor, gripper: torch.Tensor
    ) -> torch.Tensor:
        squeeze = normalized.ndim == 4
        if squeeze:
            normalized = normalized.unsqueeze(0)
            if gripper.ndim == 2:
                gripper = gripper.unsqueeze(0)
        if gripper.ndim != 3 or gripper.shape[-1] != 1:
            raise ValueError(f"Expected gripper [B,T,1], got {tuple(gripper.shape)}")
        if normalized.shape[:2] != gripper.shape[:2]:
            raise ValueError(
                f"Raymap/gripper shape mismatch: {normalized.shape[:2]} vs {gripper.shape[:2]}"
            )
        output = normalized.clone()
        border = _border_mask(
            self.config.tile_height,
            self.config.tile_width,
            self.config.outer_margin,
            normalized.device,
        )
        code = (
            gripper.to(device=normalized.device, dtype=normalized.dtype)
            .clamp(0, 1)
            .mul(2)
            .sub(1)
        )
        x_offsets = (
            (0, self.config.tile_width)
            if self.config.duplicate_horizontal
            else (0,)
        )
        for x_offset in x_offsets:
            tile = output[
                :,
                :,
                :,
                : self.config.tile_height,
                x_offset : x_offset + self.config.tile_width,
            ]
            tile[..., border] = code[..., 0, None, None].expand(
                -1, -1, 3, int(border.sum())
            )
        return output.squeeze(0) if squeeze else output

    def read_gripper(self, normalized: torch.Tensor) -> torch.Tensor:
        squeeze = normalized.ndim == 4
        if squeeze:
            normalized = normalized.unsqueeze(0)
        border = _border_mask(
            self.config.tile_height,
            self.config.tile_width,
            self.config.outer_margin,
            normalized.device,
        )
        observations = []
        x_offsets = (
            (0, self.config.tile_width)
            if self.config.duplicate_horizontal
            else (0,)
        )
        for x_offset in x_offsets:
            tile = normalized[
                :,
                :,
                :,
                : self.config.tile_height,
                x_offset : x_offset + self.config.tile_width,
            ]
            observations.append(tile[..., border].reshape(*tile.shape[:2], -1))
        code = torch.cat(observations, dim=-1).median(dim=-1).values
        gripper = ((code + 1.0) * 0.5).clamp(0, 1).unsqueeze(-1)
        return gripper.squeeze(0) if squeeze else gripper

    def encode(self, pose7: torch.Tensor, gripper: torch.Tensor) -> torch.Tensor:
        """Return normalized VAE input in ``[B,3,T,224,448]`` format."""
        pose7, squeeze = self._ensure_batched_pose(pose7)
        if gripper.ndim == 2:
            gripper = gripper.unsqueeze(0)
        normalized = self.normalize_raw(self.encode_raw(pose7))
        normalized = self.write_gripper(normalized, gripper)
        video = normalized.permute(0, 2, 1, 3, 4).contiguous()
        return video.squeeze(0) if squeeze else video

    def _decode_tile_raw(
        self, tile: torch.Tensor, base_pose7: torch.Tensor
    ) -> torch.Tensor:
        cfg = self.config
        _, origin_mask, direction_mask = _center_and_read_masks(
            cfg.tile_height,
            cfg.tile_width,
            center_frac=cfg.center_frac,
            boundary_margin=cfg.boundary_margin,
            outer_margin=cfg.outer_margin,
            device=tile.device,
        )
        values = tile.permute(0, 1, 3, 4, 2).to(torch.float64)
        origin_code = values[:, :, origin_mask].median(dim=2).values
        relative_position = (origin_code - origin_code[:, :1]) / cfg.center_scale

        direction = values[:, :, direction_mask]
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        reference = direction[:, 0]
        correlation = (
            torch.einsum("btpi,bpj->btij", direction, reference)
            / direction.shape[2]
        )
        u, _, vh = torch.linalg.svd(correlation)
        determinant = torch.linalg.det(u @ vh)
        correction = torch.eye(
            3, dtype=torch.float64, device=tile.device
        ).repeat(tile.shape[0], tile.shape[1], 1, 1)
        correction[..., 2, 2] = torch.where(determinant >= 0, 1.0, -1.0)
        relative_rotation = u @ correction @ vh
        relative_position[:, 0] = 0
        relative_rotation[:, 0] = torch.eye(
            3, dtype=torch.float64, device=tile.device
        )

        base_position = base_pose7[:, None, :3].to(torch.float64)
        base_rotation = quaternion_wxyz_to_matrix(
            base_pose7[:, 3:7].to(torch.float64)
        )[:, None]
        absolute_position = base_position + torch.einsum(
            "btij,btj->bti", base_rotation, relative_position
        )
        absolute_rotation = base_rotation @ relative_rotation
        return torch.cat(
            (absolute_position, matrix_to_quaternion_wxyz(absolute_rotation)),
            dim=-1,
        ).to(tile.dtype)

    def decode(
        self, video: torch.Tensor, current_pose7: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode normalized raymaps into absolute pose7 targets and gripper."""
        squeeze = video.ndim == 4
        if squeeze:
            video = video.unsqueeze(0)
        if (
            video.ndim != 5
            or video.shape[1] != 3
            or video.shape[-2:]
            != (self.config.image_height, self.config.image_width)
        ):
            raise ValueError(
                f"Expected LIBERO Rothko [B,3,T,H,W], got {tuple(video.shape)}"
            )
        normalized = video.permute(0, 2, 1, 3, 4).contiguous()
        gripper = self.read_gripper(normalized)
        raw = self.denormalize(normalized)
        if self.config.frame0_pose_mode == "absolute":
            # Future maps still express local displacements/rotations. Use the
            # ideal 0/I reference, NOT the absolute pose in decoded RAY0.
            cfg = self.config
            reference = self._canonical_directions(raw.device, raw.dtype).permute(2, 0, 1) * cfg.dir_scale
            center, _, _ = _center_and_read_masks(
                cfg.tile_height, cfg.tile_width, center_frac=cfg.center_frac,
                boundary_margin=0, outer_margin=0, device=raw.device,
            )
            reference[:, center] = 0
            if cfg.duplicate_horizontal:
                reference = torch.cat((reference, reference), -1)
            raw[:, 0] = reference
        left = raw[..., : self.config.tile_width]
        if (
            self.decode_mode == ROTHKO_DECODE_MODE_LEGACY
            and self.config.duplicate_horizontal
        ):
            right = raw[
                ...,
                self.config.tile_width : 2 * self.config.tile_width,
            ]
            tile = (left + right) * 0.5
        elif self.decode_mode == ROTHKO_DECODE_MODE_LEGACY:
            tile = left

        if current_pose7.ndim == 1:
            current_pose7 = current_pose7.unsqueeze(0)
        if current_pose7.shape != (video.shape[0], 7):
            raise ValueError(
                f"Expected current_pose7 {(video.shape[0], 7)}, got {current_pose7.shape}"
            )
        if self.decode_mode == ROTHKO_DECODE_MODE_LEGACY:
            pose = self._decode_tile_raw(tile, current_pose7)
        else:
            tiles = [left]
            if self.config.duplicate_horizontal:
                tiles.append(
                    raw[
                        ...,
                        self.config.tile_width : 2 * self.config.tile_width,
                    ]
                )
            cfg = self.config
            decoder = {
                ROTHKO_DECODE_MODE_BLOCK_POSITION_JOINT_ROTATION: _decode_pose_tiles_block_position_joint_rotation,
                ROTHKO_DECODE_MODE_ROBUST_BLOCK_WEIGHTED_JOINT: _decode_pose_tiles_robust_block_weighted_joint,
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
            if self._decode_anchor_raw is not None:
                anchor_raw = self._decode_anchor_raw.to(
                    device=raw.device, dtype=raw.dtype
                )
                anchor_tiles = [anchor_raw[..., : self.config.tile_width]]
                if self.config.duplicate_horizontal:
                    anchor_tiles.append(
                        anchor_raw[
                            ...,
                            self.config.tile_width : 2 * self.config.tile_width,
                        ]
                    )
                decoder_kwargs["anchor_tiles"] = torch.stack(
                    anchor_tiles, dim=2
                )
            if self.decode_mode in (ROTHKO_DECODE_MODE_ROBUST_BLOCK_CONSENSUS, ROTHKO_DECODE_MODE_BLOCK_POSITION_JOINT_ROTATION, ROTHKO_DECODE_MODE_ROBUST_BLOCK_WEIGHTED_JOINT):
                decoder_kwargs["block_grid"] = self.decode_block_grid
            pose = decoder(
                torch.stack(tiles, dim=2),
                current_pose7,
                **decoder_kwargs,
            )
        if squeeze:
            return pose.squeeze(0), gripper.squeeze(0)
        return pose, gripper
