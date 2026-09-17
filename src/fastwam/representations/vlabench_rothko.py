"""VLABench identity for the shared single-arm, duplicated-tile geometry.

Distinct metadata prevents loading LIBERO statistics by mistake. No existing
codec or configuration is modified. Inputs must already be poses in one common
coordinate frame, with wxyz quaternions and grippers using 1=open.
"""

from dataclasses import dataclass, replace
import torch
from .libero_rothko import LiberoRothkoCodec, LiberoRothkoCodecConfig
from .rothko import _border_mask, _center_and_read_masks, _decode_pose_tiles_robust_joint


@dataclass(frozen=True)
class VLABenchRothkoCodecConfig(LiberoRothkoCodecConfig):
    horizontal_copies: int = 2

    def validate(self):
        if self.horizontal_copies == 2:
            return super().validate()
        if self.horizontal_copies != 3 or not self.duplicate_horizontal:
            raise ValueError("VLABench supports two legacy tiles or three horizontal tiles")
        if self.image_width != 3 * self.tile_width:
            raise ValueError("Three-tile VLABench requires triple width")
        replace(self, image_width=2*self.tile_width, horizontal_copies=2).validate()


class VLABenchRothkoCodec(LiberoRothkoCodec):
    environment = "vlabench"
    representation = "vlabench_rothko"

    def _absolute_frame_bounds(self, tensor):
        lo, hi = super()._absolute_frame_bounds(tensor)
        if getattr(self.config, "horizontal_copies", 2) == 3:
            w = self.config.tile_width
            lo = torch.cat((lo, lo[..., :w]), -1)
            hi = torch.cat((hi, hi[..., :w]), -1)
        return lo, hi

    def metadata(self):
        result = super().metadata()
        if getattr(self.config, "horizontal_copies", 2) == 2:
            result.pop("horizontal_copies", None)  # Preserve old fingerprints.
        return result

    def encode_raw(self, pose7):
        if getattr(self.config, "horizontal_copies", 2) == 2:
            return super().encode_raw(pose7)
        pose, squeeze = self._ensure_batched_pose(pose7)
        tile = self._encode_tile_raw(pose)
        raw = torch.cat((tile, tile, tile), -1)
        return raw.squeeze(0) if squeeze else raw

    def write_gripper(self, normalized, gripper):
        result = super().write_gripper(normalized, gripper)
        if getattr(self.config, "horizontal_copies", 2) == 3:
            # All three ideal tiles are identical, including the gripper border.
            w = self.config.tile_width
            border = _border_mask(self.config.tile_height, w,
                                  self.config.outer_margin, result.device)
            result[..., 2*w:3*w][..., border] = result[..., :w][..., border]
        return result

    def read_gripper(self, normalized):
        if getattr(self.config, "horizontal_copies", 2) == 2:
            return super().read_gripper(normalized)
        border = _border_mask(self.config.tile_height, self.config.tile_width,
                              self.config.outer_margin, normalized.device)
        values = [tile[..., border].flatten(-2)
                  for tile in normalized.split(self.config.tile_width, -1)]
        return ((torch.cat(values, -1).median(-1).values + 1) / 2).clamp(0, 1).unsqueeze(-1)

    def decode(self, video, current_pose7):
        if getattr(self.config, "horizontal_copies", 2) == 2:
            return super().decode(video, current_pose7)
        if self.decode_mode not in {"legacy", "robust_joint"} or self.decode_anchor_alpha != 0:
            raise ValueError("Three-tile VLABench supports legacy/robust_joint with anchor=0")
        squeeze = video.ndim == 4
        if squeeze:
            video = video.unsqueeze(0)
        if video.shape[1] != 3 or tuple(video.shape[-2:]) != (self.config.image_height, self.config.image_width):
            raise ValueError("Three-tile VLABench video shape mismatch")
        normalized = video.permute(0, 2, 1, 3, 4)
        gripper = self.read_gripper(normalized)
        raw = self.denormalize(normalized)
        if self.config.frame0_pose_mode == "absolute":
            # Absolute RAY0 is a condition; future maps retain local geometry.
            # Decode against ideal 0/I, then compose with the physical EE pose.
            cfg = self.config
            reference = self._canonical_directions(raw.device, raw.dtype).permute(2, 0, 1) * cfg.dir_scale
            center, _, _ = _center_and_read_masks(
                cfg.tile_height, cfg.tile_width, center_frac=cfg.center_frac,
                boundary_margin=0, outer_margin=0, device=raw.device)
            reference[:, center] = 0
            raw[:, 0] = torch.cat((reference, reference, reference), -1)
        if self.decode_mode == "legacy":
            tile = torch.stack(raw.split(self.config.tile_width, -1)).mean(0)
            pose = self._decode_tile_raw(tile, current_pose7.reshape(-1, 7))
        else:
            cfg = self.config
            pose = _decode_pose_tiles_robust_joint(
                torch.stack(raw.split(cfg.tile_width, -1), dim=2),
                current_pose7.reshape(-1, 7), center_frac=cfg.center_frac,
                boundary_margin=cfg.boundary_margin, outer_margin=cfg.outer_margin,
                center_scale=cfg.center_scale,
                canonical_directions=self._canonical_directions(raw.device, torch.float64),
                anchor_alpha=0.)
        return (pose[0], gripper[0]) if squeeze else (pose, gripper)
