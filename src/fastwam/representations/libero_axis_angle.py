"""Opt-in constant axis-angle ring; legacy LIBERO codecs are untouched.

Preserve the baseline's normalized translation/gripper pixels exactly. Replace
only the rotation ring with a shortest rotation vector divided by pi. RAY0 is
absolute; future rotations and translations are relative to the observed EE.
"""
from __future__ import annotations

import math
import torch

from .libero_osc import axis_angle_to_quaternion_wxyz, quaternion_wxyz_to_axis_angle
from .rothko import (
    _border_mask, _center_and_read_masks,
    quaternion_wxyz_to_matrix,
)


def _multiply_wxyz(a, b):
    """Quaternion product with broadcasting; avoids matrix conversion at pi."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    q = torch.stack((aw*bw-ax*bx-ay*by-az*bz,
                     aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx,
                     aw*bz+ax*by-ay*bx+az*bw), dim=-1)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class LiberoAxisAngleCodec:
    representation = 'libero_axis_angle_constant'

    def __init__(self, base):
        cfg = base.config
        if base.representation != 'libero_rothko':
            raise ValueError('Wrap only the baseline absolute-RAY0 / relative-future LIBERO codec.')
        if (cfg.tile_height, cfg.tile_width, cfg.image_height, cfg.image_width) != (224, 224, 224, 448):
            raise ValueError('This ablation requires two 224x224 LIBERO tiles.')
        if not cfg.duplicate_horizontal or cfg.frame0_pose_mode != 'absolute':
            raise ValueError('This ablation requires absolute RAY0 and duplicated horizontal tiles.')
        if base.decode_mode != 'legacy' or base.decode_anchor_alpha != 0:
            raise ValueError('Constant axis-angle uses its own median reader and observed EE reference; require legacy/anchor0.')
        self.base = base

    def __getattr__(self, name):
        if name == 'base':
            raise AttributeError(name)
        return getattr(self.base, name)

    def metadata(self):
        return dict(self.base.metadata(),
            raymap_representation=self.representation,
            rotation_encoding='shortest_rotation_vector_constant_ring_div_pi_v1',
            rotation_scale_rad=math.pi,
            rotation_conversion='normalized_quaternion_products',
            rotation_write_region='outside_center_and_gripper_border',
            rotation_read_region='baseline_safe_direction_mask',
            future_rotation_frame='observed_current_ee',
            rotation_decoder='mean_corresponding_tiles_then_componentwise_spatial_median',
            rotation_decoder_reference='observed_current_ee_not_decoded_ray0')

    def masks(self, device):
        c = self.config
        center, origin, direction = _center_and_read_masks(
            c.tile_height, c.tile_width, center_frac=c.center_frac,
            boundary_margin=c.boundary_margin, outer_margin=c.outer_margin,
            device=device)
        border = _border_mask(c.tile_height, c.tile_width, c.outer_margin, device)
        return center, origin, direction, border

    def rotation_vectors(self, pose):
        """[B,T,3] radians; frame zero absolute, later frames chunk-relative."""
        if not torch.isfinite(pose).all() or (pose[..., 3:].norm(dim=-1) < 1e-12).any():
            raise ValueError('Pose must be finite with nonzero quaternions.')
        # Work directly with quaternions, including an exact-pi observed pose.
        # Keep baseline translation pixels in their original dtype via encode.
        q = pose[..., 3:].double()
        q = q / q.norm(dim=-1, keepdim=True)
        inverse = q[:, :1].clone(); inverse[..., 1:] *= -1
        relative = _multiply_wxyz(inverse, q)
        relative[:, 0] = q[:, 0]
        first = (relative[..., 1:] != 0).long().argmax(-1, keepdim=True)
        negative_axis = relative[..., 1:].gather(-1, first)[..., 0] < 0
        flip = (relative[..., 0] < 0) | ((relative[..., 0] == 0) & negative_axis)
        relative = torch.where(flip[..., None], -relative, relative)
        return quaternion_wxyz_to_axis_angle(relative)

    def encode(self, pose, gripper):
        pose, squeeze = self.base._ensure_batched_pose(pose)
        if gripper.ndim == 2:
            gripper = gripper.unsqueeze(0)
        if not torch.isfinite(gripper).all():
            raise ValueError('Gripper must be finite.')
        output = self.base.encode(pose, gripper).clone()
        center, _, _, border = self.masks(output.device)
        ring = ~center & ~border
        code = (self.rotation_vectors(pose) / math.pi).to(output).clamp(-1, 1)
        for tile in output.split(self.config.tile_width, dim=-1):
            tile[..., ring] = code.permute(0, 2, 1)[..., None]
        return output.squeeze(0) if squeeze else output

    def decode(self, video, current_pose7):
        squeeze = video.ndim == 4
        if squeeze:
            video = video.unsqueeze(0)
        if (video.ndim != 5 or video.shape[1] != 3 or video.shape[2] < 1
                or tuple(video.shape[-2:]) != (224, 448)):
            raise ValueError('Expected normalized video [B,3,T,224,448].')
        if current_pose7.ndim == 1:
            current_pose7 = current_pose7.unsqueeze(0)
        if current_pose7.shape != (video.shape[0], 7):
            raise ValueError('Expected one observed pose [B,7] per video.')
        if not torch.isfinite(video).all() or not torch.isfinite(current_pose7).all():
            raise ValueError('Video and observed pose must be finite.')
        if (current_pose7[..., 3:].norm(dim=-1) < 1e-12).any():
            raise ValueError('Observed pose quaternion must be nonzero.')

        normalized = video.permute(0, 2, 1, 3, 4).contiguous()
        gripper = self.base.read_gripper(normalized)
        raw = self.base.denormalize(normalized)
        raw_tile = (raw[..., :224] + raw[..., 224:]) * .5
        tile = (normalized[..., :224] + normalized[..., 224:]) * .5
        _, origin, direction, _ = self.masks(video.device)
        local_position = raw_tile[..., origin].double().median(-1).values / self.config.center_scale
        # Absolute RAY0 is conditioning only. Do not subtract its pixels from
        # future local translations or rotations.
        local_position[:, 0] = 0
        rotation_vector = tile[..., direction].double().median(-1).values * math.pi
        local_q = axis_angle_to_quaternion_wxyz(rotation_vector)
        current = current_pose7.to(device=video.device, dtype=torch.float64)
        r0 = quaternion_wxyz_to_matrix(current[..., 3:])
        position = current[:, None, :3] + (r0[:, None] @ local_position.unsqueeze(-1)).squeeze(-1)
        current_q = current[..., 3:] / current[..., 3:].norm(dim=-1, keepdim=True)
        absolute_q = _multiply_wxyz(current_q[:, None], local_q)
        absolute_q[:, 0] = current_q
        pose = torch.cat((position, absolute_q), dim=-1).to(video.dtype)
        if squeeze:
            return pose.squeeze(0), gripper.squeeze(0)
        return pose, gripper

    def set_decode_anchor_video(self, video):
        raise ValueError('Constant axis-angle decoding uses the observed EE; VAE anchor correction is unsupported.')
