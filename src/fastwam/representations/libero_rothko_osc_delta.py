"""Absolute world-state RAY0 followed by original LIBERO OSC commands.

Future pose7 tensors are [scaled world-axis delta_xyz, delta_rotation_wxyz],
NOT world targets or displacements relative to the chunk's starting pose.
"""
from dataclasses import replace
import torch
from .libero_rothko import LiberoRothkoCodec
from .libero_rothko_all_absolute import LiberoAllAbsoluteRothkoCodec
from .libero_osc import (
    LIBERO_POSITION_SCALE_M, LIBERO_ROTATION_SCALE_RAD,
    axis_angle_to_quaternion_wxyz, matrix_to_axis_angle,
)
from .rothko import quaternion_wxyz_to_matrix

REPRESENTATION = 'libero_rothko_osc_delta'


def action_to_delta_pose(action):
    if action.shape[-1] != 7 or not torch.isfinite(action).all():
        raise ValueError('Expected finite original LIBERO action [...,7]')
    if (action[..., :6].abs() > 1 + 1e-6).any():
        raise ValueError('OSC action exceeds controller input limits')
    return torch.cat((action[..., :3] * LIBERO_POSITION_SCALE_M,
        axis_angle_to_quaternion_wxyz(action[..., 3:6] * LIBERO_ROTATION_SCALE_RAD)), -1)


def delta_pose_to_action_motion(pose, *, clip=True):
    if pose.shape[-1] != 7 or not torch.isfinite(pose).all():
        raise ValueError('Expected finite delta pose [...,7]')
    xyz = pose[..., :3] / LIBERO_POSITION_SCALE_M
    rot = matrix_to_axis_angle(quaternion_wxyz_to_matrix(pose[..., 3:])) / LIBERO_ROTATION_SCALE_RAD
    motion = torch.cat((xyz, rot), -1)
    return motion.clamp(-1, 1) if clip else motion


class LiberoOSCDeltaRothkoCodec(LiberoRothkoCodec):
    representation = REPRESENTATION
    # Reuse independent-frame geometry, never the chunk-relative transform.
    _encode_tile_raw = LiberoAllAbsoluteRothkoCodec._encode_tile_raw
    decode = LiberoAllAbsoluteRothkoCodec.decode

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.frame0_pose_mode != 'absolute':
            raise ValueError('OSC codec requires absolute world-state RAY0')
        if self.decode_mode != 'legacy' or self.decode_anchor_alpha != 0:
            raise ValueError('OSC codec requires legacy/anchor0')

    def metadata(self):
        return {**super().metadata(), 'pose_mode': 'absolute_state_osc_commands',
                'state_coordinate_frame': 'world', 'future_pose_mode': 'osc_delta',
                'future_translation_axes': 'world', 'position_bounds_scope': 'ray0_only',
                'osc_position_scale_m': LIBERO_POSITION_SCALE_M,
                'osc_rotation_scale_rad': LIBERO_ROTATION_SCALE_RAD,
                'future_position_bounds_m': [-LIBERO_POSITION_SCALE_M, LIBERO_POSITION_SCALE_M],
                'gripper_semantics': '0=closed,1=open; env=1-2*open'}

    def delta_bounds(self, tensor):
        cfg = replace(self.config, absolute_position_min=(-LIBERO_POSITION_SCALE_M,)*3,
                      absolute_position_max=(LIBERO_POSITION_SCALE_M,)*3)
        return LiberoRothkoCodec(cfg)._absolute_frame_bounds(tensor)

    def _validate_stats_metadata(self, metadata):
        super()._validate_stats_metadata(metadata)
        expected=self.metadata()
        for key in ('pose_mode','state_coordinate_frame','future_pose_mode',
                    'future_translation_axes','position_bounds_scope','osc_position_scale_m',
                    'osc_rotation_scale_rad','future_position_bounds_m','gripper_semantics',
                    'frame0_pose_mode','absolute_position_min','absolute_position_max'):
            if metadata.get(key) != expected[key]:
                raise ValueError(f'OSC stats identity mismatch: {key}')
        lo,hi=self.delta_bounds(self.norm_stats.lo)
        if not torch.equal(lo,self.norm_stats.lo) or not torch.equal(hi,self.norm_stats.hi):
            raise ValueError('OSC stats tensors do not match controller delta bounds')
