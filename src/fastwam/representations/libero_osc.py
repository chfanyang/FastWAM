"""LIBERO OSC_POSE conversions used by the Rothko data and deployment paths.

LIBERO stores an absolute end-effector pose in each observation, while its
expert action is a normalized delta command for robosuite's ``OSC_POSE``
controller.  This module is the single source of truth for converting between
those two forms.

All public pose tensors use ``xyz + quaternion(wxyz)``.  LIBERO simulator
observations use ``quaternion(xyzw)`` and must be reordered by the caller.
"""
from __future__ import annotations

import torch

from .rothko import matrix_to_quaternion_wxyz, quaternion_wxyz_to_matrix


LIBERO_POSITION_SCALE_M = 0.05
LIBERO_ROTATION_SCALE_RAD = 0.5
PANDA_FINGER_TRAVEL_M = 0.04


def axis_angle_to_quaternion_wxyz(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert exponential-coordinate rotations to real-first quaternions."""
    if axis_angle.shape[-1] != 3:
        raise ValueError(f"Expected axis-angle [...,3], got {tuple(axis_angle.shape)}")
    angle = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    half = 0.5 * angle
    # sin(x / 2) / x is well behaved at zero.  Use its second-order limit to
    # avoid normalizing a nearly-zero axis.
    scale = torch.where(
        angle > 1e-8,
        torch.sin(half) / angle.clamp_min(1e-12),
        0.5 - angle.square() / 48.0,
    )
    quaternion = torch.cat((torch.cos(half), axis_angle * scale), dim=-1)
    return quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def quaternion_wxyz_to_axis_angle(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert real-first quaternions to shortest-path axis-angle vectors."""
    if quaternion.shape[-1] != 4:
        raise ValueError(f"Expected quaternion [...,4], got {tuple(quaternion.shape)}")
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    # q and -q represent the same rotation.  Choosing w >= 0 produces the
    # shortest rotation and is stable for controller deltas.
    quaternion = torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)
    vector = quaternion[..., 1:]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, quaternion[..., :1].clamp_min(0.0))
    scale = torch.where(
        vector_norm > 1e-8,
        angle / vector_norm.clamp_min(1e-12),
        2.0 + vector_norm.square() / 3.0,
    )
    return vector * scale


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    return quaternion_wxyz_to_matrix(axis_angle_to_quaternion_wxyz(axis_angle))


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    return quaternion_wxyz_to_axis_angle(matrix_to_quaternion_wxyz(matrix))


def ee_state_axis_angle_to_pose_wxyz(ee_state: torch.Tensor) -> torch.Tensor:
    """Convert LIBERO ``xyz + absolute axis-angle`` observations to pose7."""
    if ee_state.shape[-1] != 6:
        raise ValueError(f"Expected EE state [...,6], got {tuple(ee_state.shape)}")
    return torch.cat(
        (ee_state[..., :3], axis_angle_to_quaternion_wxyz(ee_state[..., 3:6])),
        dim=-1,
    )


def normalized_action_to_absolute_target(
    current_pose_wxyz: torch.Tensor,
    normalized_action: torch.Tensor,
    *,
    position_scale: float = LIBERO_POSITION_SCALE_M,
    rotation_scale: float = LIBERO_ROTATION_SCALE_RAD,
) -> torch.Tensor:
    """Apply a normalized LIBERO OSC delta to an absolute pose.

    Rotation follows robosuite ``set_goal_orientation`` exactly:
    ``R_target = R_delta @ R_current``.
    """
    if current_pose_wxyz.shape[-1] != 7:
        raise ValueError(
            f"Expected current pose [...,7], got {tuple(current_pose_wxyz.shape)}"
        )
    if normalized_action.shape[-1] < 6:
        raise ValueError(
            f"Expected normalized action [...,>=6], got {tuple(normalized_action.shape)}"
        )
    current_pose_wxyz, normalized_action = torch.broadcast_tensors(
        current_pose_wxyz,
        normalized_action[..., :7]
        if normalized_action.shape[-1] == 7
        else torch.nn.functional.pad(normalized_action[..., :6], (0, 1)),
    )
    target_position = (
        current_pose_wxyz[..., :3]
        + float(position_scale) * normalized_action[..., :3]
    )
    current_rotation = quaternion_wxyz_to_matrix(current_pose_wxyz[..., 3:7])
    delta_rotation = axis_angle_to_matrix(
        float(rotation_scale) * normalized_action[..., 3:6]
    )
    target_rotation = delta_rotation @ current_rotation
    target_quaternion = matrix_to_quaternion_wxyz(target_rotation)
    return torch.cat((target_position, target_quaternion), dim=-1)

def absolute_target_to_normalized_action(
    current_pose_wxyz: torch.Tensor,
    target_pose_wxyz: torch.Tensor,
    *,
    position_scale: float = LIBERO_POSITION_SCALE_M,
    rotation_scale: float = LIBERO_ROTATION_SCALE_RAD,
    clip: bool = False,
) -> torch.Tensor:
    """Convert an absolute OSC target into LIBERO's normalized 6D command."""
    if current_pose_wxyz.shape[-1] != 7 or target_pose_wxyz.shape[-1] != 7:
        raise ValueError(
            "Expected current and target poses [...,7], got "
            f"{tuple(current_pose_wxyz.shape)} and {tuple(target_pose_wxyz.shape)}"
        )
    current_pose_wxyz, target_pose_wxyz = torch.broadcast_tensors(
        current_pose_wxyz, target_pose_wxyz
    )
    position_action = (
        target_pose_wxyz[..., :3] - current_pose_wxyz[..., :3]
    ) / float(position_scale)
    current_rotation = quaternion_wxyz_to_matrix(current_pose_wxyz[..., 3:7])
    target_rotation = quaternion_wxyz_to_matrix(target_pose_wxyz[..., 3:7])
    delta_rotation = target_rotation @ current_rotation.transpose(-1, -2)
    rotation_action = matrix_to_axis_angle(delta_rotation) / float(rotation_scale)
    action = torch.cat((position_action, rotation_action), dim=-1)
    return action.clamp(-1.0, 1.0) if clip else action


def panda_gripper_qpos_to_open(
    gripper_qpos: torch.Tensor,
    *,
    finger_travel: float = PANDA_FINGER_TRAVEL_M,
) -> torch.Tensor:
    """Map mirrored Panda finger joints to ``0=closed, 1=open``.

    The LIBERO recordings use a positive left finger and a negative right
    finger.  Their separation is therefore ``q_left - q_right`` and the fully
    open separation is ``2 * finger_travel``.
    """
    if gripper_qpos.shape[-1] != 2:
        raise ValueError(
            f"Expected Panda gripper qpos [...,2], got {tuple(gripper_qpos.shape)}"
        )
    opening = (
        gripper_qpos[..., 0] - gripper_qpos[..., 1]
    ) / (2.0 * float(finger_travel))
    return opening.clamp(0.0, 1.0).unsqueeze(-1)
