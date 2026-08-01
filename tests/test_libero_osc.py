import math
import unittest

import torch

from fastwam.representations.libero_osc import (
    absolute_target_to_normalized_action,
    axis_angle_to_matrix,
    ee_state_axis_angle_to_pose_wxyz,
    normalized_action_to_absolute_target,
    panda_gripper_qpos_to_open,
)


class LiberoOscTest(unittest.TestCase):
    def test_osc_action_absolute_target_roundtrip(self) -> None:
        generator = torch.Generator().manual_seed(7)
        current = torch.randn(128, 6, generator=generator, dtype=torch.float64)
        current[..., 3:] *= math.pi / current[..., 3:].norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-9)
        pose = ee_state_axis_angle_to_pose_wxyz(current)
        action = torch.empty(128, 6, dtype=torch.float64).uniform_(
            -0.95, 0.95, generator=generator
        )
        target = normalized_action_to_absolute_target(pose, action)
        recovered = absolute_target_to_normalized_action(pose, target)
        torch.testing.assert_close(recovered, action, atol=1e-8, rtol=1e-8)

    def test_absolute_axis_angle_near_pi_is_finite(self) -> None:
        ee_state = torch.tensor(
            [[0.1, -0.2, 0.9, math.pi - 1e-8, 0.0, 0.0]],
            dtype=torch.float64,
        )
        pose = ee_state_axis_angle_to_pose_wxyz(ee_state)
        self.assertTrue(bool(torch.isfinite(pose).all()))
        torch.testing.assert_close(
            axis_angle_to_matrix(ee_state[..., 3:]),
            axis_angle_to_matrix(
                absolute_target_to_normalized_action(pose, pose)[..., 3:] * 0.5
            )
            @ axis_angle_to_matrix(ee_state[..., 3:]),
            atol=1e-8,
            rtol=1e-8,
        )

    def test_panda_gripper_mapping(self) -> None:
        qpos = torch.tensor([[0.0, 0.0], [0.04, -0.04], [0.02, -0.02]])
        expected = torch.tensor([[0.0], [1.0], [0.5]])
        torch.testing.assert_close(panda_gripper_qpos_to_open(qpos), expected)


if __name__ == "__main__":
    unittest.main()
