import unittest

import torch

from fastwam.trainer import _decoded_pose_metrics


class DecodedPoseMetricsTest(unittest.TestCase):
    def test_quaternion_sign_is_ignored_and_units_are_separate(self) -> None:
        target_pose = torch.tensor(
            [[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )
        predicted_pose = target_pose.clone()
        predicted_pose[..., 0] = 0.03
        predicted_pose[..., 3:] *= -1
        target_gripper = torch.ones(1, 1, 1)
        predicted_gripper = torch.tensor([[[0.8]]])

        metrics = _decoded_pose_metrics(
            predicted_pose,
            target_pose,
            predicted_gripper,
            target_gripper,
        )

        self.assertAlmostEqual(metrics["decoded_position_mae_m"], 0.01, places=6)
        self.assertAlmostEqual(metrics["decoded_rotation_geodesic_deg"], 0.0, places=6)
        self.assertAlmostEqual(metrics["decoded_gripper_mae"], 0.2, places=6)
        self.assertEqual(metrics["decoded_gripper_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
