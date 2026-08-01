import math
import unittest

import torch

from fastwam.representations.libero_osc import axis_angle_to_quaternion_wxyz
from fastwam.representations.libero_rothko import (
    LiberoRothkoCodec,
    LiberoRothkoCodecConfig,
)
from fastwam.representations.rothko import RothkoNormStats, quaternion_wxyz_to_matrix


class LiberoRothkoCodecTest(unittest.TestCase):
    def _codec(self) -> LiberoRothkoCodec:
        config = LiberoRothkoCodecConfig()
        lo = torch.full((1, 3, config.image_height, config.image_width), -1.0)
        hi = torch.full_like(lo, 1.0)
        # Translation values live in the center; broad synthetic bounds avoid
        # clipping in this geometry-only test.
        return LiberoRothkoCodec(
            config,
            RothkoNormStats(
                lo=lo,
                hi=hi,
                metadata={
                    "environment": "libero",
                    "layout": "single_arm_duplicated_horizontal",
                },
            ),
        )

    def test_geometry_and_gripper_roundtrip(self) -> None:
        codec = self._codec()
        time = 17
        pose = torch.zeros(time, 7, dtype=torch.float32)
        pose[:, :3] = torch.tensor([0.1, -0.2, 0.9])
        pose[:, :3] += torch.linspace(0, 0.04, time)[:, None] * torch.tensor(
            [1.0, -0.5, 0.25]
        )
        angles = torch.zeros(time, 3)
        angles[:, 2] = torch.linspace(0, 0.3, time)
        pose[:, 3:] = axis_angle_to_quaternion_wxyz(angles)
        gripper = (torch.arange(time) % 2).float().unsqueeze(-1)

        video = codec.encode(pose, gripper)
        decoded_pose, decoded_gripper = codec.decode(video, pose[0])
        torch.testing.assert_close(decoded_pose[:, :3], pose[:, :3], atol=2e-5, rtol=0)
        expected_rotation = quaternion_wxyz_to_matrix(pose[:, 3:])
        actual_rotation = quaternion_wxyz_to_matrix(decoded_pose[:, 3:])
        torch.testing.assert_close(actual_rotation, expected_rotation, atol=2e-5, rtol=0)
        torch.testing.assert_close(decoded_gripper, gripper, atol=1e-6, rtol=0)

    def test_quaternion_sign_is_equivalent(self) -> None:
        codec = self._codec()
        pose = torch.tensor(
            [
                [0, 0, 0, 1, 0, 0, 0],
                [0.01, 0, 0, 0, 1, 0, 0],
            ],
            dtype=torch.float32,
        )
        gripper = torch.ones(2, 1)
        positive = codec.encode(pose, gripper)
        pose[1, 3:] *= -1
        negative = codec.encode(pose, gripper)
        torch.testing.assert_close(positive, negative)

    def test_stats_horizon_mismatch_is_rejected(self) -> None:
        config = LiberoRothkoCodecConfig()
        lo = torch.full((1, 3, config.image_height, config.image_width), -1.0)
        stats = RothkoNormStats(
            lo=lo,
            hi=-lo,
            metadata={
                "environment": "libero",
                "layout": "single_arm_duplicated_horizontal",
                "action_horizon": 32,
            },
        )
        with self.assertRaisesRegex(ValueError, "action_horizon"):
            LiberoRothkoCodec(
                config,
                stats,
                expected_action_horizon=16,
            )


if __name__ == "__main__":
    unittest.main()
