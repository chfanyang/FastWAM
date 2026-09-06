import unittest

import torch

from fastwam.datasets.lerobot.robot_video_dataset import (
    resolve_libero_future_gripper,
)


class LiberoPlusDatasetAdapterTest(unittest.TestCase):
    def test_legacy_libero_path_is_unchanged(self) -> None:
        raw_action = {
            "default": torch.tensor(
                [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0],
                 [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]]
            )
        }
        actual = resolve_libero_future_gripper(
            raw_action, action_horizon=2, explicit_key=None
        )
        torch.testing.assert_close(actual, torch.tensor([[0.0], [1.0]]))

    def test_plus_uses_explicit_converted_side_channel(self) -> None:
        raw_action = {
            "default": torch.tensor(
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
            ),
            "gripper_open": torch.tensor([[1.0], [0.0]]),
        }
        actual = resolve_libero_future_gripper(
            raw_action, action_horizon=2, explicit_key="gripper_open"
        )
        torch.testing.assert_close(actual, torch.tensor([[1.0], [0.0]]))

    def test_plus_rejects_environment_convention_in_explicit_field(self) -> None:
        with self.assertRaisesRegex(ValueError, "0=closed,1=open"):
            resolve_libero_future_gripper(
                {
                    "default": torch.zeros(2, 7),
                    "gripper_open": torch.tensor([[-1.0], [1.0]]),
                },
                action_horizon=2,
                explicit_key="gripper_open",
            )


if __name__ == "__main__":
    unittest.main()
