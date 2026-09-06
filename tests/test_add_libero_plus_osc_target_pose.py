import unittest

import numpy as np
import pyarrow as pa
import torch

from fastwam.representations.libero_osc import absolute_target_to_normalized_action
from scripts.add_libero_plus_osc_target_pose import (
    ACTION_GRIPPER_OPEN_KEY,
    EE_POSE_KEY,
    STATE_GRIPPER_OPEN_KEY,
    TARGET_POSE_KEY,
    compute_libero_plus_outputs,
    libero_plus_env_gripper_to_open,
)


class AddLiberoPlusOscTargetPoseTest(unittest.TestCase):
    def test_environment_gripper_semantics_match_existing_training(self) -> None:
        env_command = torch.tensor([[-1.0], [1.0], [0.0]])
        expected_open = torch.tensor([[1.0], [0.0], [0.5]])
        torch.testing.assert_close(
            libero_plus_env_gripper_to_open(env_command), expected_open
        )

    def test_packed_state_and_action_conversion(self) -> None:
        state = np.asarray(
            [
                [0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.04, -0.04],
                [0.2, 0.1, 0.4, 0.2, -0.1, 0.3, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        action = np.asarray(
            [
                [0.5, -0.5, 0.25, 0.1, 0.2, -0.3, -1.0],
                [-0.4, 0.3, 0.2, -0.2, 0.1, 0.4, 1.0],
            ],
            dtype=np.float32,
        )
        table = pa.table(
            {
                "observation.state": pa.FixedSizeListArray.from_arrays(
                    pa.array(state.reshape(-1)), 8
                ),
                "action": pa.FixedSizeListArray.from_arrays(
                    pa.array(action.reshape(-1)), 7
                ),
            }
        )
        outputs, position_error, rotation_error = compute_libero_plus_outputs(table)

        self.assertEqual(outputs[EE_POSE_KEY].shape, (2, 7))
        self.assertEqual(outputs[TARGET_POSE_KEY].shape, (2, 7))
        np.testing.assert_allclose(
            outputs[STATE_GRIPPER_OPEN_KEY], [[1.0], [0.0]], atol=1e-6
        )
        np.testing.assert_allclose(
            outputs[ACTION_GRIPPER_OPEN_KEY], [[1.0], [0.0]], atol=1e-6
        )
        np.testing.assert_allclose(
            outputs[TARGET_POSE_KEY][:, :3],
            state[:, :3] + 0.05 * action[:, :3],
            atol=1e-6,
        )
        recovered = absolute_target_to_normalized_action(
            torch.from_numpy(outputs[EE_POSE_KEY]).to(torch.float64),
            torch.from_numpy(outputs[TARGET_POSE_KEY]).to(torch.float64),
        )
        torch.testing.assert_close(
            recovered,
            torch.from_numpy(action[:, :6]).to(torch.float64),
            atol=2e-6,
            rtol=2e-6,
        )
        self.assertLess(position_error, 1e-8)
        self.assertLess(rotation_error, 1e-8)
        np.testing.assert_allclose(
            np.linalg.norm(outputs[TARGET_POSE_KEY][:, 3:], axis=-1),
            np.ones(2),
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
