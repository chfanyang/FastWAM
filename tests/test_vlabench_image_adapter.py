import unittest
import torch

from fastwam.datasets.vlabench_image import (
    FIELD_MAP, VLABenchImageWindowDataset, map_vlabench_fields,
    vlabench_gripper_open_fields,
)


class VLABenchImageAdapterTest(unittest.TestCase):
    def test_xyz_euler_pose_conversion(self):
        from fastwam.datasets.vlabench_video import pose_xyz_euler_to_wxyz
        pose = torch.tensor([[1., 2., 3., 0., 0., torch.pi / 2, 0.]])
        result = pose_xyz_euler_to_wxyz(pose)
        self.assertTrue(torch.equal(result[:, :3], pose[:, :3]))
        self.assertTrue(torch.allclose(result[:, 3:], torch.tensor([[2**-.5, 0., 0., 2**-.5]])))

    def test_mapping_is_lossless_and_does_not_mutate_input(self):
        source = {k: torch.randn(3, 7) for k in FIELD_MAP}
        source.update({k + "_is_pad": torch.tensor([False, False, True])
                       for k in FIELD_MAP})
        source["task"] = "instruction"
        result = map_vlabench_fields(source)
        for k, target in FIELD_MAP.items():
            self.assertIs(result[target], source[k])
            self.assertIs(result[target + "_is_pad"], source[k + "_is_pad"])
        self.assertIn("actions", source)
        self.assertEqual(result["task"], "instruction")

    def test_rejects_collision(self):
        with self.assertRaises(ValueError):
            map_vlabench_fields({"actions": 1, "action": 2})

    def test_fastwam_sample_structure_preserves_actions_and_masks(self):
        frames = 17
        sample = {
            "image": torch.ones(frames, 3, 2, 2),
            "wrist_image": torch.zeros(frames, 3, 2, 2),
            "state": torch.randn(frames, 7), "actions": torch.randn(frames - 1, 7),
            "task": "instruction", "episode_index": torch.tensor(3),
            "frame_index": torch.tensor(0), "index": torch.tensor(100),
            "timestamp": torch.tensor(0.),
            "state_is_pad": torch.zeros(frames, dtype=torch.bool),
            "image_is_pad": torch.zeros(frames, dtype=torch.bool),
            "actions_is_pad": torch.zeros(frames - 1, dtype=torch.bool),
        }
        dataset = object.__new__(VLABenchImageWindowDataset)
        sample["state"][:, -1] = 0
        sample["actions"][:, -1] = 1
        dataset.reader = [sample]
        dataset.camera_keys = ("image", "wrist_image")
        out = dataset[0]
        self.assertIs(out["raw_action"]["default"], sample["actions"])
        self.assertIs(out["state"]["default"], sample["state"])
        self.assertIs(out["action_is_pad"], sample["actions_is_pad"])
        self.assertEqual(out["images"]["image"].dtype, torch.uint8)
        self.assertTrue((out["images"]["image"] == 255).all())
        self.assertEqual(set(out["images"]), {"image", "wrist_image"})
        self.assertTrue((out["raw_state"]["gripper_open"] == 1).all())
        self.assertTrue((out["raw_action"]["gripper_open"] == 1).all())
        with self.assertRaises(IndexError):
            dataset[1]

    def test_gripper_polarity_different_horizons_and_no_inplace_changes(self):
        state, action = torch.zeros(3, 7), torch.zeros(2, 7)
        state[:, -1] = torch.tensor([0., 0., 1.])
        action[:, -1] = torch.tensor([1., 0.])
        old_state, old_action = state.clone(), action.clone()
        observed, command = vlabench_gripper_open_fields(state, action)
        torch.testing.assert_close(observed, torch.tensor([[1.], [1.], [0.]]))
        torch.testing.assert_close(command, torch.tensor([[1.], [0.]]))
        observed.zero_()
        command.zero_()
        torch.testing.assert_close(state, old_state)
        torch.testing.assert_close(action, old_action)

    def test_gripper_does_not_silently_clip_wrong_convention(self):
        state, action = torch.zeros(17, 7), torch.zeros(16, 7)
        action[0, -1] = -1
        with self.assertRaisesRegex(ValueError, "gripper must be"):
            vlabench_gripper_open_fields(state, action)


if __name__ == "__main__":
    unittest.main()
