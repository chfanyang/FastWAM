"""Exercise deployment normalization and the workstation wire contract on CPU."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image

from fastwam.realworld.baseline_data import ActionCodec
from fastwam.realworld.baseline_policy import OriginalFastWAMPolicy
from fastwam.realworld.protocol import pack, unpack

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'data/realworld_artifacts/ray0_absolute_left_wrist_val2_g08_q9995_v1/split.json'
STATS = ROOT / 'data/realworld_artifacts/original_fastwam_wan22_5b_left_wrist_val2_g08_q9995_v1/action_stats.json'


class PolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.codec = ActionCodec(STATS, MANIFEST)
        self.pose = np.r_[(self.codec.lo+self.codec.hi)/2, np.sqrt(.5), 0, 0, np.sqrt(.5)].astype(np.float32)
        self.actions = torch.zeros(16, 7)
        self.actions[:, 0] = .1
        self.model = SimpleNamespace(action_codec=self.codec,
            infer_action=Mock(return_value={'action': self.actions}),
            infer_joint=Mock(return_value={'action': self.actions,
                'video': [Image.new('RGB', (448, 224)) for _ in range(17)]}))

    def policy(self, directory=None):
        with patch('fastwam.realworld.baseline_policy.load_context',
                   return_value=(torch.zeros(2, 4), torch.ones(2, dtype=torch.bool))):
            return OriginalFastWAMPolicy(self.model, MANIFEST, STATS, 'unused',
                task_name='pick_place', save_rgb_dir=directory)

    def observation(self, policy):
        return dict(protocol_version=1, request_id=0, ee_link='fr3_link8', color_order='RGB',
            prompt=policy.metadata['instructions'][0], ee_pose_wxyz=self.pose, gripper_open=1.,
            images={'left_camera': np.full((224,224,3), 50, np.uint8),
                    'wrist_camera': np.full((224,224,3), 100, np.uint8)},
            record_session='a'*32, record_steps=8)

    def test_normalization_decode_and_client_contract(self):
        policy = self.policy()
        result = unpack(pack(policy.infer(unpack(pack(self.observation(policy))))))
        expected, grip = self.codec.decode(self.pose, self.actions.numpy())
        np.testing.assert_allclose(result['target_pose_wxyz'], expected)
        np.testing.assert_allclose(result['gripper_open'], grip)
        kwargs = self.model.infer_action.call_args.kwargs
        np.testing.assert_allclose(kwargs['proprio'], self.codec.proprio(self.pose, 1.))
        self.assertEqual(tuple(kwargs['input_image'].shape), (1,3,224,448))
        self.assertEqual(kwargs['num_inference_steps'], 20)
        # Execute the actual workstation validation functions without hardware imports.
        import ast
        source = ast.parse((ROOT/'third_party/FastAPI/client_fastwam_ee.py').read_text())
        names = {'validate_pose', 'rotation_distance', 'checked_chunk', 'validate_policy_metadata'}
        module = ast.Module(body=[n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
        env = dict(np=np, VERSION=1, ACTION_SPACE=policy.metadata['action_space'],
            CAMERAS=tuple(policy.metadata['camera_keys']), PREPROCESSING=policy.metadata['preprocessing'],
            GRIPPER_WIDTH=policy.metadata['gripper_open_threshold_m'], PROMPT=policy.metadata['instructions'][0])
        exec(compile(module, '<client contract>', 'exec'), env)
        env['validate_policy_metadata'](unpack(pack(policy.metadata)))
        poses, _ = env['checked_chunk'](result, 0, self.pose, replan_steps=8,
            position_min=self.codec.lo, position_max=self.codec.hi)
        self.assertEqual(poses.shape, (8,7))

    def test_joint_rgb_and_invalid_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = self.policy(Path(directory)/'deleted_or_new')
            obs = self.observation(policy)
            result = policy.infer(obs)
            folder = policy.save_rgb_dir/result['prediction_artifact_key']
            self.assertEqual(len(list(folder.glob('step_*_pred_rgb.png'))), 8)
            self.assertFalse(list(folder.glob('*rothko*')))
            self.assertFalse(self.model.infer_joint.call_args.kwargs['test_action_with_infer_action'])
            self.model.infer_action.assert_not_called()
            for field, value in [('prompt', 'wrong'), ('gripper_open', float('nan')),
                                 ('ee_link', 'tool'), ('record_session', '../escape')]:
                with self.subTest(field=field), self.assertRaises(ValueError):
                    policy.infer(dict(obs, **{field:value}))


if __name__ == '__main__':
    unittest.main()
