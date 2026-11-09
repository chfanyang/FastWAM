"""Use fastwam_libero: real eval preprocessing/control helpers, no rollout."""
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

import experiments.libero.eval_libero_single as base
from experiments.libero.eval_axis_angle import main as axis_entry
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.representations.libero_axis_angle import LiberoAxisAngleCodec
from fastwam.representations.libero_rothko import LiberoRothkoCodec,LiberoRothkoCodecConfig
from fastwam.representations.libero_osc import axis_angle_to_quaternion_wxyz
from fastwam.utils.config_resolvers import register_default_resolvers

ROOT=Path(__file__).resolve().parents[1]


def config():
    register_default_resolvers()
    with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base='1.3'):
        return compose(config_name='sim_libero_goal_axis_angle')


class AxisAngleEvalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_shared_entry_reaches_new_factory_without_action_expert_argument(self):
        cfg=config();cfg.ckpt='/diagnostic/not_loaded.pt';cfg.EVALUATION.device='cpu'
        captured={}
        class ReachedFactory(Exception):pass
        def factory(c,**kwargs):
            captured.update(OmegaConf.to_container(c,resolve=True));raise ReachedFactory()
        with patch.object(base,'instantiate',side_effect=factory):
            with self.assertRaises(ReachedFactory):axis_entry.__wrapped__(cfg)
        self.assertEqual(captured['_target_'],'fastwam.axis_angle_runtime.create_axis_angle_raymap')
        self.assertFalse(captured['skip_dit_load_from_pretrain'])
        self.assertNotIn('action_dit_pretrained_path',captured)
        self.assertIsNone(captured['vae_safetensors_path'])

    def test_new_ray0_image_and_recovered_pose_use_unchanged_osc_control(self):
        cfg=config();OmegaConf.resolve(cfg)
        reference=LiberoRothkoCodec(LiberoRothkoCodecConfig(**OmegaConf.to_container(cfg.data.train.rothko_config)),
            ROOT/cfg.data.train.rothko_norm_stats)
        codec=LiberoAxisAngleCodec(reference)
        obs=dict(agentview_image=np.zeros((256,256,3),dtype=np.uint8),
            robot0_eye_in_hand_image=np.zeros((256,256,3),dtype=np.uint8),
            robot0_eef_pos=np.array([.05,.02,.7]),robot0_eef_quat=np.array([0.,0.,0.,1.]),
            robot0_gripper_qpos=np.array([.04,.04]))
        current=base._extract_absolute_pose_wxyz(obs);grip=base._extract_gripper_open(obs)
        poses=current[None].repeat(17,1);poses[1:,:3]+=torch.tensor([.01,.02,-.01])
        poses[1:,3:]=axis_angle_to_quaternion_wxyz(torch.tensor([.1,.2,-.1]))
        grips=grip.reshape(1,1).repeat(17,1);grips[8:]=0
        class Model:
            torch_dtype=torch.float32
            raymap_codec=codec
            def infer(self,**kw):
                self.last_kwargs=kw
                torch.testing.assert_close(kw['input_raymap'],codec.encode(current[None],grip.reshape(1,1))[:,0][None],rtol=0,atol=0)
                torch.testing.assert_close(kw['current_endpose'],current)
                assert kw['num_inference_steps']==20 and kw['num_frames']==17
                p,g=codec.decode(codec.encode(poses,grips),current)
                return {'pose':p[None],'gripper':g[None]}
        processor=instantiate(cfg.data.train.processor).eval()
        processor.set_normalizer_from_stats(load_dataset_stats_from_json(cfg.EVALUATION.dataset_stats_path))
        model=Model()
        chunk,_,_=base._predict_action_chunk(obs,'put the bowl on the plate',model,processor,cfg,
            action_horizon=16,input_w=448,input_h=224,model_device='cpu')
        self.assertEqual(model.last_kwargs['seed'],42)
        self.assertEqual(chunk['target_pose'].shape,(16,7))
        for i in range(16):
            actual=base._absolute_target_to_env_action(obs,chunk['target_pose'][i],chunk['gripper_open'][i],binarize_gripper=True)
            expected=base._absolute_target_to_env_action(obs,poses[i+1],grips[i+1],binarize_gripper=True)
            np.testing.assert_allclose(actual,expected,rtol=0,atol=3e-6)


if __name__=='__main__':unittest.main()
