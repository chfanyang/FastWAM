"""Run in fastwam_libero; exercises the real eval helpers without a simulator."""
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from hydra import compose,initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf,open_dict
from experiments.libero.eval_matched_fastwam import predict,base,run_matched_evaluation
from fastwam.libero_matched_fastwam import MatchedPoseCodec
from fastwam.representations.libero_rothko import LiberoRothkoCodec,LiberoRothkoCodecConfig
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.config_resolvers import register_default_resolvers


class EvalTest(unittest.TestCase):
    def test_full_entry_keeps_matched_factory_arguments_and_restores_hooks(self):
        register_default_resolvers()
        root=Path(__file__).resolve().parents[1]
        with initialize_config_dir(config_dir=str(root/'configs'),version_base='1.3'):
            cfg=compose(config_name='sim_libero_matched_fastwam')
        cfg.ckpt='/test/not_loaded.pt'
        cfg.EVALUATION.device='cpu'
        original_predict=base._predict_action_chunk
        class ReachedFactory(Exception):pass
        captured={}
        def factory(config,**kwargs):
            captured.update(OmegaConf.to_container(config,resolve=True))
            raise ReachedFactory()
        with patch.object(base,'instantiate',side_effect=factory) as mocked:
            with self.assertRaises(ReachedFactory):run_matched_evaluation(cfg)
            self.assertIs(base.instantiate,mocked)
        self.assertIs(base._predict_action_chunk,original_predict)
        self.assertTrue(captured['skip_dit_load_from_pretrain'])
        self.assertIn('action_dit_pretrained_path',captured)
        self.assertIsNone(captured['action_dit_pretrained_path'])

    def test_same_osc_commands_and_current_state(self):
        torch.set_num_threads(2);register_default_resolvers()
        root=Path(__file__).resolve().parents[1]
        with initialize_config_dir(config_dir=str(root/'configs'),version_base='1.3'):
            cfg=compose(config_name='sim_libero_matched_fastwam')
        OmegaConf.resolve(cfg)
        codec=LiberoRothkoCodec(LiberoRothkoCodecConfig(**OmegaConf.to_container(cfg.data.train.rothko_config)),cfg.data.train.rothko_norm_stats)
        numeric=MatchedPoseCodec(codec)
        obs=dict(agentview_image=np.zeros((256,256,3),dtype=np.uint8),
            robot0_eye_in_hand_image=np.zeros((256,256,3),dtype=np.uint8),
            robot0_eef_pos=np.array([.05,.02,.7]),robot0_eef_quat=np.array([0.,0.,0.,1.]),
            robot0_gripper_qpos=np.array([.04,.04]))
        pose=base._extract_absolute_pose_wxyz(obs);grip=base._extract_gripper_open(obs)
        poses=pose[None,None].repeat(1,17,1);poses[:,1:,:3]+=torch.tensor([.01,.02,-.01])
        grips=grip.reshape(1,1,1).repeat(1,17,1);vectors=numeric.vectors(poses,grips)
        class Model:
            torch_dtype=torch.float32
            numeric_codec=numeric
            def infer_action(self,**kw):
                self.last_kwargs=kw
                torch.testing.assert_close(kw['proprio'],vectors[:,0])
                assert kw['num_inference_steps']==20 and kw['action_horizon']==16
                return {'action':vectors[0,1:]}
        processor=instantiate(cfg.data.train.processor).eval()
        processor.set_normalizer_from_stats(load_dataset_stats_from_json(cfg.EVALUATION.dataset_stats_path))
        model=Model()
        chunk,_,_=predict(obs,'put the bowl on the plate',model,processor,cfg,
            action_horizon=16,input_w=448,input_h=224,model_device='cpu')
        self.assertEqual(model.last_kwargs['seed'],42)
        self.assertIsNone(model.last_kwargs['sigma_shift'])
        expected,_=codec.decode(codec.encode(poses[0],grips[0]),pose)
        for i in range(16):
            a=base._absolute_target_to_env_action(obs,chunk['target_pose'][i],chunk['gripper_open'][i],binarize_gripper=True)
            b=base._absolute_target_to_env_action(obs,expected[i+1],grips[0,i+1],binarize_gripper=True)
            np.testing.assert_allclose(a,b,atol=2e-6,rtol=0)
        with open_dict(cfg.EVALUATION):
            cfg.EVALUATION.inference_seed=43
            cfg.EVALUATION.sigma_shift=3.
        predict(obs,'put the bowl on the plate',model,processor,cfg,
            action_horizon=16,input_w=448,input_h=224,model_device='cpu')
        self.assertEqual(model.last_kwargs['seed'],43)
        self.assertEqual(model.last_kwargs['sigma_shift'],3.)
        cfg.EVALUATION.text_cfg_scale=2.
        with self.assertRaisesRegex(ValueError,'CFG'):
            predict(obs,'put the bowl on the plate',model,processor,cfg,
                action_horizon=16,input_w=448,input_h=224,model_device='cpu')


if __name__=='__main__':unittest.main()
