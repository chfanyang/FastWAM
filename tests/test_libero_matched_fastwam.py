"""CPU checks for the isolated numeric-target MoT comparison."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from fastwam.libero_matched_fastwam import MatchedPoseCodec, MatchedFastWAM, create_model
from fastwam.representations.libero_rothko import LiberoRothkoCodec, LiberoRothkoCodecConfig
from fastwam.representations.rothko import quaternion_wxyz_to_matrix
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.mot import MoT
from fastwam.utils.config_resolvers import register_default_resolvers

ROOT=Path(__file__).resolve().parents[1]


def geometry():
    p=ROOT/'data/libero_mujoco3.3.2'
    a=json.loads((p/'libero_absolute_ray0_all4_minmax_v1.json').read_text())
    cfg=LiberoRothkoCodecConfig(frame0_pose_mode='absolute',absolute_position_min=a['absolute_position_min'],absolute_position_max=a['absolute_position_max'])
    return LiberoRothkoCodec(cfg,p/'libero_rothko_region_symmetric_q99p95_h16_224x448.pt')


class FakeVAE(torch.nn.Module):
    temporal_downsample_factor=4
    def encode(self,x,**kwargs):
        if isinstance(x,list):return [torch.zeros(16,1,28,56)]
        return torch.zeros(x.shape[0],16,5,28,56)


def tiny_model():
    video=WanVideoDiT(hidden_dim=32,in_dim=16,ffn_dim=64,out_dim=16,text_dim=32,freq_dim=16,
        eps=1e-6,patch_size=(1,2,2),num_heads=2,attn_head_dim=24,num_layers=1,
        has_image_input=False,seperated_timestep=True,require_vae_embedding=False,
        require_clip_embedding=False,fuse_vae_embedding_in_latents=True,
        video_attention_mask_mode='first_frame_causal')
    action=ActionDiT(hidden_dim=24,action_dim=8,ffn_dim=48,text_dim=32,freq_dim=16,
        eps=1e-6,num_heads=2,attn_head_dim=24,num_layers=1)
    mot=MoT({'video':video,'action':action},mot_checkpoint_mixed_attn=False)
    m=MatchedFastWAM(video,action,mot,FakeVAE(),text_dim=32,proprio_dim=8,torch_dtype=torch.float32)
    m.raymap_codec=geometry();m.numeric_codec=MatchedPoseCodec(m.raymap_codec)
    m.video_config={'test':True};m.action_config={'test':True};m.original_vae_sha256='fixture'
    return m


class MatchedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_configs_and_legacy_are_separate(self):
        register_default_resolvers()
        with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base='1.3'):
            c=compose(config_name='train',overrides=['task=libero_goal_matched_fastwam_wan21'])
            e=compose(config_name='sim_libero_matched_fastwam')
            old=compose(config_name='train',overrides=['task=libero_uncond_2cam224_1e-4'])
        OmegaConf.resolve(c);OmegaConf.resolve(e)
        self.assertEqual(c.model._target_,'fastwam.libero_matched_fastwam.create_model')
        self.assertNotIn('vae_safetensors_path',c.model)
        self.assertEqual(c.data.train.num_frames,17);self.assertEqual(c.data.train.action_video_freq_ratio,1)
        self.assertEqual(c.batch_size,4);self.assertEqual(c.gradient_accumulation_steps,8)
        self.assertEqual(c.batch_size*c.gradient_accumulation_steps*4,128)
        self.assertEqual(c.num_epochs,10);self.assertEqual(e.EVALUATION.replan_steps,8)
        self.assertEqual(c.save_every_epochs,1)
        self.assertEqual(c.state_save_every,2000);self.assertTrue(c.save_at_end)
        self.assertTrue(e.model.action_dit_pretrained_path)
        self.assertEqual(old.model._target_,'fastwam.runtime.create_fastwam')
        self.assertEqual(old.data.train.num_frames,33)
        self.assertIs(MatchedFastWAM.training_loss,FastWAM.training_loss)
        self.assertIs(MatchedFastWAM.infer_action,FastWAM.infer_action)

    def test_pose_roundtrip_and_observed_state(self):
        codec=MatchedPoseCodec(geometry())
        pose=torch.tensor([[[0.,0.,.7,.9238795,0.,0.,.3826834]]]).repeat(1,17,1)
        pose[:,1:,:3]+=torch.tensor([.01,-.02,.03]);grip=torch.ones(1,17,1)*.3
        v=codec.vectors(pose,grip);p,g,bad=codec.recover(v[:,1:],pose[:,0])
        torch.testing.assert_close(p[...,:3],pose[:,1:,:3],atol=1e-6,rtol=0)
        torch.testing.assert_close(quaternion_wxyz_to_matrix(p[...,3:]),quaternion_wxyz_to_matrix(pose[:,1:,3:]),atol=1e-6,rtol=0)
        torch.testing.assert_close(g,grip[:,1:]);self.assertFalse(bad.any())
        torch.testing.assert_close(codec.state_vector(pose[:,0],grip[:,0]),v[:,0])

    def test_real_mot_cached_forward_backward_and_action_inference(self):
        m=tiny_model()
        sample=dict(rgb_latents=torch.zeros(1,16,5,28,56),action=torch.rand(1,16,8),
            proprio=torch.rand(1,16,8),context=torch.rand(1,3,32),context_mask=torch.ones(1,3,dtype=torch.bool),
            action_is_pad=torch.zeros(1,16,dtype=torch.bool),image_is_pad=torch.zeros(1,17,dtype=torch.bool))
        online={k:v for k,v in sample.items() if k!='rgb_latents'}
        online['video']=torch.zeros(1,3,17,224,448)
        cached_inputs=m.build_inputs(sample);online_inputs=m.build_inputs(online)
        for key in cached_inputs:
            if isinstance(cached_inputs[key],torch.Tensor):
                torch.testing.assert_close(cached_inputs[key],online_inputs[key],rtol=0,atol=0)
        with patch.object(m.vae,'encode',side_effect=AssertionError('Cached training called VAE encoder')):
            loss,_=m.training_loss(sample)
        self.assertTrue(torch.isfinite(loss));loss.backward()
        for part in [m.video_expert,m.action_expert,m.proprio_encoder]:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in part.parameters()))
        pred=m.infer_action(prompt=None,input_image=torch.zeros(1,3,224,448),action_horizon=16,
            proprio=sample['proprio'][:,0],context=sample['context'],context_mask=sample['context_mask'],
            num_inference_steps=2,seed=42)
        self.assertEqual(pred['action'].shape,(16,8));self.assertTrue(torch.isfinite(pred['action']).all())

    def test_checkpoint_rejects_wrong_identity(self):
        m=tiny_model()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'w.pt';m.save_checkpoint(p,step=1);m.load_checkpoint(p)
            payload=torch.load(p,weights_only=False);payload['matched_identity']['horizon']=32;torch.save(payload,p)
            with self.assertRaisesRegex(ValueError,'identity'):m.load_checkpoint(p)

    def test_production_factory_architecture_without_allocating_weights(self):
        register_default_resolvers()
        with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base='1.3'):
            cfg=compose(config_name='train',overrides=['task=libero_goal_matched_fastwam_wan21'])
        OmegaConf.resolve(cfg)
        args=OmegaConf.to_container(cfg.model,resolve=True);args.pop('_target_')
        args.update(device='meta',model_dtype=torch.bfloat16,skip_dit_load_from_pretrain=True,
                    action_dit_pretrained_path=None)
        codec=geometry()
        from fastwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
        compute_stats=WanContinuousFlowMatchScheduler._precompute_training_weight_stats
        def cpu_stats(scheduler):
            with torch.device('cpu'):return compute_stats(scheduler)
        with torch.device('meta'):
            video=WanVideoDiT(**args['video_dit_config'])
            components=SimpleNamespace(dit=video,vae=FakeVAE(),text_encoder=None,tokenizer=None,vae_path='unused')
            with patch('fastwam.libero_matched_fastwam.load_wan_video_components',return_value=components) as loader, \
                 patch('fastwam.libero_matched_fastwam.LiberoRothkoCodec',return_value=codec), \
                 patch('fastwam.datasets.latent_cache.sha256_file',return_value='fixture'), \
                 patch.object(WanContinuousFlowMatchScheduler,'_precompute_training_weight_stats',cpu_stats):
                model=create_model(**args)
        self.assertEqual(loader.call_args.kwargs['model_variant'],'wan2.1-t2v-1.3b')
        self.assertEqual(model.action_expert.action_encoder.weight.shape,(1024,8))
        self.assertEqual(model.action_expert.head.weight.shape,(8,1024))
        self.assertEqual(model.proprio_encoder.weight.shape,(4096,8))
        self.assertEqual(model.mot.num_heads,12);self.assertEqual(model.mot.attn_head_dim,128)
        self.assertEqual(model.mot.num_layers,30)
        self.assertEqual(sum(p.numel() for p in model.video_expert.parameters()),1418996800)
        self.assertTrue(all(p.device.type=='meta' for p in model.parameters()))
        # Every ActionDiT backbone key must have a Wan2.1 source; shape-only
        # interpolation exercises the real initializer without allocating GBs.
        from scripts.preprocess_action_dit_backbone import _resize_tensor_to_shape
        source=model.video_expert.state_dict();target=model.action_expert.state_dict()
        keys=ActionDiT.backbone_key_set(target)
        self.assertTrue(keys.issubset(source))
        for key in keys:
            resized=_resize_tensor_to_shape(source[key],tuple(target[key].shape))
            self.assertEqual(resized.shape,target[key].shape)

    def test_rejects_wrong_backbone_before_loading_large_models(self):
        register_default_resolvers()
        with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base='1.3'):
            cfg=compose(config_name='train',overrides=['task=libero_goal_matched_fastwam_wan21'])
        args=OmegaConf.to_container(cfg.model,resolve=True);args.pop('_target_')
        with patch('torch.load',return_value={'source_model':'Wan-AI/Wan2.2-TI2V-5B'}), \
             patch('fastwam.libero_matched_fastwam.load_wan_video_components') as loader:
            with self.assertRaisesRegex(ValueError,'Wan2.2'):create_model(**args,device='cpu')
        loader.assert_not_called()


if __name__=='__main__':unittest.main()
