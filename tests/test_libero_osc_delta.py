import unittest
from dataclasses import asdict
from pathlib import Path
import torch
from fastwam.representations.libero_rothko import LiberoRothkoCodecConfig
from fastwam.representations.libero_rothko_osc_delta import LiberoOSCDeltaRothkoCodec,action_to_delta_pose,delta_pose_to_action_motion
from fastwam.representations.libero_rothko_all_absolute import LiberoAllAbsoluteRothkoCodec
from fastwam.representations.rothko import RothkoNormStats,quaternion_wxyz_to_matrix

class OSCDeltaTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.cfg=LiberoRothkoCodecConfig(image_height=32,image_width=64,tile_height=32,tile_width=32,boundary_margin=2,outer_margin=2,frame0_pose_mode='absolute',absolute_position_min=(-1.,-1.,-1.),absolute_position_max=(2.,2.,2.))
        c=LiberoOSCDeltaRothkoCodec(self.cfg);lo,hi=c.delta_bounds(torch.zeros(1,3,32,64));self.stats=RothkoNormStats(lo,hi,c.metadata());self.c=LiberoOSCDeltaRothkoCodec(self.cfg,self.stats)
        gen=torch.Generator().manual_seed(5);self.a=torch.rand(2,16,7,generator=gen)*2-1;self.a[...,6]=(self.a[...,6]>0).float();self.a[:,0,:6]=0
        self.p=torch.tensor([.2,.1,1.,1.,0.,0.,0.]).expand(2,1,7).clone();self.d=action_to_delta_pose(self.a);self.poses=torch.cat((self.p,self.d),1);self.g=torch.cat((torch.ones(2,1,1),self.a[...,-1:]),1)

    def test_raw_commands_roundtrip_including_zero(self):
        torch.testing.assert_close(delta_pose_to_action_motion(self.d),self.a[...,:6],atol=1e-6,rtol=0)
        p,g=self.c.decode(self.c.encode(self.poses,self.g))
        torch.testing.assert_close(p[:,:1,:3],self.p[:,:,:3],atol=1e-6,rtol=0)
        torch.testing.assert_close(delta_pose_to_action_motion(p[:,1:]),self.a[...,:6],atol=2e-6,rtol=0)
        torch.testing.assert_close(g,self.g)
        torch.testing.assert_close(p[:,1,:3],torch.zeros(2,3),atol=1e-8,rtol=0)

    def test_future_independent_of_current_state_and_no_cumulative_sum(self):
        v=self.c.encode(self.poses,self.g);changed=self.poses.clone();changed[:,0,0]+=.5;changed[:,0,3:]=torch.tensor([0.,1.,0.,0.]);vv=self.c.encode(changed,self.g)
        self.assertTrue(torch.equal(v[:,:,1:],vv[:,:,1:]));self.assertFalse(torch.equal(v[:,:,:1],vv[:,:,:1]))
        self.assertTrue(torch.equal(v[:,:,:1],self.c.encode(self.p,self.g[:,:1])))
        seq=self.poses.clone();seq[:,1:,:3]=.01
        decoded,_=self.c.decode(self.c.encode(seq,self.g));torch.testing.assert_close(decoded[:,1:,:3],seq[:,1:,:3],atol=1e-7,rtol=0)
        damaged=v.clone();damaged[:,:,0]=0
        pd,_=self.c.decode(damaged,torch.randn(2,7));p,_=self.c.decode(v)
        torch.testing.assert_close(pd[:,1:],p[:,1:],atol=0,rtol=0)

    def test_stats_and_controller_limits_rejected(self):
        with self.assertRaises(ValueError):LiberoAllAbsoluteRothkoCodec(self.cfg,self.stats)
        bad=dict(self.stats.metadata,osc_position_scale_m=.1)
        with self.assertRaises(ValueError):LiberoOSCDeltaRothkoCodec(self.cfg,RothkoNormStats(self.stats.lo,self.stats.hi,bad))
        with self.assertRaises(ValueError):action_to_delta_pose(torch.ones(1,7)*2)
        with self.assertRaises(ValueError):LiberoOSCDeltaRothkoCodec(self.cfg,self.stats,decode_mode='robust_joint')

    def test_dispatch_and_identity(self):
        from test_visual_action_representation_metadata import _DummyVideoExpert,_DummyVae
        from fastwam.models.wan22.fastwam_visual_action import FastWAMVideoOnlyRaymap
        m=FastWAMVideoOnlyRaymap(video_expert=_DummyVideoExpert(),vae=_DummyVae(),text_dim=16,device='cpu',action_horizon=16,raymap_representation=self.c.representation,rothko_norm_stats=self.stats,rothko_config=asdict(self.cfg))
        self.assertIsInstance(m.raymap_codec,LiberoOSCDeltaRothkoCodec);meta=m._visual_action_checkpoint_config();m._validate_visual_action_checkpoint_config(meta,checkpoint_path='new.pt')
        bad=dict(meta);bad.pop('osc_position_scale_m')
        with self.assertRaises(ValueError):m._validate_visual_action_checkpoint_config(bad,checkpoint_path='bad.pt')

    def test_config_budget_and_raw_fields(self):
        from hydra import compose,initialize_config_dir
        from fastwam.utils.config_resolvers import register_default_resolvers
        register_default_resolvers()
        with initialize_config_dir(config_dir=str(Path('configs').resolve()),version_base='1.3'):
            c=compose(config_name='train',overrides=['task=libero_goal_rothko_osc_delta_2cam224_full_wan21_1_3b_1e-4']);ev=compose(config_name='sim_libero_goal_osc_delta')
        self.assertEqual(c.data.train.raw_action_meta[0].key,'default');self.assertEqual(len(c.data.train.raw_action_meta),1)
        self.assertEqual(c.data.train.raw_state_meta[0].key,'default');self.assertEqual(c.model.raymap_representation,self.c.representation)
        self.assertEqual((c.num_epochs,c.batch_size,c.gradient_accumulation_steps,c.save_every),(10,4,4,2000));self.assertEqual(ev.EVALUATION.replan_steps,8)
        self.assertFalse(ev.EVALUATION.use_action_ensembler)
