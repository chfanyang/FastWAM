import unittest
from dataclasses import replace
import torch
from fastwam.representations.libero_rothko import LiberoRothkoCodecConfig,LiberoRothkoCodec
from fastwam.representations.libero_rothko_all_absolute import LiberoAllAbsoluteRothkoCodec
from fastwam.representations.rothko import RothkoNormStats,quaternion_wxyz_to_matrix

class AllAbsoluteTest(unittest.TestCase):
    def setUp(self):
        self.cfg=LiberoRothkoCodecConfig(image_height=32,image_width=64,tile_height=32,tile_width=32,boundary_margin=2,outer_margin=2,frame0_pose_mode='absolute',absolute_position_min=(-1.,-1.,-1.),absolute_position_max=(1.,1.,1.))
        codec=LiberoAllAbsoluteRothkoCodec(self.cfg)
        self.stats=RothkoNormStats(lo=torch.full((1,3,32,64),-1.),hi=torch.ones(1,3,32,64),metadata=codec.metadata())
        self.codec=LiberoAllAbsoluteRothkoCodec(self.cfg,self.stats)
        g=torch.Generator().manual_seed(43)
        self.pose=torch.rand(2,17,7,generator=g)*.6-.3
        self.pose[...,3:]=torch.nn.functional.normalize(torch.randn(2,17,4,generator=g),dim=-1)
        self.grip=torch.randint(0,2,(2,17,1),generator=g).float()
    def test_roundtrip_and_no_anchor_dependency(self):
        v=self.codec.encode(self.pose,self.grip);p,g=self.codec.decode(v)
        torch.testing.assert_close(p[...,:3],self.pose[...,:3],atol=1e-6,rtol=0)
        torch.testing.assert_close(quaternion_wxyz_to_matrix(p[...,3:]),quaternion_wxyz_to_matrix(self.pose[...,3:]),atol=1e-6,rtol=0)
        torch.testing.assert_close(g,self.grip)
        changed=v.clone();changed[:,:,0]=0
        pp,_=self.codec.decode(changed,torch.randn(2,7))
        torch.testing.assert_close(pp[:,1:],p[:,1:],atol=0,rtol=0)
        one,_=self.codec.decode(v[0]);torch.testing.assert_close(one,p[0])
    def test_framewise_encoding_and_duplicates(self):
        v=self.codec.encode(self.pose,self.grip)
        mixed=LiberoRothkoCodec(self.cfg,self.stats_without_identity())
        expected=mixed.encode(self.pose.reshape(-1,1,7),self.grip.reshape(-1,1,1)).reshape(2,17,3,32,64).permute(0,2,1,3,4)
        self.assertTrue(torch.equal(v,expected))
        self.assertTrue(torch.equal(v[:,:,:1],self.codec.encode(self.pose[:,:1],self.grip[:,:1])))
        for tile in v.split(32,-1):self.assertTrue(torch.equal(tile,v[...,:32]))
    def stats_without_identity(self):
        return RothkoNormStats(lo=self.stats.lo,hi=self.stats.hi,metadata={})
    def test_reject_wrong_stats_bounds_and_decoder(self):
        with self.assertRaises(ValueError):LiberoRothkoCodec(self.cfg,self.stats)
        with self.assertRaises(ValueError):LiberoAllAbsoluteRothkoCodec(self.cfg,self.stats_without_identity())
        with self.assertRaises(ValueError):LiberoAllAbsoluteRothkoCodec(replace(self.cfg,absolute_position_max=(2.,1.,1.)),self.stats)
        with self.assertRaises(ValueError):LiberoAllAbsoluteRothkoCodec(self.cfg,self.stats,decode_mode='robust_joint')

    def test_model_dispatch_and_checkpoint_identity(self):
        from dataclasses import asdict
        from test_visual_action_representation_metadata import _DummyVideoExpert, _DummyVae
        from fastwam.models.wan22.fastwam_visual_action import FastWAMVideoOnlyRaymap
        model = FastWAMVideoOnlyRaymap(
            video_expert=_DummyVideoExpert(), vae=_DummyVae(), text_dim=16,
            device='cpu', action_horizon=16,
            raymap_representation='libero_rothko_all_absolute',
            rothko_norm_stats=self.stats, rothko_config=asdict(self.cfg))
        self.assertIsInstance(model.raymap_codec, LiberoAllAbsoluteRothkoCodec)
        metadata = model._visual_action_checkpoint_config()
        model._validate_visual_action_checkpoint_config(metadata, checkpoint_path='new.pt')
        wrong = dict(metadata, raymap_representation='libero_rothko')
        with self.assertRaises(ValueError):
            model._validate_visual_action_checkpoint_config(wrong, checkpoint_path='old.pt')

if __name__=='__main__':unittest.main()
