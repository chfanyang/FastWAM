import unittest
from types import SimpleNamespace
import torch
from fastwam.representations.libero_latent_repeat import LiberoLatentRepeat
from fastwam.representations.libero_osc import axis_angle_to_quaternion_wxyz
from fastwam.representations.rothko import quaternion_wxyz_to_matrix


class LatentRepeatTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        fake=SimpleNamespace(config=SimpleNamespace(frame0_pose_mode='absolute',image_height=224,image_width=448,
            absolute_position_min=[-1,-1,-1],absolute_position_max=[1,1,1]),
            norm_stats=SimpleNamespace(metadata={'translation_abs_bounds_xyz_m':[.2,.3,.4]}))
        self.c=LiberoLatentRepeat(fake)
        self.p=torch.cat((torch.rand(2,17,3)*.08,axis_angle_to_quaternion_wxyz(torch.randn(2,17,3))),-1)
        self.g=torch.randint(0,2,(2,17,1)).float()

    def test_roundtrip_nonidentity_reference(self):
        z=self.c.encode(self.p,self.g);p,g,bad=self.c.decode(z,self.p[:,0])
        torch.testing.assert_close(p[...,:3],self.p[...,:3],atol=2e-6,rtol=0)
        torch.testing.assert_close(quaternion_wxyz_to_matrix(p[...,3:]),quaternion_wxyz_to_matrix(self.p[...,3:]),atol=3e-6,rtol=0)
        torch.testing.assert_close(g,self.g);self.assertFalse(bad.any())

    def test_chw_group_mapping(self):
        v=torch.arange(17*8).reshape(1,17,8).float();z=self.c.pack(v)
        for group in range(4):
            torch.testing.assert_close(z[:,:,group+1].flatten()[:32],v[:,1+4*group:5+4*group].flatten())
        torch.testing.assert_close(self.c.unpack(z),v)

    def test_clean_state_cannot_contain_future(self):
        z=self.c.encode(self.p,self.g);changed=self.p.clone();changed[:,1:,:3]+=.01
        # Only future changes, including future grip, leave the condition bit exact.
        g=self.g.clone();g[:,1:]=1-g[:,1:]
        z2=self.c.encode(changed,g)
        self.assertTrue(torch.equal(z[:,:,:1],z2[:,:,:1]))

    def test_sign_invariance(self):
        other=self.p.clone();other[...,3:]*=-1
        torch.testing.assert_close(self.c.encode(other,self.g),self.c.encode(self.p,self.g),atol=1e-6,rtol=0)

    def test_repeated_copies_are_averaged(self):
        v=torch.zeros(1,17,8);z=self.c.pack(v);z[:,0,1,0,0]=784
        self.assertAlmostEqual(self.c.unpack(z)[0,1,0].item(),1.)

    def test_degenerate_quaternion_and_bf16(self):
        z=self.c.encode(self.p,self.g).bfloat16();p,g,bad=self.c.decode(z,self.p[:,0])
        self.assertTrue(torch.isfinite(p).all());self.assertFalse(bad.any())
        p,g,bad=self.c.decode(torch.zeros_like(z),self.p[:,0]);self.assertTrue(bad.all());self.assertTrue(torch.isfinite(p).all())

    def test_group_validity_matches_baseline(self):
        from fastwam.models.wan22.fastwam_visual_action import FastWAMVideoOnlyRaymap
        from fastwam.models.wan22.fastwam_latent_repeat import FastWAMLatentRepeat
        self.assertIs(FastWAMLatentRepeat.training_loss,FastWAMVideoOnlyRaymap.training_loss)
        self.assertIs(FastWAMLatentRepeat._latent_future_valid_mask,FastWAMVideoOnlyRaymap._latent_future_valid_mask)

    def test_current_gripper_image_layout(self):
        from fastwam.representations.libero_rothko import LiberoRothkoCodec, LiberoRothkoCodecConfig
        from pathlib import Path
        import json
        root=Path(__file__).resolve().parents[1]/'data/libero_mujoco3.3.2'
        meta=json.loads((root/'libero_absolute_ray0_all4_minmax_v1.json').read_text())
        cfg=LiberoRothkoCodecConfig(frame0_pose_mode='absolute',absolute_position_min=meta['absolute_position_min'],absolute_position_max=meta['absolute_position_max'])
        codec=LiberoRothkoCodec(cfg,root/'libero_rothko_region_symmetric_q99p95_h16_224x448.pt')
        pose=torch.tensor([[[0.,0.,.7,1.,0.,0.,0.]]]);grip=torch.tensor([[[.25]]])
        image=codec.encode(pose,grip)[:,:,0]
        actual=codec.read_gripper(image[:,None])
        self.assertEqual(actual.shape,(1,1,1));torch.testing.assert_close(actual,grip)

    def test_checkpoint_rejects_baseline(self):
        from fastwam.models.wan22.fastwam_latent_repeat import FastWAMLatentRepeat
        with self.assertRaisesRegex(ValueError,'identity'):
            FastWAMLatentRepeat._validate_visual_action_checkpoint_config(SimpleNamespace(repeat_codec=self.c),{},checkpoint_path='unused')


if __name__=='__main__':unittest.main()
