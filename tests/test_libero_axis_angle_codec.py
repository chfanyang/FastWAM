import math
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from fastwam.representations.libero_rothko import LiberoRothkoCodec, LiberoRothkoCodecConfig
from fastwam.representations.libero_axis_angle import LiberoAxisAngleCodec
from fastwam.representations.libero_osc import axis_angle_to_quaternion_wxyz
from fastwam.representations.rothko import quaternion_wxyz_to_matrix, matrix_to_quaternion_wxyz

ROOT = Path(__file__).resolve().parents[1]


def geometry():
    cfg = OmegaConf.load(ROOT / 'runs/libero_goal_rothko_absolute_ray0_2cam224_full_wan21_1_3b_1e-4/2026-09-21_18-42-23/config.yaml')
    base = LiberoRothkoCodec(LiberoRothkoCodecConfig(**OmegaConf.to_container(cfg.data.train.rothko_config)),
                            ROOT / cfg.data.train.rothko_norm_stats)
    return base, LiberoAxisAngleCodec(base)


def example():
    base_q = axis_angle_to_quaternion_wxyz(torch.tensor([.4, -.2, .1], dtype=torch.float64))
    r0 = quaternion_wxyz_to_matrix(base_q)
    angles = torch.tensor([[0., 0., 0.], [1e-10, 0., 0.], [0., 0., .8],
                           [math.pi-1e-8, 0., 0.], [0., math.pi, 0.]], dtype=torch.float64)
    rotations = r0 @ quaternion_wxyz_to_matrix(axis_angle_to_quaternion_wxyz(angles))
    pose = torch.zeros(5, 7, dtype=torch.float64)
    pose[:, :3] = torch.tensor([.02, -.03, .7])
    pose[1:, :3] += torch.tensor([.01, -.02, .03], dtype=torch.float64)
    pose[:, 3:] = matrix_to_quaternion_wxyz(rotations)
    return pose, torch.tensor([[1.], [1.], [0.], [.4], [0.]], dtype=torch.float64)


class AxisAngleCodecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_only_rotation_ring_changes(self):
        base, codec = geometry(); pose, grip = example()
        old_meta = base.metadata(); old_stats = base.norm_stats.fingerprint()
        saved_pose = pose.clone(); saved_grip = grip.clone()
        a = base.encode(pose, grip); b = codec.encode(pose, grip)
        center, _, _, border = codec.masks(b.device)
        ring = ~center & ~border
        expected = (codec.rotation_vectors(pose[None])[0]/math.pi).clamp(-1,1).T
        for old,new in zip(a.split(224,-1),b.split(224,-1)):
            self.assertTrue(torch.equal(old[..., center|border],new[..., center|border]))
            torch.testing.assert_close(new[...,ring],expected[...,None].expand(-1,-1,int(ring.sum())),rtol=0,atol=0)
        self.assertTrue(torch.equal(b[...,:224],b[...,224:]))
        self.assertTrue(torch.equal(saved_pose,pose));self.assertTrue(torch.equal(saved_grip,grip))
        self.assertEqual(base.metadata(),old_meta)
        self.assertEqual(base.norm_stats.fingerprint(),old_stats)
        self.assertNotEqual(codec.metadata()['raymap_representation'],old_meta['raymap_representation'])

    def test_roundtrip_zero_pi_and_no_svd(self):
        base, codec = geometry(); pose, grip = example()
        pixels = codec.encode(pose,grip)
        with patch.object(base,'decode',side_effect=AssertionError('Must not fit direction fields')), \
             patch('torch.linalg.svd',side_effect=AssertionError('No SVD for constant axis-angle')):
            recovered,g = codec.decode(pixels,pose[0])
        torch.testing.assert_close(recovered[:,:3],pose[:,:3],rtol=0,atol=1e-12)
        torch.testing.assert_close(quaternion_wxyz_to_matrix(recovered[:,3:]),quaternion_wxyz_to_matrix(pose[:,3:]),rtol=0,atol=1e-7)
        torch.testing.assert_close(g,grip,rtol=0,atol=1e-12)

    def test_batch_sign_and_observed_reference(self):
        _,codec = geometry(); pose,grip = example()
        batched = pose[None].repeat(2,1,1); batched[1,:,3:] *= -1
        pixels = codec.encode(batched,grip[None].repeat(2,1,1))
        self.assertTrue(torch.equal(pixels[0],pixels[1]))
        p,g = codec.decode(pixels,batched[:,0])
        damaged = pixels.clone();damaged[:,:,0] = .9
        changed,_ = codec.decode(damaged,batched[:,0])
        torch.testing.assert_close(changed,p,rtol=0,atol=0)
        self.assertEqual(g.shape,(2,5,1))
        one = codec.encode(pose[:1],grip[:1]);p,_ = codec.decode(one,pose[0])
        self.assertEqual(one.shape,(3,1,224,448))
        torch.testing.assert_close(quaternion_wxyz_to_matrix(p[:,3:]),quaternion_wxyz_to_matrix(pose[:1,3:]),rtol=0,atol=1e-12)

    def test_translation_clipping_and_invalid_inputs(self):
        base,codec = geometry();pose,grip = example()
        pose[2,:3] += 2
        pixels=codec.encode(pose,grip);p,g=codec.decode(pixels,pose[0]);expected,eg=base.decode(base.encode(pose,grip),pose[0])
        torch.testing.assert_close(p[:,:3],expected[:,:3],rtol=0,atol=1e-12)
        torch.testing.assert_close(g,eg,rtol=0,atol=0)
        bad=pose.clone();bad[0,3:]=0
        with self.assertRaises(ValueError):codec.encode(bad,grip)
        with self.assertRaises(ValueError):codec.set_decode_anchor_video(pixels)

    def test_exact_pi_observed_state_is_encoded_and_recovered(self):
        _,codec=geometry()
        pose=torch.tensor([[.02,-.03,.7,0.,1.,0.,0.]],dtype=torch.float64).repeat(17,1)
        grip=torch.ones(17,1,dtype=torch.float64)
        rv=codec.rotation_vectors(pose[None])[0]
        torch.testing.assert_close(rv[0],torch.tensor([math.pi,0.,0.],dtype=torch.float64),rtol=0,atol=1e-12)
        torch.testing.assert_close(rv[1:],torch.zeros_like(rv[1:]),rtol=0,atol=1e-12)
        restored,_=codec.decode(codec.encode(pose,grip),pose[0])
        torch.testing.assert_close(quaternion_wxyz_to_matrix(restored[:,3:]),quaternion_wxyz_to_matrix(pose[:,3:]),rtol=0,atol=1e-12)
        opposite=pose.clone();opposite[:,3:]*=-1
        self.assertTrue(torch.equal(codec.encode(pose,grip),codec.encode(opposite,grip)))


if __name__=='__main__':unittest.main()
