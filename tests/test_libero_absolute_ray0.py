"""Opt-in absolute RAY0 contract; old relative codec remains the default."""
import unittest
from dataclasses import replace

import torch

from fastwam.representations.libero_rothko import LiberoRothkoCodec, LiberoRothkoCodecConfig
from fastwam.representations.libero_osc import axis_angle_to_quaternion_wxyz
from fastwam.representations.rothko import RothkoNormStats, quaternion_wxyz_to_matrix


class AbsoluteRay0Test(unittest.TestCase):
    def setUp(self):
        self.cfg = LiberoRothkoCodecConfig(image_height=32, image_width=64,
            tile_height=32, tile_width=32, outer_margin=2, boundary_margin=2)
        lo = torch.full((1, 3, 32, 64), -1.)
        self.stats = RothkoNormStats(lo=lo, hi=-lo, metadata={})
        self.old = LiberoRothkoCodec(self.cfg, self.stats)
        self.new = LiberoRothkoCodec(replace(self.cfg, frame0_pose_mode='absolute',
            absolute_position_min=(-1., -1., 0.), absolute_position_max=(1., 1., 2.)), self.stats)
        self.pose = torch.zeros(2, 17, 7)
        self.pose[..., :3] = torch.tensor([0.25, -0.35, 0.9])
        self.pose[1, :, 0] += 0.1
        self.pose[:, :, 0] += torch.linspace(0, .04, 17)
        self.pose[..., 3:] = axis_angle_to_quaternion_wxyz(torch.tensor([.2, -.3, .4]).expand(2,17,3))
        self.pose[:, 1:, 3:] = axis_angle_to_quaternion_wxyz(torch.tensor([.3, -.2, .5]).expand(2,16,3))
        self.grip = torch.rand(2,17,1, generator=torch.Generator().manual_seed(3))

    def test_future_frames_bitwise_unchanged_and_frame0_informative(self):
        old, new = self.old.encode(self.pose, self.grip), self.new.encode(self.pose, self.grip)
        self.assertTrue(torch.equal(old[:, :, 1:], new[:, :, 1:]))
        self.assertFalse(torch.equal(old[:, :, 0], new[:, :, 0]))
        current = self.new.encode(self.pose[:, :1], self.grip[:, :1])
        self.assertTrue(torch.equal(current[:, :, 0], new[:, :, 0]))
        shifted = self.pose.clone(); shifted[..., 0] += .1
        self.assertFalse(torch.equal(self.new.encode(shifted, self.grip)[:, :, 0], new[:, :, 0]))

    def test_roundtrip_nonidentity_current_rotation(self):
        video = self.new.encode(self.pose, self.grip)
        actual, grip = self.new.decode(video, self.pose[:, 0])
        torch.testing.assert_close(actual[..., :3], self.pose[..., :3], atol=2e-6, rtol=0)
        torch.testing.assert_close(quaternion_wxyz_to_matrix(actual[..., 3:]),
            quaternion_wxyz_to_matrix(self.pose[..., 3:]), atol=2e-6, rtol=0)
        torch.testing.assert_close(grip, self.grip, atol=1e-6, rtol=0)
        # Changing the decoded absolute reference image must not change future
        # local geometry. The physical current pose is supplied externally.
        damaged = video.clone(); damaged[:, :, 0] = torch.rand_like(damaged[:, :, 0])
        changed, _ = self.new.decode(damaged, self.pose[:, 0])
        torch.testing.assert_close(changed, actual, atol=0, rtol=0)
        single, _ = self.new.decode(video[0], self.pose[0, 0])
        torch.testing.assert_close(single, actual[0])

    def test_absolute_raw_roundtrip(self):
        raw = self.new.encode_raw(self.pose)
        recovered = self.new.denormalize(self.new.normalize_raw(raw))
        torch.testing.assert_close(recovered, raw, atol=2e-7, rtol=0)

    def test_legacy_metadata_unchanged(self):
        meta = self.old.metadata()
        for k in ('frame0_pose_mode', 'absolute_position_min', 'absolute_position_max'):
            self.assertNotIn(k, meta)
        self.assertEqual(self.new.metadata()['frame0_pose_mode'], 'absolute')

    def test_requires_bounds_and_explicit_supported_decoder(self):
        with self.assertRaises(ValueError):
            LiberoRothkoCodec(replace(self.cfg, frame0_pose_mode='absolute'), self.stats)
        with self.assertRaises(ValueError):
            LiberoRothkoCodec(self.new.config, self.stats, decode_mode='robust_joint')


if __name__ == '__main__':
    unittest.main()
