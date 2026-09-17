import unittest
import torch
from dataclasses import replace
from fastwam.datasets.vlabench_rgb import build_vlabench_rgb_canvas
from fastwam.representations.libero_rothko import LiberoRothkoCodecConfig
from fastwam.representations.vlabench_rothko import VLABenchRothkoCodec, VLABenchRothkoCodecConfig
from fastwam.representations.rothko import RothkoNormStats


class ThreeCameraTest(unittest.TestCase):
    def test_camera_order_and_numpy_equivalence(self):
        images = {k: torch.full((2, 3, 20, 20), v, dtype=torch.uint8)
                  for k,v in zip(("image", "second_image", "wrist_image"), (0,128,255))}
        result = build_vlabench_rgb_canvas(images)
        self.assertEqual(tuple(result.shape), (2,3,192,576))
        self.assertTrue(torch.allclose(result[..., :192], torch.full_like(result[..., :192], -1)))
        self.assertTrue(torch.allclose(result[..., 384:], torch.ones_like(result[...,384:])))
        arrays = {k:v.movedim(-3,-1).numpy() for k,v in images.items()}
        self.assertTrue(torch.equal(result, build_vlabench_rgb_canvas(arrays)))

    def test_legacy_metadata_unchanged(self):
        self.assertEqual(VLABenchRothkoCodec(config=LiberoRothkoCodecConfig()).metadata(),
                         VLABenchRothkoCodec(config=VLABenchRothkoCodecConfig()).metadata())

    def test_three_tiles_gripper_and_pose_roundtrip(self):
        cfg = VLABenchRothkoCodecConfig(image_height=192,image_width=576,
                tile_height=192,tile_width=192,horizontal_copies=3)
        lo = torch.full((1,3,192,576), -1.)
        codec = VLABenchRothkoCodec(cfg, RothkoNormStats(lo=lo,hi=-lo,metadata={}))
        pose = torch.tensor([[0.,0.,0.,1.,0.,0.,0.], [.03,.02,-.01,1.,0.,0.,0.]])
        grip = torch.tensor([[1.],[0.]])
        ray = codec.encode(pose,grip)
        self.assertEqual(tuple(ray.shape), (3,2,192,576))
        self.assertTrue(torch.equal(ray[...,:192],ray[...,192:384]))
        self.assertTrue(torch.equal(ray[...,:192],ray[...,384:]))
        decoded, g = codec.decode(ray, pose[0])
        torch.testing.assert_close(decoded,pose,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(g,grip,atol=0,rtol=0)
        codec.decode_mode = 'robust_joint'
        decoded_robust, grip_robust = codec.decode(ray, pose[0])
        torch.testing.assert_close(decoded_robust,pose,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(grip_robust,grip,atol=0,rtol=0)
        codec.decode_mode = 'legacy'
        decoded_again, _ = codec.decode(ray, pose[0])
        torch.testing.assert_close(decoded_again,decoded,atol=0,rtol=0)
        with self.assertRaises(ValueError):
            replace(cfg,image_width=448).validate()


if __name__ == "__main__":
    unittest.main()
