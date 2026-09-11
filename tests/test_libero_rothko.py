import math
import unittest

import torch

from fastwam.representations.libero_osc import axis_angle_to_quaternion_wxyz
from fastwam.representations.libero_rothko import (
    LiberoRothkoCodec,
    LiberoRothkoCodecConfig,
)
from fastwam.representations.rothko import (
    RothkoNormStats,
    _center_and_read_masks,
    quaternion_wxyz_to_matrix,
)


class LiberoRothkoCodecTest(unittest.TestCase):
    def _codec(
        self,
        decode_mode: str = "legacy",
        *,
        decode_anchor_alpha: float = 0.0,
        decode_block_grid: int = 4,
    ) -> LiberoRothkoCodec:
        config = LiberoRothkoCodecConfig()
        lo = torch.full((1, 3, config.image_height, config.image_width), -1.0)
        hi = torch.full_like(lo, 1.0)
        # Translation values live in the center; broad synthetic bounds avoid
        # clipping in this geometry-only test.
        return LiberoRothkoCodec(
            config,
            RothkoNormStats(
                lo=lo,
                hi=hi,
                metadata={
                    "environment": "libero",
                    "layout": "single_arm_duplicated_horizontal",
                },
            ),
            decode_mode=decode_mode,
            decode_anchor_alpha=decode_anchor_alpha,
            decode_block_grid=decode_block_grid,
        )

    @staticmethod
    def _trajectory() -> tuple[torch.Tensor, torch.Tensor]:
        time = 17
        pose = torch.zeros(time, 7, dtype=torch.float32)
        pose[:, :3] = torch.tensor([0.1, -0.2, 0.9])
        pose[:, :3] += torch.linspace(0, 0.04, time)[:, None] * torch.tensor(
            [1.0, -0.5, 0.25]
        )
        angles = torch.zeros(time, 3)
        angles[:, 2] = torch.linspace(0, 0.3, time)
        pose[:, 3:] = axis_angle_to_quaternion_wxyz(angles)
        gripper = (torch.arange(time) % 2).float().unsqueeze(-1)
        return pose, gripper

    def test_hybrid_matches_component_solvers_on_noisy_video(self) -> None:
        pose, gripper = self._trajectory()
        video = self._codec().encode(pose, gripper)
        noise = torch.randn(video.shape, generator=torch.Generator().manual_seed(42))
        video = (video + noise * 0.015).clamp(-1, 1)
        for grid in (2, 4):
            hybrid, hg = self._codec("block_position_joint_rotation", decode_block_grid=grid).decode(video, pose[0])
            block, bg = self._codec("robust_block_consensus", decode_block_grid=grid).decode(video, pose[0])
            joint, jg = self._codec("robust_joint").decode(video, pose[0])
            torch.testing.assert_close(hybrid[:, :3], block[:, :3], atol=0, rtol=0)
            torch.testing.assert_close(hybrid[:, 3:], joint[:, 3:], atol=0, rtol=0)
            torch.testing.assert_close(hg, bg, atol=0, rtol=0)
            torch.testing.assert_close(hg, jg, atol=0, rtol=0)

    def test_block_weighted_joint_roundtrip_and_corruption(self) -> None:
        pose, gripper = self._trajectory()
        video = self._codec().encode(pose, gripper)
        for grid in (2, 4):
            codec = self._codec('robust_block_weighted_joint', decode_block_grid=grid)
            decoded, grip = codec.decode(video, pose[0])
            torch.testing.assert_close(decoded, pose, atol=2e-5, rtol=0)
            torch.testing.assert_close(grip, gripper, atol=1e-6, rtol=0)
            damaged = video.clone()
            damaged[:, 1:, 64:110, 64:110] += 0.5
            damaged[:, 1:, 8:50, 8:50] *= -1
            decoded, _ = codec.decode(damaged, pose[0])
            self.assertTrue(torch.isfinite(decoded).all())
            self.assertLess((decoded[:, :3] - pose[:, :3]).norm(dim=-1).max().item(), 0.002)
            torch.testing.assert_close(decoded[0], pose[0], atol=1e-6, rtol=0)
            torch.testing.assert_close(decoded[:, 3:].norm(dim=-1), torch.ones(17), atol=1e-6, rtol=0)
        with self.assertRaisesRegex(ValueError, 'requires anchor_alpha=0'):
            self._codec('robust_block_weighted_joint', decode_anchor_alpha=.5).decode(video, pose[0])

    def test_geometry_and_gripper_roundtrip(self) -> None:
        codec = self._codec()
        pose, gripper = self._trajectory()

        video = codec.encode(pose, gripper)
        decoded_pose, decoded_gripper = codec.decode(video, pose[0])
        torch.testing.assert_close(decoded_pose[:, :3], pose[:, :3], atol=2e-5, rtol=0)
        expected_rotation = quaternion_wxyz_to_matrix(pose[:, 3:])
        actual_rotation = quaternion_wxyz_to_matrix(decoded_pose[:, 3:])
        torch.testing.assert_close(actual_rotation, expected_rotation, atol=2e-5, rtol=0)
        torch.testing.assert_close(decoded_gripper, gripper, atol=1e-6, rtol=0)

    def test_robust_joint_geometry_and_gripper_roundtrip(self) -> None:
        codec = self._codec(decode_mode="robust_joint")
        pose, gripper = self._trajectory()

        video = codec.encode(pose, gripper)
        decoded_pose, decoded_gripper = codec.decode(video, pose[0])
        torch.testing.assert_close(decoded_pose[:, :3], pose[:, :3], atol=2e-5, rtol=0)
        torch.testing.assert_close(
            quaternion_wxyz_to_matrix(decoded_pose[:, 3:]),
            quaternion_wxyz_to_matrix(pose[:, 3:]),
            atol=2e-5,
            rtol=0,
        )
        torch.testing.assert_close(decoded_gripper, gripper, atol=1e-6, rtol=0)

    def test_alternative_robust_decoders_roundtrip(self) -> None:
        pose, gripper = self._trajectory()
        video = self._codec().encode(pose, gripper)
        zero_pose = pose[:1].expand_as(pose).clone()
        zero_anchor = self._codec().encode(
            zero_pose, torch.full_like(gripper, 0.5)
        ).unsqueeze(0)
        for mode in ("robust_tilewise", "robust_block_consensus", "block_position_joint_rotation"):
            for anchor_alpha in (0.0, 0.5, 1.0):
                with self.subTest(mode=mode, anchor_alpha=anchor_alpha):
                    codec = self._codec(
                        mode, decode_anchor_alpha=anchor_alpha
                    )
                    if anchor_alpha:
                        codec.set_decode_anchor_video(zero_anchor)
                    decoded_pose, decoded_gripper = codec.decode(video, pose[0])
                    torch.testing.assert_close(
                        decoded_pose[:, :3], pose[:, :3], atol=2e-5, rtol=0
                    )
                    torch.testing.assert_close(
                        quaternion_wxyz_to_matrix(decoded_pose[:, 3:]),
                        quaternion_wxyz_to_matrix(pose[:, 3:]),
                        atol=2e-5,
                        rtol=0,
                    )
                    torch.testing.assert_close(
                        decoded_gripper, gripper, atol=1e-6, rtol=0
                    )

    def test_template_anchor_removes_corrupt_frame_zero_reference(self) -> None:
        pose, gripper = self._trajectory()
        video = self._codec().encode(pose, gripper)
        zero_pose = pose[:1].expand_as(pose).clone()
        zero_gripper = torch.full_like(gripper, 0.5)
        clean_anchor = self._codec().encode(
            zero_pose, zero_gripper
        ).unsqueeze(0)
        cfg = self._codec().config
        center_mask, _, direction_mask = _center_and_read_masks(
            cfg.tile_height,
            cfg.tile_width,
            center_frac=cfg.center_frac,
            boundary_margin=0,
            outer_margin=0,
            device=video.device,
        )
        rotation = quaternion_wxyz_to_matrix(
            axis_angle_to_quaternion_wxyz(torch.tensor([[0.0, 0.0, 0.25]]))
        )[0]
        canonical = self._codec()._canonical_directions(video.device, video.dtype)
        corrupt_direction = torch.einsum("ij,hwj->ihw", rotation, canonical)
        for x_offset in (0, cfg.tile_width):
            frame_zero = video[
                :, 0, :, x_offset : x_offset + cfg.tile_width
            ]
            frame_zero[:, direction_mask] = corrupt_direction[:, direction_mask]
            frame_zero[:, center_mask] = 0.05

        unanchored, _ = self._codec("robust_joint").decode(video, pose[0])
        anchored_codec = self._codec(
            "robust_joint", decode_anchor_alpha=1.0
        )
        anchored_codec.set_decode_anchor_video(clean_anchor)
        anchored, _ = anchored_codec.decode(video, pose[0])
        expected_rotation = quaternion_wxyz_to_matrix(pose[:, 3:])
        anchored_rotation = quaternion_wxyz_to_matrix(anchored[:, 3:])
        torch.testing.assert_close(
            anchored[1:, :3], pose[1:, :3], atol=2e-5, rtol=0
        )
        torch.testing.assert_close(
            anchored_rotation[1:], expected_rotation[1:], atol=2e-5, rtol=0
        )
        self.assertGreater(
            (unanchored[1:, :3] - pose[1:, :3]).norm(dim=-1).mean().item(),
            0.05,
        )

    def test_default_decoder_is_exactly_legacy(self) -> None:
        pose, gripper = self._trajectory()
        video = self._codec().encode(pose, gripper)
        video = video + torch.randn_like(video) * 0.01
        default_pose, default_gripper = self._codec().decode(video, pose[0])
        legacy_pose, legacy_gripper = self._codec("legacy").decode(video, pose[0])
        torch.testing.assert_close(default_pose, legacy_pose, atol=0, rtol=0)
        torch.testing.assert_close(default_gripper, legacy_gripper, atol=0, rtol=0)

    def test_decode_mode_is_not_checkpoint_geometry_metadata(self) -> None:
        self.assertEqual(
            self._codec("legacy").metadata(),
            self._codec("robust_joint").metadata(),
        )
        self.assertEqual(
            self._codec("legacy").metadata(),
            self._codec(
                "robust_block_consensus", decode_anchor_alpha=0.5
            ).metadata(),
        )

    def test_unknown_decode_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported Rothko decode_mode"):
            self._codec("unknown")

    def test_invalid_inference_decoder_options_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "decode_anchor_alpha"):
            self._codec("robust_joint", decode_anchor_alpha=1.1)
        with self.assertRaisesRegex(ValueError, "decode_block_grid"):
            self._codec("robust_block_consensus", decode_block_grid=1)

    def test_robust_joint_downweights_localized_duplicate_outliers(self) -> None:
        pose, gripper = self._trajectory()
        video = self._codec().encode(pose, gripper)
        cfg = self._codec().config
        _, _, direction_mask = _center_and_read_masks(
            cfg.tile_height,
            cfg.tile_width,
            center_frac=cfg.center_frac,
            boundary_margin=cfg.boundary_margin,
            outer_margin=cfg.outer_margin,
            device=video.device,
        )
        right = video[:, :, :, cfg.tile_width :]
        right_flat = right.permute(1, 0, 2, 3).reshape(pose.shape[0], 3, -1)
        outliers = torch.where(direction_mask.reshape(-1))[0][::3]
        right_flat[1:, 0, outliers] = 1.0
        right_flat[1:, 1, outliers] = -1.0
        right_flat[1:, 2, outliers] = 0.2
        right.copy_(
            right_flat.reshape(
                pose.shape[0], 3, cfg.tile_height, cfg.tile_width
            ).permute(1, 0, 2, 3)
        )

        expected = quaternion_wxyz_to_matrix(pose[:, 3:])
        rotation_errors = {}
        for mode in ("legacy", "robust_joint"):
            decoded, _ = self._codec(mode).decode(video, pose[0])
            actual = quaternion_wxyz_to_matrix(decoded[:, 3:])
            relative = actual.transpose(-1, -2) @ expected
            cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(
                -1, 1
            )
            rotation_errors[mode] = torch.acos(cosine) * 180.0 / math.pi
        self.assertGreater(rotation_errors["legacy"].mean().item(), 5.0)
        self.assertLess(rotation_errors["robust_joint"].max().item(), 1.0)

    def test_quaternion_sign_is_equivalent(self) -> None:
        codec = self._codec()
        pose = torch.tensor(
            [
                [0, 0, 0, 1, 0, 0, 0],
                [0.01, 0, 0, 0, 1, 0, 0],
            ],
            dtype=torch.float32,
        )
        gripper = torch.ones(2, 1)
        positive = codec.encode(pose, gripper)
        pose[1, 3:] *= -1
        negative = codec.encode(pose, gripper)
        torch.testing.assert_close(positive, negative)

    def test_stats_horizon_mismatch_is_rejected(self) -> None:
        config = LiberoRothkoCodecConfig()
        lo = torch.full((1, 3, config.image_height, config.image_width), -1.0)
        stats = RothkoNormStats(
            lo=lo,
            hi=-lo,
            metadata={
                "environment": "libero",
                "layout": "single_arm_duplicated_horizontal",
                "action_horizon": 32,
            },
        )
        with self.assertRaisesRegex(ValueError, "action_horizon"):
            LiberoRothkoCodec(
                config,
                stats,
                expected_action_horizon=16,
            )


if __name__ == "__main__":
    unittest.main()
