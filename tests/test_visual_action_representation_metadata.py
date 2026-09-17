import unittest
from dataclasses import replace

import torch

from fastwam.models.wan22.fastwam_visual_action import FastWAMVideoOnlyRaymap
from fastwam.representations.rothko import RothkoNormStats
from fastwam.representations.libero_rothko import LiberoRothkoCodec


class _DummyVideoExpert(torch.nn.Module):
    video_attention_mask_mode = "rgb_then_raymap_block_causal"

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.in_dim = 16
        self.patch_size = (1, 2, 2)
        self.head = torch.nn.Module()
        self.head.head = torch.nn.Linear(1, 16 * 4, bias=False)


class _DummyVae(torch.nn.Module):
    temporal_downsample_factor = 4
    upsampling_factor = 8

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.z_dim = 16


def _stats(height: int, width: int, metadata: dict) -> RothkoNormStats:
    lo = torch.full((1, 3, height, width), -1.0)
    return RothkoNormStats(lo=lo, hi=-lo, metadata=metadata)


class VisualActionRepresentationMetadataTest(unittest.TestCase):
    def test_absolute_ray0_checkpoint_contract(self):
        relative = self._model("libero_rothko")
        absolute = self._model("libero_rothko")
        codec = absolute.raymap_codec
        absolute.raymap_codec = LiberoRothkoCodec(
            replace(codec.config, frame0_pose_mode="absolute",
                    absolute_position_min=(-1., -1., 0.),
                    absolute_position_max=(1., 1., 2.)), codec.norm_stats)
        old = relative._visual_action_checkpoint_config()
        new = absolute._visual_action_checkpoint_config()
        absolute._validate_visual_action_checkpoint_config(new, checkpoint_path="new.pt")
        relative._validate_visual_action_checkpoint_config(old, checkpoint_path="old.pt")
        for model, config in ((relative, new), (absolute, old)):
            with self.assertRaisesRegex(ValueError, "frame0_pose_mode mismatch"):
                model._validate_visual_action_checkpoint_config(config, checkpoint_path="mismatch.pt")
        for key in ("absolute_position_min", "absolute_position_max"):
            missing = dict(new); missing.pop(key)
            with self.assertRaisesRegex(ValueError, "checkpoint missing"):
                absolute._validate_visual_action_checkpoint_config(missing, checkpoint_path="missing.pt")
        changed = dict(new); changed["absolute_position_min"] = [-2., -1., 0.]
        with self.assertRaisesRegex(ValueError, "codec mismatch"):
            absolute._validate_visual_action_checkpoint_config(changed, checkpoint_path="bounds.pt")

    def _model(
        self,
        representation: str,
        action_horizon: int = 16,
        *,
        center_frac: float = 0.5,
        stats_offset: float = 0.0,
        future_rgb_mode: str = "joint",
        rothko_decode_mode: str = "legacy",
    ) -> FastWAMVideoOnlyRaymap:
        if representation == "rothko":
            stats = _stats(384, 320, {})
            config = None
        else:
            stats = _stats(
                224,
                448,
                {
                    "environment": "vlabench" if representation == "vlabench_rothko" else "libero",
                    "layout": "single_arm_duplicated_horizontal",
                },
            )
            config = {
                "image_height": 224,
                "image_width": 448,
                "tile_height": 224,
                "tile_width": 224,
                "center_frac": center_frac,
            }
        if stats_offset:
            stats.lo = stats.lo + stats_offset
            stats.hi = stats.hi + stats_offset
        video_expert = _DummyVideoExpert()
        if future_rgb_mode == "train_only_auxiliary":
            video_expert.video_attention_mask_mode = "independent_rgb_aux_ray"
        return FastWAMVideoOnlyRaymap(
            video_expert=video_expert,
            vae=_DummyVae(),
            text_dim=16,
            device="cpu",
            action_horizon=action_horizon,
            raymap_representation=representation,
            rothko_norm_stats=stats,
            rothko_config=config,
            rothko_decode_mode=rothko_decode_mode,
            future_rgb_mode=future_rgb_mode,
        )

    def test_new_checkpoint_metadata_is_environment_specific(self) -> None:
        robotwin = self._model("rothko")._visual_action_checkpoint_config()
        libero = self._model("libero_rothko")._visual_action_checkpoint_config()
        self.assertEqual(robotwin["raymap_representation"], "rothko")
        self.assertEqual(libero["raymap_representation"], "libero_rothko")
        self.assertEqual(libero["environment"], "libero")

    def test_vlabench_is_opt_in_and_rejects_libero_checkpoint(self):
        model = self._model("vlabench_rothko")
        metadata = model._visual_action_checkpoint_config()
        self.assertEqual(metadata["environment"], "vlabench")
        self.assertEqual(metadata["raymap_representation"], "vlabench_rothko")
        model._validate_visual_action_checkpoint_config(metadata, checkpoint_path="vlabench.pt")
        with self.assertRaises(ValueError):
            model._validate_visual_action_checkpoint_config(
                self._model("libero_rothko")._visual_action_checkpoint_config(),
                checkpoint_path="libero.pt")
        with self.assertRaises(ValueError):
            self._model("libero_rothko")._validate_visual_action_checkpoint_config(
                metadata, checkpoint_path="vlabench.pt")

    def test_legacy_robotwin_checkpoint_cannot_load_as_libero(self) -> None:
        robotwin = self._model("rothko")
        legacy = robotwin._visual_action_checkpoint_config()
        legacy.pop("raymap_representation")
        robotwin._validate_visual_action_checkpoint_config(
            legacy, checkpoint_path="/tmp/legacy.pt"
        )
        with self.assertRaisesRegex(ValueError, "representation mismatch"):
            self._model("libero_rothko")._validate_visual_action_checkpoint_config(
                legacy, checkpoint_path="/tmp/legacy.pt"
            )

    def test_horizon_controls_latent_layout_and_metadata(self) -> None:
        horizon16 = self._model("libero_rothko", action_horizon=16)
        self.assertEqual(horizon16.num_pixel_frames, 17)
        self.assertEqual(horizon16.num_latent_frames_per_modality, 5)
        self.assertEqual(horizon16.condition_latent_indices, (0, 5))
        self.assertEqual(horizon16.temporal_rope_mode, "continuous_0_9")

        horizon32 = self._model("libero_rothko", action_horizon=32)
        self.assertEqual(horizon32.num_pixel_frames, 33)
        self.assertEqual(horizon32.num_latent_frames_per_modality, 9)
        self.assertEqual(horizon32.condition_latent_indices, (0, 9))
        self.assertEqual(horizon32.temporal_rope_mode, "continuous_0_17")
        metadata = horizon32._visual_action_checkpoint_config()
        self.assertEqual(metadata["action_horizon"], 32)
        self.assertEqual(metadata["temporal_rope_mode"], "continuous_0_17")

    def test_horizon_must_align_with_vae_temporal_factor(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive multiple"):
            self._model("libero_rothko", action_horizon=18)

    def test_checkpoint_rejects_codec_geometry_drift(self) -> None:
        trained = self._model("libero_rothko", center_frac=0.5)
        checkpoint_config = trained._visual_action_checkpoint_config()
        changed = self._model("libero_rothko", center_frac=0.6)
        with self.assertRaisesRegex(ValueError, "codec mismatch for center_frac"):
            changed._validate_visual_action_checkpoint_config(
                checkpoint_config, checkpoint_path="/tmp/geometry_drift.pt"
            )

    def test_decoder_mode_does_not_change_checkpoint_contract(self) -> None:
        legacy = self._model(
            "libero_rothko", rothko_decode_mode="legacy"
        )._visual_action_checkpoint_config()
        robust = self._model(
            "libero_rothko", rothko_decode_mode="robust_joint"
        )._visual_action_checkpoint_config()
        self.assertEqual(legacy, robust)

    def test_checkpoint_rejects_norm_stats_content_drift(self) -> None:
        trained = self._model("libero_rothko")
        checkpoint_config = trained._visual_action_checkpoint_config()
        changed = self._model("libero_rothko", stats_offset=0.1)
        with self.assertRaisesRegex(ValueError, "normalization stats mismatch"):
            changed._validate_visual_action_checkpoint_config(
                checkpoint_config, checkpoint_path="/tmp/stats_drift.pt"
            )

    def test_checkpoint_rejects_vae_content_drift(self) -> None:
        trained = self._model("libero_rothko")
        trained._vae_identity_cache = {
            "kind": "original_wan22",
            "filename": "Wan2.2_VAE.safetensors",
            "sha256": "trained",
        }
        checkpoint_config = trained._visual_action_checkpoint_config()
        changed = self._model("libero_rothko")
        changed._vae_identity_cache = {
            "kind": "custom",
            "filename": "custom.safetensors",
            "sha256": "changed",
        }
        with self.assertRaisesRegex(ValueError, "Checkpoint VAE mismatch"):
            changed._validate_visual_action_checkpoint_config(
                checkpoint_config, checkpoint_path="/tmp/vae_drift.pt"
            )

    def test_checkpoint_separates_joint_and_training_only_future_rgb(self) -> None:
        joint = self._model("libero_rothko")
        auxiliary = self._model(
            "libero_rothko", future_rgb_mode="train_only_auxiliary"
        )
        joint_config = joint._visual_action_checkpoint_config()
        auxiliary_config = auxiliary._visual_action_checkpoint_config()
        joint_config["video_attention_mask_mode"] = "independent_rgb_aux_ray"
        with self.assertRaisesRegex(ValueError, "future RGB mode mismatch"):
            auxiliary._validate_visual_action_checkpoint_config(
                joint_config, checkpoint_path="/tmp/joint.pt"
            )
        auxiliary_config["video_attention_mask_mode"] = (
            "rgb_then_raymap_block_causal"
        )
        with self.assertRaisesRegex(ValueError, "future RGB mode mismatch"):
            joint._validate_visual_action_checkpoint_config(
                auxiliary_config, checkpoint_path="/tmp/auxiliary.pt"
            )


if __name__ == "__main__":
    unittest.main()
