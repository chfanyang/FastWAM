import unittest

import torch

from fastwam.models.wan22.fastwam_visual_action import FastWAMVideoOnlyRaymap
from fastwam.representations.rothko import RothkoNormStats


class _DummyVideoExpert(torch.nn.Module):
    video_attention_mask_mode = "rgb_then_raymap_block_causal"

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))


class _DummyVae(torch.nn.Module):
    temporal_downsample_factor = 4
    upsampling_factor = 8


def _stats(height: int, width: int, metadata: dict) -> RothkoNormStats:
    lo = torch.full((1, 3, height, width), -1.0)
    return RothkoNormStats(lo=lo, hi=-lo, metadata=metadata)


class VisualActionRepresentationMetadataTest(unittest.TestCase):
    def _model(
        self,
        representation: str,
        action_horizon: int = 16,
        *,
        center_frac: float = 0.5,
        stats_offset: float = 0.0,
    ) -> FastWAMVideoOnlyRaymap:
        if representation == "rothko":
            stats = _stats(384, 320, {})
            config = None
        else:
            stats = _stats(
                224,
                448,
                {
                    "environment": "libero",
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
        return FastWAMVideoOnlyRaymap(
            video_expert=_DummyVideoExpert(),
            vae=_DummyVae(),
            text_dim=16,
            device="cpu",
            action_horizon=action_horizon,
            raymap_representation=representation,
            rothko_norm_stats=stats,
            rothko_config=config,
        )

    def test_new_checkpoint_metadata_is_environment_specific(self) -> None:
        robotwin = self._model("rothko")._visual_action_checkpoint_config()
        libero = self._model("libero_rothko")._visual_action_checkpoint_config()
        self.assertEqual(robotwin["raymap_representation"], "rothko")
        self.assertEqual(libero["raymap_representation"], "libero_rothko")
        self.assertEqual(libero["environment"], "libero")

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


if __name__ == "__main__":
    unittest.main()
