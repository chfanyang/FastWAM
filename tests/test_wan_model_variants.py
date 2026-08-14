from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastwam.models.wan22.helpers.loader import (  # noqa: E402
    WAN22_MODEL_REGISTRY,
    _resolve_configs,
    resolve_wan_model_spec,
)
from fastwam.models.wan22.wan_video_vae import (  # noqa: E402
    WanVideoVAE,
    WanVideoVAE38,
)


class WanModelVariantTest(unittest.TestCase):
    def test_wan21_1_3b_spec_uses_native_16_channel_vae(self):
        spec = resolve_wan_model_spec(
            "Wan-AI/Wan2.1-T2V-1.3B", "wan2.1-t2v-1.3b"
        )
        self.assertEqual(spec.vae_filename, "Wan2.1_VAE.pth")
        self.assertIs(spec.vae_class, WanVideoVAE)

    def test_wan22_5b_spec_preserves_native_48_channel_vae(self):
        spec = resolve_wan_model_spec(
            "Wan-AI/Wan2.2-TI2V-5B", "wan2.2-ti2v-5b"
        )
        self.assertEqual(spec.vae_filename, "Wan2.2_VAE.pth")
        self.assertIs(spec.vae_class, WanVideoVAE38)

    def test_model_id_and_variant_must_agree(self):
        with self.assertRaisesRegex(ValueError, "model_id/model_variant mismatch"):
            resolve_wan_model_spec(
                "Wan-AI/Wan2.2-TI2V-5B", "wan2.1-t2v-1.3b"
            )

    def test_wan21_resolver_keeps_native_vae_and_redirects_shared_text_encoder(self):
        _, text, vae, _ = _resolve_configs(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
            redirect_common_files=True,
            model_variant="wan2.1-t2v-1.3b",
        )
        self.assertEqual(
            text.model_id,
            "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
        )
        self.assertEqual(
            text.origin_file_pattern,
            "models_t5_umt5-xxl-enc-bf16.safetensors",
        )
        self.assertEqual(vae.model_id, "Wan-AI/Wan2.1-T2V-1.3B")
        self.assertEqual(vae.origin_file_pattern, "Wan2.1_VAE.pth")

    def test_official_wan21_hashes_are_registered(self):
        registrations = {
            (entry["model_hash"], entry["model_name"])
            for entry in WAN22_MODEL_REGISTRY
        }
        self.assertIn(
            ("9269f8db9040a9d860eaca435be61814", "wan_video_dit"),
            registrations,
        )
        self.assertIn(
            ("ccc42284ea13e1ad04693284c7a09be6", "wan_video_vae"),
            registrations,
        )


if __name__ == "__main__":
    unittest.main()
