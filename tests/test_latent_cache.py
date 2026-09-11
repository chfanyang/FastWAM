import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from fastwam.datasets.latent_cache import (
    LATENT_CACHE_DTYPE,
    LATENT_CACHE_VERSION,
    LatentCacheReader,
)
from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from fastwam.models.wan22.fastwam_visual_action import FastWAMVideoOnlyRaymap
from fastwam.representations.rothko import RothkoNormStats


class _DummyVideoExpert(torch.nn.Module):
    video_attention_mask_mode = "rgb_then_raymap_block_causal"

    def __init__(self) -> None:
        super().__init__()
        self.in_dim = 16
        self.patch_size = (1, 2, 2)
        self.head = torch.nn.Module()
        self.head.head = torch.nn.Linear(1, 64, bias=False)


class _NoEncodeVae(torch.nn.Module):
    temporal_downsample_factor = 4
    upsampling_factor = 8

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.z_dim = 16

    def encode(self, *args, **kwargs):
        raise AssertionError("VAE encode must not run when cached latents are present")


class LatentCacheTest(unittest.TestCase):
    def test_sharded_bfloat16_round_trip_is_bit_exact(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            latent_shape = (3, 2, 2, 2)
            expected = torch.randn(5, 2, *latent_shape, dtype=torch.bfloat16)
            raw = expected.contiguous().view(torch.uint16).numpy()
            shard_path = root / "latents-00000-of-00001.bin"
            memmap = np.memmap(
                shard_path,
                mode="w+",
                dtype=np.uint16,
                shape=raw.shape,
            )
            memmap[:] = raw
            memmap.flush()
            del memmap
            contract = {"identity": "unit-test"}
            metadata = {
                "cache_version": LATENT_CACHE_VERSION,
                "complete": True,
                "dtype": LATENT_CACHE_DTYPE,
                "num_samples": len(expected),
                "latent_shape": list(latent_shape),
                "dataset_contract": contract,
                "shards": [
                    {
                        "index": 0,
                        "start": 0,
                        "end": len(expected),
                        "file": shard_path.name,
                    }
                ],
            }
            (root / "metadata.json").write_text(json.dumps(metadata))
            (root / "_SUCCESS").write_text("complete\n")

            reader = LatentCacheReader(
                root, expected_dataset_contract=contract
            )
            for index in range(len(expected)):
                rgb, raymap = reader[index]
                self.assertTrue(torch.equal(rgb, expected[index, 0]))
                self.assertTrue(torch.equal(raymap, expected[index, 1]))
            reader._maps.clear()

    def test_reader_rejects_incomplete_or_mismatched_cache(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            metadata = {
                "cache_version": LATENT_CACHE_VERSION,
                "complete": True,
                "dtype": LATENT_CACHE_DTYPE,
                "num_samples": 1,
                "latent_shape": [1, 1, 1, 1],
                "dataset_contract": {"identity": "expected"},
                "shards": [
                    {
                        "index": 0,
                        "start": 0,
                        "end": 1,
                        "file": "latents-00000-of-00001.bin",
                    }
                ],
            }
            (root / "metadata.json").write_text(json.dumps(metadata))
            with self.assertRaises(FileNotFoundError):
                LatentCacheReader(
                    root,
                    expected_dataset_contract={"identity": "expected"},
                )

            (root / "_SUCCESS").write_text("complete\n")
            np.zeros((1, 2, 1, 1, 1, 1), dtype=np.uint16).tofile(
                root / "latents-00000-of-00001.bin"
            )
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                LatentCacheReader(
                    root,
                    expected_dataset_contract={"identity": "different"},
                )

    def test_raise_mode_never_substitutes_a_random_base_sample(self) -> None:
        class _BrokenDataset:
            num_frames = 1

            def __getitem__(self, index):
                raise OSError(f"broken sample {index}")

        dataset = BaseLerobotDataset.__new__(BaseLerobotDataset)
        dataset.multi_dataset = _BrokenDataset()
        dataset.sample_error_mode = "raise"
        with self.assertRaisesRegex(RuntimeError, "Error loading sample 0"):
            dataset[0]

    def test_cache_lookup_follows_fallback_sample_index(self) -> None:
        class _FallbackDataset:
            def __getitem__(self, index):
                # Simulate BaseLerobotDataset replacing requested index 2 with
                # a readable sample at index 7.
                return {"idx": 7}

        dataset = RobotVideoDataset.__new__(RobotVideoDataset)
        dataset.lerobot_dataset = _FallbackDataset()
        dataset.max_padding_retry = 0
        dataset.skip_padding_as_possible = False
        dataset.latent_cache_only = True
        dataset._get_latent_cache_only = mock.Mock(return_value="sample-7")

        result = dataset._get(2)

        self.assertEqual(result, "sample-7")
        dataset._get_latent_cache_only.assert_called_once_with(
            7, {"idx": 7}
        )

    def test_text_context_is_loaded_from_disk_only_once_per_dataset(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            prompt = "test instruction"
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            path = (
                Path(temporary)
                / f"{digest}.t5_len4.wan22ti2v5b.pt"
            )
            expected_context = torch.randn(4, 8, dtype=torch.bfloat16)
            expected_mask = torch.tensor([True, True, False, False])
            torch.save(
                {"context": expected_context, "mask": expected_mask}, path
            )

            dataset = RobotVideoDataset.__new__(RobotVideoDataset)
            dataset.text_embedding_cache_dir = temporary
            dataset.context_len = 4
            dataset._warned_legacy_text_cache = False
            dataset._text_context_memory_cache = {}
            dataset.text_context_cache_max_entries = None
            with mock.patch("torch.load", wraps=torch.load) as load:
                context0, mask0 = dataset._get_cached_text_context(prompt)
                context1, mask1 = dataset._get_cached_text_context(prompt)
            self.assertEqual(load.call_count, 1)
            self.assertIs(context0, context1)
            self.assertIs(mask0, mask1)
            self.assertTrue(torch.equal(context0, expected_context))
            self.assertTrue(torch.equal(mask0, expected_mask))

    def test_model_uses_cached_latents_without_calling_vae(self) -> None:
        lo = torch.full((1, 3, 224, 448), -1.0)
        stats = RothkoNormStats(
            lo=lo,
            hi=-lo,
            metadata={
                "environment": "libero",
                "layout": "single_arm_duplicated_horizontal",
            },
        )
        model = FastWAMVideoOnlyRaymap(
            video_expert=_DummyVideoExpert(),
            vae=_NoEncodeVae(),
            text_dim=16,
            device="cpu",
            torch_dtype=torch.bfloat16,
            action_horizon=16,
            raymap_representation="libero_rothko",
            rothko_norm_stats=stats,
            rothko_config={
                "image_height": 224,
                "image_width": 448,
                "tile_height": 224,
                "tile_width": 224,
            },
        )
        batch_size = 2
        rgb_latents = torch.randn(
            batch_size, 16, 5, 28, 56, dtype=torch.bfloat16
        )
        raymap_latents = torch.randn_like(rgb_latents)
        sample = {
            "video": torch.zeros(batch_size, 3, 17, 224, 448),
            "raymap": torch.zeros(batch_size, 3, 17, 224, 448),
            "context": torch.zeros(batch_size, 4, 16),
            "context_mask": torch.ones(batch_size, 4, dtype=torch.bool),
            "rgb_latents": rgb_latents,
            "raymap_latents": raymap_latents,
        }
        inputs = model.build_inputs(sample)
        self.assertTrue(torch.equal(inputs["rgb_latents"], rgb_latents))
        self.assertTrue(torch.equal(inputs["raymap_latents"], raymap_latents))

        cache_only_sample = dict(sample)
        cache_only_sample.pop("video")
        cache_only_sample.pop("raymap")
        cache_only_inputs = model.build_inputs(cache_only_sample)
        self.assertTrue(
            torch.equal(cache_only_inputs["rgb_latents"], rgb_latents)
        )
        self.assertTrue(
            torch.equal(cache_only_inputs["raymap_latents"], raymap_latents)
        )


if __name__ == "__main__":
    unittest.main()
