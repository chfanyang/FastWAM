"""CPU-only tests; no model weights, camera drivers or robot connections."""
import copy
import gc
import hashlib
import importlib.util
import json
import pickle
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageOps
import torch

from fastwam.realworld.common import (CAMERAS, PREPROCESSING, preprocess_rgb, rgb_canvas,
    xyzw_to_wxyz, wxyz_to_xyzw, gripper_open, sha256_file, image_inventory_fingerprint)
from fastwam.realworld.dataset import RealWorldVideoDataset, load_context
from fastwam.realworld.common import PROMPT_TEMPLATE
from fastwam.realworld.execution import checked_chunk, execute_chunk, rotation_distance
from fastwam.realworld.protocol import pack, unpack, VERSION, ACTION_SPACE
from fastwam.realworld.policy import RealWorldPolicy

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "data/realworld_artifacts/ray0_absolute_left_wrist_val2_g08_v2"


class ConventionsTest(unittest.TestCase):
    def test_complete_text_cache_is_read_and_modified_file_rejected(self):
        instruction = "Pick up the pen and insert it into the holder."
        prompt = PROMPT_TEMPLATE.format(task=instruction)
        identity = dict(context_len=128, encoder_sha256="test-encoder")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / (hashlib.sha256(prompt.encode()).hexdigest() + ".pt")
            payload = dict(context=torch.zeros(128, 4096, dtype=torch.bfloat16),
                context_mask=torch.ones(128, dtype=torch.bool), metadata={**identity, "prompt": prompt})
            torch.save(payload, path)
            (root / "manifest.json").write_text(json.dumps(dict(identity=identity, files={path.name: sha256_file(path)})))
            with self.assertRaisesRegex(ValueError, "Incomplete text cache"):
                load_context(root, instruction)
            (root / "_SUCCESS").write_text("complete\n")
            context, mask = load_context(root, instruction)
            torch.testing.assert_close(context, payload["context"])
            self.assertTrue(mask.all())
            with path.open("ab") as stream:
                stream.write(b"modified")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_context(root, instruction)

    def test_preflight_rejects_changed_source_images(self):
        spec = importlib.util.spec_from_file_location("realworld_preflight", ROOT / "scripts/realworld/preflight.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ep = root / "test/episode_00000"
            for camera in ("right_camera", "left_camera", "wrist_camera"):
                folder = ep / "image" / camera
                folder.mkdir(parents=True)
                Image.new("RGB", (224, 224)).save(folder / "000000.png")
            (ep / "states.npz").write_bytes(b"test fixture")
            record = dict(path="test/episode_00000", length=1,
                states_sha256=sha256_file(ep / "states.npz"),
                image_inventory_sha256=image_inventory_fingerprint(ep, 1))
            module.check_sources(root, {"episodes": [record]})
            path = ep / "image/left_camera/000000.png"
            path.write_bytes(path.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "Source images changed"):
                module.check_sources(root, {"episodes": [record]})

    def test_camera_order_and_resize_match_original(self):
        rng = np.random.default_rng(3)
        rgb = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "src", Path(tmp) / "dst"
            src.mkdir()
            Image.fromarray(rgb).save(src / "frame.png")
            spec = importlib.util.spec_from_file_location("original_resize", ROOT / "third_party/FastAPI/resize_stack_blocks.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            count, failures = module.process_chunk([src / "frame.png"], src, dst, (224, 224))
            self.assertEqual((count, failures), (1, []))
            reference = np.asarray(Image.open(dst / "frame.png"))
            np.testing.assert_array_equal(preprocess_rgb(rgb[..., ::-1], color_order="BGR"), reference)
            np.testing.assert_array_equal(preprocess_rgb(reference), reference)
        left = np.full((224, 224, 3), [255, 0, 0], dtype=np.uint8)
        wrist = np.full((224, 224, 3), [0, 0, 255], dtype=np.uint8)
        canvas = rgb_canvas(dict(left_camera=left, wrist_camera=wrist))
        np.testing.assert_array_equal(canvas[:, :224], left)
        np.testing.assert_array_equal(canvas[:, 224:], wrist)

    def test_quaternion_and_gripper_conventions(self):
        raw = np.array([.5, 0, .25, 1, 0, 0, 0], np.float32)
        np.testing.assert_array_equal(xyzw_to_wxyz(raw), [.5, 0, .25, 0, 1, 0, 0])
        np.testing.assert_array_equal(wxyz_to_xyzw(xyzw_to_wxyz(raw)), raw)
        np.testing.assert_array_equal(gripper_open([.015, .042, .055, .0607, .0652, .085]), [0, 0, 0, 0, 0, 1])
        with self.assertRaises(ValueError):
            xyzw_to_wxyz(np.zeros(7))

    def test_wire_compatibility_with_existing_openpi_client(self):
        path = ROOT / "third_party/FastAPI/package/src/openpi_client/msgpack_numpy.py"
        spec = importlib.util.spec_from_file_location("original_wire", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        value = dict(request_id=4, pose=np.arange(7, dtype=np.float32), images=np.zeros((4, 5, 3), np.uint8))
        for restored in (module.unpackb(pack(value)), unpack(module.packb(value))):
            np.testing.assert_array_equal(restored["pose"], value["pose"])
            np.testing.assert_array_equal(restored["images"], value["images"])


class ExecutionTest(unittest.TestCase):
    def setUp(self):
        self.pose = np.array([.5, 0, .25, 1, 0, 0, 0], np.float32)
        self.response = dict(protocol_version=VERSION, request_id=2, action_space=ACTION_SPACE,
            action_dt=.1, target_pose_wxyz=np.tile(self.pose, (16, 1)), gripper_open=np.ones(16))
        self.kw = dict(position_min=[.3, -.4, .1], position_max=[.8, .4, .5])

    def test_rejects_wrong_frame_stale_reply_nan_and_jumps(self):
        poses, grip = checked_chunk(self.response, 2, self.pose, **self.kw)
        self.assertEqual(poses.shape, (4, 7))
        for key, bad in (("request_id", 1), ("action_space", "joint"), ("action_dt", .05)):
            response = {**self.response, key: bad}
            with self.assertRaises(ValueError):
                checked_chunk(response, 2, self.pose, **self.kw)
        for x in (np.nan, .7, .9):
            response = copy.deepcopy(self.response)
            response["target_pose_wxyz"][0, 0] = x
            with self.assertRaises(ValueError):
                checked_chunk(response, 2, self.pose, **self.kw)

    def test_gripper_is_not_executed_when_future_pose_is_enqueued(self):
        events, now = [], [0.]
        class Stop:
            def is_set(self): return False
            def wait(self, seconds):
                events.append(("wait", seconds))
                now[0] += seconds
                return False
        arm = SimpleNamespace(schedule_waypoint=lambda p, t: events.append(("arm", p, t)))
        gripper = SimpleNamespace(grasp=lambda **_: events.append(("close", now[0])),
                                 release=lambda: events.append(("open", now[0])))
        targets = np.tile(self.pose, (3, 1))
        result = execute_chunk(arm, gripper, targets, [0., 0., 1.], self.pose, True, Stop(),
            clock=lambda: now[0], wall_clock=lambda: 100 + now[0])
        self.assertTrue(result)
        self.assertEqual([e[0] for e in events], ["arm", "arm", "arm", "wait", "close", "wait", "wait", "open"])
        # Startup lead once, then exactly 10 Hz without cumulative 30 ms drift.
        np.testing.assert_allclose([e[2] for e in events if e[0] == "arm"], [100.13, 100.23, 100.33])
        self.assertAlmostEqual(now[0], .33)
        np.testing.assert_array_equal(events[0][1][3:], [0, 0, 0, 1])
        stopped = threading.Event(); stopped.set()
        events.clear()
        execute_chunk(arm, gripper, targets, [0, 0, 0], self.pose, True, stopped)
        self.assertEqual(events, [])

    def test_identical_quaternion_has_no_artificial_rotation(self):
        pose = np.array([.5, 0, .25, .323, .512, .691, .395])
        pose[3:] /= np.linalg.norm(pose[3:])
        self.assertLess(float(rotation_distance(pose, pose)), 1e-7)


@unittest.skipUnless((ARTIFACTS / "split.json").exists(), "Run CPU preparation first")
class DatasetIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.kw = dict(dataset_dir=ROOT / "data/realworld_data", split_manifest=ARTIFACTS / "split.json",
            rothko_norm_stats=ARTIFACTS / "rothko_stats.pt", include_text_context=False)
        cls.train = RealWorldVideoDataset(**cls.kw)
        cls.val = RealWorldVideoDataset(**cls.kw, split="val")

    def test_split_and_hydra_contract(self):
        train = {r["path"] for r in self.train.records}
        val = {r["path"] for r in self.val.records}
        self.assertFalse(train & val)
        self.assertEqual((len(train), len(val)), (371, 8))
        self.assertEqual(len(self.train) + len(self.val), 99012)
        for task in {r["task"] for r in self.val.records}:
            self.assertEqual(sum(r["task"] == task for r in self.val.records), 2)
        self.assertEqual(self.train.camera_keys, CAMERAS)
        self.assertEqual(self.train.codec.config.frame0_pose_mode, "absolute")
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from fastwam.utils.config_resolvers import register_default_resolvers
        register_default_resolvers()
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
            cfg = compose(config_name="train", overrides=["task=realworld_all4_ray0_absolute_wan21_ep8",
                "data.train.include_text_context=false"])
            OmegaConf.resolve(cfg)
        self.assertEqual(cfg.num_epochs, 8)
        self.assertEqual(cfg.model.video_dit_config.action_dim, 8)
        self.assertEqual(cfg.model._target_, "fastwam.realworld.runtime.create_model")
        self.assertEqual(len(instantiate(cfg.data.train)), len(self.train))
        from fastwam.runtime import _validate_visual_action_data_contract
        model = SimpleNamespace(raymap_codec=self.train.codec, action_horizon=16)
        _validate_visual_action_data_contract(model, self.train, self.val)

    def test_spawn_worker_can_read_dataset_after_cache_warmup(self):
        self.train._episode(0)
        restored = pickle.loads(pickle.dumps(self.train))
        self.assertEqual(len(restored), len(self.train))
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(restored, [0]),
            batch_size=1, num_workers=1, multiprocessing_context="spawn", prefetch_factor=1,
            timeout=30)
        batch = next(iter(loader))
        self.assertEqual(batch["video"].shape, (1, 3, 17, 224, 448))

    def test_cached_dataset_reads_bf16_shards_without_loading_pngs(self):
        # Real NPZ source, a two-window fixture manifest, and a genuine BF16 shard.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = copy.deepcopy(self.train.manifest)
            record = copy.deepcopy(self.train.records[0])
            record["valid_starts"] = record["valid_starts"][:2]
            manifest["episodes"] = [record]
            split_path = root / "split.json"
            split_path.write_text(json.dumps(manifest))
            stats = self.train.codec.norm_stats
            stats_meta = {**stats.metadata, "split_manifest_sha256": sha256_file(split_path)}
            stats_path = root / "stats.pt"
            torch.save(dict(lo=stats.lo, hi=stats.hi, metadata=stats_meta), stats_path)
            kwargs = {**self.kw, "split_manifest": split_path, "rothko_norm_stats": stats_path}
            raw = RealWorldVideoDataset(**kwargs)
            cache = root / "latents"
            cache.mkdir()
            shape = [16, 5, 28, 56]
            data = torch.empty(2, 2, *shape, dtype=torch.bfloat16)
            data[0, 0].fill_(.25); data[0, 1].fill_(.5)
            data[1, 0].fill_(.75); data[1, 1].fill_(1.)
            data.view(torch.uint16).numpy().tofile(cache / "shard.bin")
            metadata = dict(cache_version=1, complete=True, dtype="bfloat16_raw_uint16",
                num_samples=2, latent_shape=shape, modalities=["rgb", "raymap"],
                model_variant="wan2.1-t2v-1.3b", encoding_torch_dtype="torch.bfloat16",
                vae_identity={"sha256": "test-vae"},
                dataset_contract=raw.latent_cache_dataset_contract,
                shards=[dict(start=0, end=2, file="shard.bin")])
            (cache / "metadata.json").write_text(json.dumps(metadata))
            (cache / "_SUCCESS").write_text("complete\n")
            ds = RealWorldVideoDataset(**kwargs, latent_cache_dir=cache, latent_cache_only=True)
            with patch("fastwam.realworld.dataset.Image.open", side_effect=AssertionError("Cached path read a PNG")):
                for index in (0, 1):
                    sample = ds[index]
                    self.assertNotIn("video", sample)
                    torch.testing.assert_close(sample["rgb_latents"], data[index, 0])
                    torch.testing.assert_close(sample["raymap_latents"], data[index, 1])
                    start = record["valid_starts"][index]
                    expected_pose = torch.from_numpy(raw._episode(0)[0][start+1:start+17])
                    torch.testing.assert_close(sample["future_endpose"], expected_pose)
                restored = pickle.loads(pickle.dumps(ds))
                torch.testing.assert_close(restored[1]["rgb_latents"], data[1, 0])
            metadata["dataset_contract"]["camera_keys"] = ["right_camera", "wrist_camera"]
            (cache / "metadata.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                RealWorldVideoDataset(**kwargs, latent_cache_dir=cache, latent_cache_only=True)
            # Release mapped shards before TemporaryDirectory removes them on NFS.
            del ds, restored
            gc.collect()  # Dataset's bound LRU wrappers form reference cycles.

    def test_actual_data_roundtrip_future_alignment_and_absolute_ray0(self):
        probes = json.loads((ARTIFACTS / "validation_windows.json").read_text())["samples"]
        for probe in probes:
            sample = self.val[probe["val_dataset_index"]]
            self.assertEqual(sample["video"].shape, (3, 17, 224, 448))
            self.assertEqual(sample["raymap"].shape, sample["video"].shape)
            self.assertFalse(sample["action_is_pad"].any())
            poses, grip = self.val.codec.decode(sample["raymap"], sample["current_endpose"])
            torch.testing.assert_close(poses[1:, :3], sample["future_endpose"][:, :3], atol=2e-5, rtol=0)
            q1, q2 = poses[1:, 3:], sample["future_endpose"][:, 3:]
            self.assertGreater(float((q1 * q2).sum(-1).abs().min()), .99999)
            torch.testing.assert_close(grip[1:], sample["future_gripper"], atol=1e-6, rtol=0)
            # Future translations are chunk-relative, RAY0's center holds world position.
            raw = self.val.codec.denormalize(sample["raymap"].permute(1, 0, 2, 3))
            torch.testing.assert_close(raw[0, :, 112, 112], sample["current_endpose"][:3], atol=1e-6, rtol=0)
            # First target really is t+1, not t or t+2.
            r = next(r for r in self.val.records if r["path"].replace("/", "_") in probe["sample_id"])
            with np.load(ROOT / "data/realworld_data" / r["path"] / "states.npz") as state:
                target = xyzw_to_wxyz(state["robot/ee_pose"][probe["frame_index"] + 1])
            np.testing.assert_allclose(sample["future_endpose"][0].numpy(), target)

    def test_policy_uses_same_images_and_skips_current_output_frame(self):
        sample = self.val[20]
        captured = {}
        def infer(**kwargs):
            captured.update(kwargs)
            return dict(pose=torch.cat((sample["current_endpose"][None], sample["future_endpose"]))[None],
                        gripper=torch.cat((torch.ones(1, 1), sample["future_gripper"]))[None])
        model = SimpleNamespace(raymap_codec=self.val.codec, infer=infer)
        with patch("fastwam.realworld.policy.load_context", return_value=(torch.zeros(128, 4096), torch.ones(128, dtype=torch.bool))):
            policy = RealWorldPolicy(model, ARTIFACTS / "split.json", ARTIFACTS / "rothko_stats.pt", "/unused")
        images = ((sample["video"][:, 0].permute(1, 2, 0) + 1) * 127.5).round().byte().numpy()
        obs = dict(protocol_version=VERSION, request_id=7, ee_link="fr3_link8", color_order="RGB",
            prompt=self.val.records[0]["instruction"], ee_pose_wxyz=sample["current_endpose"].numpy(),
            gripper_open=1., images=dict(left_camera=images[:, :224], wrist_camera=images[:, 224:]))
        response = policy.infer(obs)
        self.assertEqual(response["request_id"], 7)
        np.testing.assert_allclose(response["target_pose_wxyz"], sample["future_endpose"].numpy(), atol=1e-7)
        torch.testing.assert_close(captured["input_image"][0], sample["video"][:, 0])
        self.assertFalse(captured["decode_future_rgb"])
        with self.assertRaises(ValueError):
            policy.infer({**obs, "color_order": "BGR"})


if __name__ == "__main__":
    unittest.main()
