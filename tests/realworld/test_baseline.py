"""CPU tests for the original FastWAM adapter; never load pretrained GPU weights."""
import copy
import json
import pickle
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fastwam.realworld.baseline_data import ActionCodec, quaternion_multiply
from fastwam.realworld.baseline_model import OriginalRealWorldFastWAM
from fastwam.realworld.baseline_training import OriginalRealWorldTrainer
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.mot import MoT
from fastwam.realworld.dataset import RealWorldVideoDataset
from fastwam.utils.config_resolvers import register_default_resolvers

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/realworld_artifacts/ray0_absolute_left_wrist_val2_g08_q9995_v1"
STATS = ROOT / "data/realworld_artifacts/original_fastwam_wan22_5b_left_wrist_val2_g08_q9995_v1/action_stats.json"


def config():
    register_default_resolvers()
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=["task=realworld_all4_original_fastwam_wan22_5b_ep8",
            "data.train.include_text_context=false"])
    OmegaConf.resolve(cfg)
    return cfg


class TinyVAE(torch.nn.Module):
    temporal_downsample_factor = 4

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def encode(self, video, **kwargs):
        if isinstance(video, list):
            return [v[:2, ::4, ::16, ::16] * self.scale for v in video]
        return video[:, :2, ::4, ::16, ::16] * self.scale


def tiny_model(codec, checkpointing=True):
    video = WanVideoDiT(hidden_dim=32, in_dim=2, ffn_dim=64, out_dim=2,
        text_dim=16, freq_dim=16, eps=1e-6, patch_size=(1, 1, 1), num_heads=2,
        attn_head_dim=8, num_layers=2, has_image_input=False, seperated_timestep=True,
        action_dim=7, video_attention_mask_mode="first_frame_causal",
        use_gradient_checkpointing=checkpointing)
    action = ActionDiT(hidden_dim=16, action_dim=7, ffn_dim=32, text_dim=16, freq_dim=16,
        eps=1e-6, num_heads=2, attn_head_dim=8, num_layers=2,
        use_gradient_checkpointing=checkpointing)
    mot = MoT({"video": video, "action": action}, mot_checkpoint_mixed_attn=checkpointing)
    model = OriginalRealWorldFastWAM(video, action, mot, TinyVAE(), text_dim=16,
        proprio_dim=8, device="cpu", torch_dtype=torch.float32)
    model.action_codec = codec
    return model


@unittest.skipUnless(STATS.exists(), "Run workflow_baseline.sh prepare first")
class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cfg = config()
        cls.train = instantiate(cls.cfg.data.train)
        cls.val = instantiate(cls.cfg.data.val)
        cls.codec = cls.train.codec

    def test_same_split_rgb_and_current_frame_relative_targets(self):
        reference = RealWorldVideoDataset(dataset_dir=ROOT / "data/realworld_data",
            split_manifest=SOURCE / "split.json", rothko_norm_stats=SOURCE / "rothko_stats.pt",
            include_text_context=False)
        self.assertEqual(self.train.records, reference.records)
        self.assertEqual((len(self.train), len(self.val)), (96602, 2410))
        self.assertEqual((len(self.train.records), len(self.val.records)), (371, 8))
        # Check a window from each task, including actual non-identity orientations.
        seen = set()
        for ep, record in enumerate(self.train.records):
            if record["task"] in seen:
                continue
            seen.add(record["task"])
            index = self.train.offsets[ep] + len(record["valid_starts"]) // 2
            sample, old = self.train[index], reference[index]
            self.assertNotIn("raymap", sample)
            torch.testing.assert_close(sample["video"], old["video"], rtol=0, atol=0)
            self.assertEqual(sample["action"].shape, (16, 7))
            self.assertEqual(sample["proprio"].shape, (17, 8))
            pose, grip = self.codec.decode(sample["current_endpose"].numpy(), sample["action"].numpy())
            target = sample["future_endpose"].numpy()
            np.testing.assert_allclose(pose[:, :3], target[:, :3], atol=1e-6)
            np.testing.assert_allclose(np.abs(np.sum(pose[:, 3:] * target[:, 3:], axis=-1)), 1., atol=1e-6)
            np.testing.assert_array_equal(grip, sample["future_gripper"].numpy()[:, 0])
        restored = pickle.loads(pickle.dumps(self.train))
        torch.testing.assert_close(restored[0]["action"], self.train[0]["action"])

    def test_rotation_sign_pi_and_local_translation(self):
        q0 = np.array([np.sqrt(.5), 0, 0, np.sqrt(.5)])
        current = np.r_[.5, 0, .25, q0]
        relative = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [-1, 0, 0, 0]])
        future = np.tile(current, (3, 1))
        future[:, 3:] = quaternion_multiply(q0, relative)
        future[:, 1] += .01
        actions = self.codec.encode(current, future, np.array([0, 1, 0]))
        np.testing.assert_allclose(actions[:, 0] * self.codec.scale[0], .01, atol=1e-7)
        np.testing.assert_allclose(actions[:, 1:3], 0, atol=1e-7)
        opposite = future.copy()
        opposite[:, 3:] *= -1
        np.testing.assert_allclose(actions, self.codec.encode(current, opposite, np.array([0, 1, 0])), atol=1e-7)
        poses, _ = self.codec.decode(current, actions)
        np.testing.assert_allclose(np.abs(np.sum(poses[:, 3:] * future[:, 3:], axis=-1)), 1, atol=1e-6)
        invalid = actions.copy()
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.codec.decode(current, invalid)

    def test_5b_config_factory_and_isolation(self):
        cfg = self.cfg
        self.assertEqual(cfg.batch_size * cfg.gradient_accumulation_steps * 8, 128)
        self.assertEqual(cfg.num_epochs, 8)
        self.assertEqual(cfg.model.model_id, "Wan-AI/Wan2.2-TI2V-5B")
        self.assertTrue(cfg.finetune.train_proprio_encoder)
        self.assertTrue(cfg.model.mot_checkpoint_mixed_attn)
        self.assertTrue(cfg.model.video_dit_config.use_gradient_checkpointing)
        self.assertTrue(cfg.model.action_dit_config.use_gradient_checkpointing)
        self.assertIs(OriginalRealWorldFastWAM.training_loss, FastWAM.training_loss)
        with patch.object(OriginalRealWorldFastWAM, "from_wan22_pretrained", return_value=SimpleNamespace()) as factory:
            model = instantiate(cfg.model, device="cpu", model_dtype=torch.float32)
        kwargs = factory.call_args.kwargs
        self.assertEqual(kwargs["video_dit_config"]["hidden_dim"], 3072)
        self.assertEqual(kwargs["action_dit_config"]["hidden_dim"], 1024)
        self.assertEqual(kwargs["loss_lambda_video"], 1.)
        self.assertEqual(kwargs["loss_lambda_action"], 1.)
        self.assertEqual(model.action_codec.fingerprint, self.codec.fingerprint)
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
            modified = compose(config_name="train", overrides=["task=realworld_all4_ray0_absolute_wan21_ep8"])
        self.assertEqual(modified.model.model_id, "Wan-AI/Wan2.1-T2V-1.3B")
        self.assertNotEqual(cfg.output_dir, modified.output_dir)

    def test_per_task_configs_epochs_counts_and_fixed_validation(self):
        expected = {
            "realworld_pick_place_original_fastwam_wan22_5b_ep8": ("pick_place", 19042, 494, 8),
            "realworld_insert_pen_original_fastwam_wan22_5b_ep8": ("insert_pen", 30266, 879, 8),
            "realworld_stack_bowls_three_original_fastwam_wan22_5b_ep8": (
                "stack_bowls_three", 26940, 501, 8),
            "realworld_white_board_original_fastwam_wan22_5b_ep5": ("white_board", 20354, 536, 5),
        }
        for config_name, (task_name, train_count, val_count, epochs) in expected.items():
            with self.subTest(task=task_name):
                with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
                    cfg = compose(config_name="train", overrides=[f"task={config_name}",
                        "data.train.include_text_context=false"])
                train, val = instantiate(cfg.data.train), instantiate(cfg.data.val)
                self.assertEqual((train.task_name, val.task_name), (task_name, task_name))
                self.assertEqual({record["task"] for record in train.records}, {task_name})
                self.assertEqual({record["task"] for record in val.records}, {task_name})
                self.assertEqual((len(train), len(val), cfg.num_epochs),
                    (train_count, val_count, epochs))
                manifest = json.loads(Path(cfg.eval_sample_manifest).read_text())
                self.assertEqual(len(manifest["samples"]), 2)
                for sample in manifest["samples"]:
                    value = val[sample["val_dataset_index"]]
                    self.assertEqual(value["action"].shape, (16, 7))

    def test_original_loss_backward_checkpointing_and_checkpoint_identity(self):
        model = tiny_model(self.codec)
        sample = dict(video=torch.randn(1, 3, 17, 16, 16), action=torch.randn(1, 16, 7),
            proprio=torch.randn(1, 17, 8), context=torch.randn(1, 4, 16),
            context_mask=torch.ones(1, 4, dtype=torch.bool),
            image_is_pad=torch.zeros(1, 17, dtype=torch.bool), action_is_pad=torch.zeros(1, 16, dtype=torch.bool))
        model.train()
        state = torch.get_rng_state()
        loss, metrics = model.training_loss(sample)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(metrics), {"loss_video", "loss_action"})
        loss.backward()
        for module in (model.video_expert, model.action_expert, model.proprio_encoder):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)
        self.assertIsNone(model.vae.scale.grad)
        grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        model.video_expert.use_gradient_checkpointing = False
        model.action_expert.use_gradient_checkpointing = False
        model.mot.mot_checkpoint_mixed_attn = False
        torch.set_rng_state(state)
        other, _ = model.training_loss(sample)
        other.backward()
        torch.testing.assert_close(loss, other)
        for name, p in model.named_parameters():
            if name in grads:
                torch.testing.assert_close(p.grad, grads[name], rtol=1e-4, atol=1e-6)
        model.eval()
        pred = model.infer_action(prompt=None, input_image=sample["video"][:, :, 0], action_horizon=16,
            proprio=sample["proprio"][0, 0], context=sample["context"][0],
            context_mask=sample["context_mask"][0], num_inference_steps=2, seed=42)
        self.assertEqual(pred["action"].shape, (16, 7))
        self.assertTrue(torch.isfinite(pred["action"]).all())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            model.save_checkpoint(path, step=7)
            self.assertEqual(model.load_checkpoint(path)["step"], 7)
            original = model.action_codec
            model.action_codec = copy.copy(original)
            model.action_codec.fingerprint = "wrong-split"
            with self.assertRaisesRegex(ValueError, "statistics / split mismatch"):
                model.load_checkpoint(path)
            model.action_codec = original

    def test_fixed_four_task_evaluation(self):
        model = tiny_model(self.codec)
        records = json.loads((SOURCE / "validation_windows.json").read_text())["samples"]
        samples = {}
        for record in records:
            value = self.val[record["val_dataset_index"]]
            value["video"] = value["video"][:, :, :16, :16]
            value["context"] = torch.zeros(4, 16)
            value["context_mask"] = torch.ones(4, dtype=torch.bool)
            samples[record["val_dataset_index"]] = value
        class Data:
            codec = self.codec
            def __getitem__(self, index):
                return samples[index]
        trainer = object.__new__(OriginalRealWorldTrainer)
        trainer.val_dataset = Data()
        trainer.fixed_validation_manifest = dict(samples=records)
        trainer.accelerator = SimpleNamespace(unwrap_model=lambda m: m, device=torch.device("cpu"),
            process_index=0, num_processes=1, is_main_process=True,
            autocast=nullcontext, gather=lambda t: t)
        trainer.model = model.eval()
        trainer.eval_num_inference_steps, trainer.global_step = 2, 0
        with tempfile.TemporaryDirectory() as tmp:
            trainer.eval_dir = tmp
            result = trainer.evaluate()
            self.assertTrue(all(np.isfinite(v) for v in result.values()))
            self.assertEqual(len(list(Path(tmp).glob('*.npz'))), 4)
            self.assertTrue((Path(tmp) / "step_000000_metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
