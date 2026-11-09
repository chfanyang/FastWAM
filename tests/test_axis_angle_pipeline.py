"""Isolation and actual DiT/backward checks for the opt-in axis-angle experiment."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from fastwam.models.wan22.fastwam_axis_angle import FastWAMAxisAngle
from fastwam.models.wan22.fastwam_visual_action import FastWAMVideoOnlyRaymap
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from fastwam.utils.config_resolvers import register_default_resolvers

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'evaluate_results/libero/preparation/axis_angle_pipeline_20261108'


def composed(name='train', task='libero_goal_axis_angle_constant_wan21'):
    register_default_resolvers()
    with initialize_config_dir(config_dir=str(ROOT/'configs'), version_base='1.3'):
        return compose(config_name=name, overrides=['task='+task] if name=='train' else [])


class CachedOnlyVAE(torch.nn.Module):
    temporal_downsample_factor = 4
    upsampling_factor = 8

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.z_dim = 16

    def encode(self, *args, **kwargs):
        raise AssertionError('Cache-only forward must not run the VAE')


def tiny(cls=FastWAMAxisAngle):
    cfg = composed()
    dit = WanVideoDiT(hidden_dim=36, in_dim=16, ffn_dim=72, out_dim=16,
        text_dim=36, freq_dim=18, eps=1e-6, patch_size=(1,4,4), num_heads=2,
        attn_head_dim=18, num_layers=1, has_image_input=False,
        seperated_timestep=True, require_vae_embedding=False,
        require_clip_embedding=False, fuse_vae_embedding_in_latents=True,
        video_attention_mask_mode='rgb_then_raymap_block_causal',
        use_gradient_checkpointing=False)
    m = cls(video_expert=dit, vae=CachedOnlyVAE(), text_dim=36, device='cpu',
        raymap_representation='libero_rothko',
        rothko_norm_stats=str(ROOT/cfg.data.train.rothko_norm_stats),
        rothko_config=OmegaConf.to_container(cfg.data.train.rothko_config),
        loss_lambda_rgb=1., loss_lambda_raymap=5.)
    return m


class AxisAnglePipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        ARTIFACTS.mkdir(parents=True, exist_ok=True)

    def test_scientific_configuration_matches_reference(self):
        new = composed()
        old = OmegaConf.load(ROOT/'configs/libero_goal_rothko_absolute_ray0_wan21_epoch_weights_8gpu.yaml')
        for key in ('batch_size','gradient_accumulation_steps','num_epochs','learning_rate',
                'warmup_ratio','lr_scheduler_type','weight_decay','max_grad_norm',
                'mixed_precision','seed','eval_every','eval_num_samples','state_save_every'):
            self.assertEqual(new[key], old[key], key)
        self.assertEqual(new.save_every_epochs, 1)
        a = OmegaConf.to_container(new.model, resolve=True)
        b = OmegaConf.to_container(old.model, resolve=True)
        a.pop('_target_'); b.pop('_target_')
        self.assertEqual(a,b)
        a = OmegaConf.to_container(new.data.train, resolve=True)
        b = OmegaConf.to_container(old.data.train, resolve=True)
        a.pop('_target_'); b.pop('_target_')
        self.assertNotEqual(a.pop('latent_cache_dir'), b.pop('latent_cache_dir'))
        self.assertEqual(a.pop('source_latent_cache_dir'), old.data.train.latent_cache_dir)
        self.assertEqual(a,b)
        e = composed('sim_libero_goal_axis_angle')
        self.assertEqual(e.EVALUATION.replan_steps,8)
        self.assertEqual(e.EVALUATION.num_inference_steps,20)
        self.assertEqual(e.EVALUATION.num_trials,50)
        self.assertTrue(e.model.load_text_encoder)
        self.assertIs(FastWAMAxisAngle.training_loss, FastWAMVideoOnlyRaymap.training_loss)
        self.assertIs(FastWAMAxisAngle.infer, FastWAMVideoOnlyRaymap.infer)

    def test_cache_and_checkpoint_reject_other_representation(self):
        model = tiny(); old = tiny(FastWAMVideoOnlyRaymap)
        new_meta = model._visual_action_checkpoint_config()
        old_meta = old._visual_action_checkpoint_config()
        model._validate_visual_action_checkpoint_config(new_meta,checkpoint_path='new')
        for m,meta in ((model,old_meta),(old,new_meta)):
            with self.assertRaises(ValueError):
                m._validate_visual_action_checkpoint_config(meta,checkpoint_path='wrong')
        missing = dict(new_meta); missing.pop('rotation_conversion')
        with self.assertRaises(ValueError):
            model._validate_visual_action_checkpoint_config(missing,checkpoint_path='missing')
        meta = json.loads((ROOT/'data/libero_goal_rothko_absolute_ray0_wan21_bf16_h16_latents_20260921/metadata.json').read_text())
        model.model_variant = meta['model_variant']
        model.torch_dtype = torch.bfloat16
        model._current_vae_identity = lambda: meta['vae_identity']
        with self.assertRaises(ValueError): model.validate_latent_cache_metadata(meta)
        correct = copy.deepcopy(meta)
        correct['dataset_contract'].update(raymap_representation=model.raymap_representation,
            raymap_codec_metadata=model.raymap_codec.metadata())
        model.validate_latent_cache_metadata(correct)
        correct['benchmark_only'] = True
        with self.assertRaises(ValueError): model.validate_latent_cache_metadata(correct)

    def test_actual_dit_cached_backward_save_and_restore(self):
        torch.manual_seed(41)
        model = tiny(); optimizer = torch.optim.AdamW(model.dit.parameters(), lr=1e-4)
        sample = dict(rgb_latents=torch.randn(1,16,5,28,56),
            raymap_latents=torch.randn(1,16,5,28,56),context=torch.randn(1,3,36),
            context_mask=torch.ones(1,3,dtype=torch.bool),
            image_is_pad=torch.zeros(1,17,dtype=torch.bool),
            raymap_is_pad=torch.zeros(1,17,dtype=torch.bool))
        inputs = model.build_inputs(sample)
        self.assertEqual(inputs['joint_latents'].shape,(1,16,10,28,56))
        sample['raymap_is_pad'][0,13:] = True
        loss,metrics = model.training_loss(sample)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(metrics['loss_total'], metrics['loss_rgb_raw']+5*metrics['loss_raymap_raw'], places=5)
        before = {k:v.clone() for k,v in model.dit.state_dict().items()}
        loss.backward()
        grads = [p.grad for p in model.dit.parameters() if p.grad is not None]
        self.assertTrue(grads); self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(g.abs().sum().item() for g in grads),0.)
        optimizer.step()
        self.assertTrue(any(not torch.equal(before[k],v) for k,v in model.dit.state_dict().items()))
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as tmp:
            path = Path(tmp)/'step1.pt'
            model.save_checkpoint(path,optimizer=optimizer,step=1)
            restored = tiny(); restored_opt = torch.optim.AdamW(restored.dit.parameters(),lr=1e-4)
            restored.load_checkpoint(path,optimizer=restored_opt)
            for k,v in model.dit.state_dict().items():
                torch.testing.assert_close(restored.dit.state_dict()[k],v,rtol=0,atol=0)
            self.assertTrue(restored_opt.state_dict()['state'])
            torch.manual_seed(97); a,_ = model.training_loss(sample)
            torch.manual_seed(97); b,_ = restored.training_loss(sample)
            torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__=='__main__': unittest.main()
