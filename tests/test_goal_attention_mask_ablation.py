import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from fastwam.utils.config_resolvers import register_default_resolvers

ROOT=Path(__file__).resolve().parents[1]
BASE='libero_goal_rothko_all_absolute_2cam224_full_wan21_1_3b_1e-4'
DECOUPLED='rgb_raymap_decoupled'
BIDIRECTIONAL='rgb_raymap_future_bidirectional'
BASEMODE='rgb_then_raymap_block_causal'


def mask(mode,tokens=1,conditions=(0,5),length=10):
    return WanVideoDiT.build_video_to_video_mask(SimpleNamespace(video_attention_mask_mode=mode),
        length*tokens,tokens,torch.device('cpu'),condition_frame_indices=conditions)


def tiny(mode):
    return WanVideoDiT(hidden_dim=36,in_dim=16,ffn_dim=72,out_dim=16,text_dim=36,
        freq_dim=18,eps=1e-6,patch_size=(1,1,1),num_heads=2,attn_head_dim=18,
        num_layers=3,has_image_input=False,seperated_timestep=True,
        require_vae_embedding=False,require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,video_attention_mask_mode=mode,
        use_gradient_checkpointing=False).eval()


class GoalMaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_documented_matrices_and_spatial_expansion(self):
        c=[1,0,0,0,0,1,0,0,0,0]
        v=[1,1,1,1,1,1,0,0,0,0]
        a=[1,0,0,0,0,1,1,1,1,1]
        expected={DECOUPLED:[c]+[v]*4+[c]+[a]*4,
                  BIDIRECTIONAL:[c]+[[1]*10]*4+[c]+[[1]*10]*4,
                  BASEMODE:[c]+[v]*4+[c]+[[1]*10]*4}
        for mode,rows in expected.items():
            want=torch.tensor(rows,dtype=torch.bool)
            self.assertTrue(torch.equal(mask(mode),want))
            self.assertTrue(torch.equal(mask(mode,3),want.repeat_interleave(3,0).repeat_interleave(3,1)))

    def test_legacy_bidirectional_is_unchanged_and_distinct(self):
        self.assertTrue(mask('bidirectional').all())
        self.assertFalse(mask(BIDIRECTIONAL)[0,1])
        self.assertTrue(torch.equal(mask('condition_frames_causal'),mask(BIDIRECTIONAL)))

    def test_invalid_layouts_fail(self):
        for mode in [DECOUPLED,BIDIRECTIONAL]:
            for cond,length in [(None,10),((0,4),10),((0,5),9),((0,1),2),((0,5,6),10)]:
                with self.assertRaises(ValueError):mask(mode,conditions=cond,length=length)
            with self.assertRaises(ValueError):
                WanVideoDiT.build_video_to_video_mask(SimpleNamespace(video_attention_mask_mode=mode),
                    21,2,torch.device('cpu'),condition_frame_indices=(0,5))

    @torch.no_grad()
    def test_three_layer_perturbations_obey_information_flow(self):
        # Exercise actual forward through every block, not only a mask helper.
        for mode in [DECOUPLED,BIDIRECTIONAL,BASEMODE]:
            torch.manual_seed(71);model=tiny(mode)
            x=torch.randn(1,16,10,2,2);context=torch.randn(1,3,36)
            def forward(inp):
                return model(x=inp,timestep=torch.tensor([.6]),context=context,
                    context_mask=torch.ones(1,3,dtype=torch.bool),
                    fuse_vae_embedding_in_latents=True,condition_latent_indices=(0,5))
            y=forward(x)
            for changed,other in [(slice(1,5),slice(6,10)),(slice(6,10),slice(1,5))]:
                z=x.clone();z[:,:,changed]+=torch.randn_like(z[:,:,changed])*2
                altered=forward(z)
                torch.testing.assert_close(y[:,:,[0,5]],altered[:,:,[0,5]],rtol=0,atol=1e-6)
                delta=(y[:,:,other]-altered[:,:,other]).abs().max().item()
                allowed=mode==BIDIRECTIONAL or (mode==BASEMODE and changed.start==1)
                if allowed:self.assertGreater(delta,1e-5)
                else:self.assertLess(delta,1e-6)
                self.assertGreater((y[:,:,changed]-altered[:,:,changed]).abs().max().item(),1e-5)

    def test_cpu_backward_is_finite(self):
        for mode in [DECOUPLED,BIDIRECTIONAL]:
            model=tiny(mode).train();x=torch.randn(1,16,10,2,2,requires_grad=True)
            y=model(x=x,timestep=torch.tensor([.4]),context=torch.randn(1,3,36),
                fuse_vae_embedding_in_latents=True,condition_latent_indices=(0,5))
            y.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(x.grad.abs().sum().item(),0)

    def test_configs_only_change_mask_and_run_labels(self):
        register_default_resolvers()
        configs=[]
        with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base='1.3'):
            for task in [BASE,'libero_goal_all_absolute_mask_decoupled_wan21',
                         'libero_goal_all_absolute_mask_future_bidirectional_wan21']:
                cfg=compose(config_name='train',overrides=['task='+task])
                configs.append(OmegaConf.to_container(cfg,resolve=True))
        for c,mode in zip(configs,[BASEMODE,DECOUPLED,BIDIRECTIONAL]):
            self.assertEqual(c['model']['video_dit_config']['video_attention_mask_mode'],mode)
            self.assertEqual(len(c['data']['train']['dataset_dirs']),1)
            self.assertIn('libero_goal_no_noops',c['data']['train']['dataset_dirs'][0])
            self.assertEqual(c['save_every'],2000);self.assertEqual(c['state_save_every'],2000)
            self.assertEqual(c['batch_size']*c['gradient_accumulation_steps']*8,128)
            self.assertEqual(c['num_epochs'],10)
            self.assertIsNone(c['model']['vae_safetensors_path'])
            c.pop('output_dir');c['wandb'].pop('name')
            c['model']['video_dit_config'].pop('video_attention_mask_mode')
        self.assertEqual(configs[0],configs[1]);self.assertEqual(configs[0],configs[2])

if __name__=='__main__':unittest.main()
