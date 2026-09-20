import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import torch
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import prepare_all4_absolute_decoder as workflow
from fastwam.representations.rothko import quaternion_wxyz_to_matrix
from fastwam.utils.config_resolvers import register_default_resolvers

class All4DecoderPreparationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.inputs=workflow.Inputs()

    def test_fixed_split_all_starts_and_counts(self):
        x=self.inputs
        self.assertEqual((len(x.train),len(x.val),len(x.refs),len(x.val_windows)),(1632,80,264409,800))
        self.assertFalse(set(x.train)&set(x.val))
        self.assertEqual(set(Counter((e.dataset_root,e.task) for e in x.val).values()),{2})
        self.assertEqual(set(Counter((w.episode.dataset_root,w.episode.task) for w in x.val_windows).values()),{20})
        self.assertEqual(len(set(x.manifest['train_cache_indices'])),264409)
        self.assertEqual((len(x.refs)+63)//64,4132)
        self.assertEqual(int(8264*.05),413)
        expected=dict(libero_spatial_no_noops_lerobot=50824,libero_object_no_noops_lerobot=64331,
                      libero_goal_no_noops_lerobot=50445,libero_10_no_noops_lerobot=98809)
        self.assertEqual(dict(Counter(Path(r.episode.dataset_root).name for r in x.refs)),expected)

    def test_suite_offsets_match_experiment1_order(self):
        x=self.inputs;offset=0
        for root,count in zip(x.roots,[53229,67309,52895,104280]):
            self.assertEqual(x.offsets[(root,0)][0],offset)
            offset+=count
        self.assertEqual(offset,277713)
        for ep in x.train+x.val:
            start,length=x.offsets[(ep.dataset_root,ep.episode_index)]
            self.assertEqual(x.index(workflow.core.WindowRef(ep,length-1)),start+length-1)
            with self.assertRaises(AssertionError):x.index(workflow.core.WindowRef(ep,length))
        # Identical episode IDs in different suites must not collide.
        self.assertEqual(len({x.offsets[(root,0)][0] for root in x.roots}),4)

    def test_all_40_tasks_first_and_final_window_roundtrip(self):
        x=self.inputs;store=workflow.core.EpisodeStore(2)
        for root,task in sorted({(e.dataset_root,e.task) for e in x.train}):
            ep=next(e for e in x.train if (e.dataset_root,e.task)==(root,task))
            refs=[workflow.core.WindowRef(ep,0),workflow.core.WindowRef(ep,ep.length-1)]
            for w in workflow.materialize_padded_windows(store,refs,16):
                target=x.target(w);pose,g=x.codec.decode(target)
                self.assertEqual(tuple(target.shape),(3,17,224,448))
                torch.testing.assert_close(pose[:,:3],torch.from_numpy(w.pose)[:,:3],rtol=0,atol=1e-6)
                torch.testing.assert_close(quaternion_wxyz_to_matrix(pose[:,3:]),quaternion_wxyz_to_matrix(torch.from_numpy(w.pose)[:,3:]),rtol=1e-5,atol=1e-5)
                torch.testing.assert_close(g,torch.from_numpy(w.gripper),rtol=0,atol=1e-6)
                if w.start==ep.length-1:
                    self.assertEqual(workflow.valid_steps(w).sum().item(),1)
                    self.assertTrue((w.pose[1:]==w.pose[1]).all())
                    self.assertTrue((w.gripper[1:]==w.gripper[1]).all())

    def test_diT_cache_contract_is_identical(self):
        register_default_resolvers()
        with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base='1.3'):
            c=compose(config_name='train',overrides=['task=libero_all4_rothko_all_absolute_2cam224_full_wan21_1_3b_1e-4'])
        d=c.data.train
        codec=workflow.LiberoAllAbsoluteRothkoCodec(
            workflow.LiberoRothkoCodecConfig(**OmegaConf.to_container(d.rothko_config,resolve=True)),d.rothko_norm_stats)
        contract=workflow.build_dataset_contract(dataset_dirs=list(d.dataset_dirs),dataset_length=277713,num_frames=17,
            video_size=list(d.video_size),raymap_representation=d.raymap_representation,
            raymap_codec_metadata=codec.metadata(),norm_stats_sha256=codec.norm_stats.fingerprint())
        self.assertEqual(contract,self.inputs.contract)
        self.assertEqual(Path(d.latent_cache_dir).resolve(),workflow.CACHE)

    def test_default_stages_do_not_open_inputs_or_launch(self):
        for stage in ['audit','train']:
            with patch.object(sys,'argv',['workflow',stage]),patch.object(workflow,'Inputs') as inputs:
                workflow.main();inputs.assert_not_called()

    def test_config_matches_goal_except_scope_and_labels(self):
        import json
        four=dict(self.inputs.c)
        goal=json.loads((ROOT/'configs/vae/libero_goal_all_absolute_decoder_wan21_bs2_ga4_lr1e-5_ep2.json').read_text())
        for k in ['suites','output_dir','wandb_name','wandb_group','eval_windows']:
            four.pop(k);goal.pop(k)
        self.assertEqual(four,goal)
        with patch.object(sys,'argv',['core','--config',str(workflow.CONFIG)]):
            args=workflow.core.parse_args()
        self.assertEqual((args.save_every,args.eval_every,args.step_checkpoint_every,args.export_every),(400,800,800,800))
        self.assertFalse(args.auto_resume)
        self.assertTrue(args.zero_optimizer)

if __name__=='__main__':unittest.main()
