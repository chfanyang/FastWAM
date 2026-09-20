import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import prepare_goal_absolute_decoder as workflow
from fastwam.representations.rothko import quaternion_wxyz_to_matrix

class GoalDecoderPreparationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.inputs=workflow.Inputs()

    def test_split_matches_legacy_and_includes_all_starts(self):
        x=self.inputs
        self.assertEqual(len(x.refs),50445)
        self.assertFalse(set(x.train)&set(x.val))
        self.assertEqual(len(x.val_windows),200)
        for task in {e.task for e in x.val}:
            self.assertEqual(sum(e.task==task for e in x.val),2)
            self.assertEqual(sum(w.episode.task==task for w in x.val_windows),20)
        for r in x.refs:self.assertLess(r.start,r.episode.length)
        self.assertEqual(len(set(x.manifest['train_cache_indices'])),50445)
        self.assertEqual((len(x.refs)+63)//64,789)
        self.assertEqual(int(1578*.05),78)

    def test_cache_index_boundaries(self):
        x=self.inputs
        for e in x.train+x.val:
            lo,length=x.offsets[e.episode_index]
            self.assertEqual(x.index(workflow.core.WindowRef(e,0)),lo)
            self.assertEqual(x.index(workflow.core.WindowRef(e,length-1)),lo+length-1)
            with self.assertRaises(AssertionError):x.index(workflow.core.WindowRef(e,length))

    def test_absolute_target_roundtrip_all_tasks(self):
        x=self.inputs;store=workflow.core.EpisodeStore(2)
        for task in sorted({e.task for e in x.train}):
            ep=next(e for e in x.train if e.task==task)
            refs=[workflow.core.WindowRef(ep,0),workflow.core.WindowRef(ep,ep.length-1)]
            for w in workflow.materialize_padded_windows(store,refs,16):
                target=x.target(w);pose,g=x.codec.decode(target)
                self.assertEqual(tuple(target.shape),(3,17,224,448))
                torch.testing.assert_close(pose[:,:3],torch.from_numpy(w.pose)[:,:3],rtol=0,atol=1e-6)
                torch.testing.assert_close(quaternion_wxyz_to_matrix(pose[:,3:]),quaternion_wxyz_to_matrix(torch.from_numpy(w.pose)[:,3:]),rtol=1e-5,atol=1e-5)
                torch.testing.assert_close(g,torch.from_numpy(w.gripper),rtol=0,atol=1e-6)

    def test_tail_repeats_final_action_and_masks_only_future_padding(self):
        import numpy as np
        from types import SimpleNamespace
        ep=workflow.core.EpisodeRef(self.inputs.root,0,'synthetic',3)
        state=np.arange(21,dtype=np.float32).reshape(3,7)
        action=state+100
        grip=np.array([[0.],[.5],[1.]],dtype=np.float32)
        store=SimpleNamespace(get=lambda e:SimpleNamespace(state_pose=state,
            action_pose=action,state_gripper=grip,action=np.concatenate((state[:,:6],grip),axis=1)))
        refs=workflow.enumerate_padded_refs([ep],16)
        windows=workflow.materialize_padded_windows(store,refs,16)
        self.assertEqual(len(windows),3)
        for w in windows:
            np.testing.assert_array_equal(w.pose[0],state[w.start])
            indices=np.minimum(np.arange(w.start,w.start+16),2)
            np.testing.assert_array_equal(w.pose[1:],action[indices])
            np.testing.assert_array_equal(w.gripper[1:],grip[indices])
            self.assertEqual(workflow.valid_steps(w).sum().item(),3-w.start)
        values=torch.full((1,16),999.,dtype=torch.float64);values[0,0]=2.
        report=workflow.masked_metrics(values,workflow.valid_steps(windows[-1])[None],16)
        self.assertEqual(report,dict(n=1,mean=2.,p95=2.,max=2.))
        # The shared legacy enumerator is still unchanged outside the train entry.
        self.assertEqual(len(workflow.core.enumerate_window_refs(self.inputs.train,16)),44250)

    def test_gpu_stages_default_to_print_only(self):
        for stage in ['audit','train']:
            with patch.object(sys,'argv',['workflow',stage]),patch.object(workflow,'Inputs') as inputs:
                workflow.main();inputs.assert_not_called()

    def test_core_config_schema_and_policy(self):
        with patch.object(sys,'argv',['core','--config',str(workflow.CONFIG)]):
            args=workflow.core.parse_args()
        self.assertFalse(args.auto_resume)
        self.assertEqual((args.save_every,args.eval_every,args.step_checkpoint_every,args.export_every),(400,800,800,800))
        self.assertTrue(args.zero_optimizer)
        self.assertTrue(args.save_step_checkpoints)
        self.assertEqual(self.inputs.codec.representation,'libero_rothko_all_absolute')

if __name__=='__main__':unittest.main()
