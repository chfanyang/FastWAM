import json,unittest
from pathlib import Path
from collections import Counter
from fastwam.trainer import resolve_visual_eval_indices
from hydra import compose,initialize_config_dir
from fastwam.utils.config_resolvers import register_default_resolvers
class ValidationTest(unittest.TestCase):
    def test_legacy_and_invalid_indices(self):
        self.assertEqual(resolve_visual_eval_indices({'eval_sample_index':8,'eval_num_samples':4},10),[8,9,0,1])
        for x in [[],[0,0],[-1],[10],[True],[1.5]]:
            with self.assertRaises(ValueError):resolve_visual_eval_indices({'eval_sample_indices':x},10)
        with self.assertRaises(ValueError):resolve_visual_eval_indices({'eval_sample_indices':[0],'eval_sample_manifest':'old.json'},10)
    def test_config_coverage_and_rank_distribution(self):
        register_default_resolvers()
        for label,count in [('goal',30),('all4',120)]:
            m=json.loads(Path(f'configs/validation/libero_{label}_osc_probes.json').read_text())
            with initialize_config_dir(config_dir=str(Path('configs').resolve()),version_base='1.3'):
                c=compose(config_name='train',overrides=[f'task=libero_{label}_rothko_osc_delta_2cam224_full_wan21_1_3b_1e-4'])
            ids=resolve_visual_eval_indices(c,m['dataset_length']);self.assertEqual(ids,[r['dataset_index'] for r in m['samples']]);self.assertEqual(len(ids),count)
            counts=Counter((r['suite'],r['task_index']) for r in m['samples']);self.assertEqual(set(counts.values()),{3})
            self.assertEqual(sorted(i for rank in range(8) for i in ids[rank::8]),sorted(ids))
            for key in counts:
                rows=[r for r in m['samples'] if (r['suite'],r['task_index'])==key]
                self.assertEqual(len({r['episode_index'] for r in rows}),3)
                self.assertEqual({r['phase'] for r in rows},{'start','middle','tail'})
    def test_manifest_matches_actual_parquet(self):
        import pyarrow.parquet as pq
        for label in ['goal','all4']:
            m=json.loads(Path(f'configs/validation/libero_{label}_osc_probes.json').read_text());offsets={};base=0
            for suite in m['suite_order']:
                root=Path('data/libero_mujoco3.3.2')/f'libero_{suite}_no_noops_lerobot';off=0
                for line in (root/'meta/episodes.jsonl').read_text().splitlines():
                    e=json.loads(line);offsets[(suite,e['episode_index'])]=base+off;off+=e['length']
                base+=off
            for r in m['samples']:
                root=Path('data/libero_mujoco3.3.2')/f'libero_{r["suite"]}_no_noops_lerobot';ep=r['episode_index'];files=list(root.glob(f'data/chunk-*/episode_{ep:06d}.parquet'));self.assertEqual(len(files),1)
                t=pq.read_table(files[0],columns=['episode_index','frame_index','task_index']);f=r['frame_index']
                self.assertEqual(t['episode_index'][f].as_py(),ep);self.assertEqual(t['frame_index'][f].as_py(),f);self.assertEqual(t['task_index'][f].as_py(),r['task_index'])
                self.assertEqual(r['dataset_index'],offsets[(r['suite'],ep)]+f)
