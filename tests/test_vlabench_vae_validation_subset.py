import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('subset',Path(__file__).resolve().parents[1]/'scripts/vlabench_vae_validation_subset.py')
subset=importlib.util.module_from_spec(spec);spec.loader.exec_module(subset)

class ValidationSubsetTests(unittest.TestCase):
    def test_fixed_sample_mapping_and_rank_partition(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);split=p/'split.json';meta=p/'episodes.jsonl'
            split.write_text(json.dumps(dict(val_episode_indices=[3,1],records=[dict(episode_index=i,base_task=f't{i}') for i in [1,2,3]])))
            meta.write_text('\n'.join(json.dumps(dict(episode_index=i,length=n)) for i,n in [(1,7),(2,100),(3,12)]))
            m=subset.build_manifest(split,meta,42)
            self.assertEqual(m,subset.build_manifest(split,meta,42))
            self.assertEqual((m['total_windows'],m['selected_windows']),(19,9))
            lookup=[(3,i) for i in range(12)]+[(1,i) for i in range(7)]
            for row in m['windows']:
                self.assertEqual((row['episode_index'],row['frame_start']),lookup[row['val_index']])
            path=p/'subset.json';path.write_text(json.dumps(m))
            indices=subset.load_indices(path,total_windows=19,episode_layout=[(3,12),(1,7)],split_sha256=m['split_manifest_sha256'])
            ranks=[indices[r::8] for r in range(8)]
            self.assertEqual(sorted(i for rank in ranks for i in rank),indices)
            self.assertEqual(len(set(indices)),9)
            with self.assertRaises(ValueError):subset.load_indices(path,total_windows=19,episode_layout=[(1,7),(3,12)],split_sha256=m['split_manifest_sha256'])
            with self.assertRaises(ValueError):subset.load_indices(path,total_windows=19,episode_layout=[(3,12),(1,7)],split_sha256='wrong')
            m['indices'][1]=m['indices'][0];path.write_text(json.dumps(m))
            with self.assertRaises(ValueError):subset.load_indices(path,total_windows=19,episode_layout=[(3,12),(1,7)],split_sha256=m['split_manifest_sha256'])

if __name__=='__main__':unittest.main()
