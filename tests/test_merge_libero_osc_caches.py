import importlib.util,tempfile,unittest
from pathlib import Path
SPEC=importlib.util.spec_from_file_location('merge_osc',Path(__file__).resolve().parents[1]/'scripts/merge_libero_osc_caches.py');m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)
class MergeTest(unittest.TestCase):
    def test_reordered_sources_cross_shard_and_suite_boundaries(self):
        with tempfile.TemporaryDirectory(dir='evaluate_results/libero') as d:
            p=Path(d);a=p/'other';b=p/'goal';a.write_bytes(bytes([10,11,30,31,32,33]));b.write_bytes(bytes([20,21,22,23]))
            seg=[(0,1,a,0),(1,3,b,0),(3,5,a,2)]
            m.copy_segments(seg,0,5,2,p/'all');self.assertEqual((p/'all').read_bytes(),bytes([10,11,20,21,22,23,30,31,32,33]))
            m.copy_segments(seg,2,4,2,p/'cross');self.assertEqual((p/'cross').read_bytes(),bytes([22,23,30,31]))
            with self.assertRaises(FileExistsError):m.copy_segments(seg,0,5,2,p/'all')
            with self.assertRaises(ValueError):m.copy_segments(seg[:1]+seg[2:],0,5,2,p/'gap')
            with self.assertRaises(EOFError):m.copy_segments([(0,20,a,0)],0,20,2,p/'short')
