"""CPU-only checks; do not import the simulator or load a model."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    'vla_manager', Path(__file__).resolve().parents[1] / 'experiments/vlabench/run_select_book_manager.py')
manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manager)


class ShardingTests(unittest.TestCase):
    def test_exact_coverage(self):
        for workers in (1, 2, 3, 8):
            jobs = manager.partition(50, list(range(8)), workers)
            ids = [episode for _, episodes in jobs for episode in episodes]
            self.assertEqual(sorted(ids), list(range(50)))
            self.assertEqual(len(ids), len(set(ids)))
            self.assertLessEqual(max(map(lambda j: len(j[1]), jobs)) - min(map(lambda j: len(j[1]), jobs)), 1)
            for gpu in range(8):
                self.assertLessEqual(sum(g == gpu for g, _ in jobs), workers)

    def test_invalid(self):
        for args in ((50, [0, 0], 1), (50, [0], 0), (0, [0], 1)):
            with self.assertRaises(ValueError):
                manager.partition(*args)

    def test_weighted_summary_and_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = manager.partition(5, [4, 5], 1)
            for rank, (gpu, ids) in enumerate(jobs):
                folder = root / f'worker{rank}_gpu{gpu}'
                folder.mkdir()
                for episode in ids:
                    (folder / f'episode_{episode:03d}.json').write_text(json.dumps(dict(
                        episode=episode, success=episode % 2 == 0,
                        intention_score=1., progress_score=.5)))
            result = manager.summarize(root, jobs, 5)
            self.assertTrue(result['complete'])
            self.assertEqual(result['success_rate'], .6)  # Not mean of worker rates (0.5).
            (root / 'worker0_gpu4/episode_004.json').unlink()
            result = manager.summarize(root, jobs, 5)
            self.assertFalse(result['complete'])
            self.assertIsNone(result['success_rate'])
            self.assertEqual(result['missing'], [4])


if __name__ == '__main__':
    unittest.main()
