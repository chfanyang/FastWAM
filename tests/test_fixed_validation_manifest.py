import json
import tempfile
import unittest
from pathlib import Path

from fastwam.trainer import load_fixed_validation_manifest
from scripts.create_libero_plus_fixed_val_windows import (
    build_fixed_validation_manifest,
)


class FixedValidationManifestTest(unittest.TestCase):
    def _write_split(self, root: Path) -> Path:
        path = root / "split.jsonl"
        records = []
        episode_index = 0
        for suite_index, suite in enumerate(("a", "b", "c", "d")):
            for task_offset in range(10):
                task_index = suite_index * 10 + task_offset
                records.append(
                    {
                        "episode_index": episode_index,
                        "source_trajectory_id": f"source-{task_index}",
                        "split": "val",
                        "task_index": task_index,
                        "task": f"task {task_index}",
                        "suite": suite,
                        "episode_length": 32,
                        "replay_group_size": 1,
                    }
                )
                episode_index += 1
        path.write_text("".join(json.dumps(x) + "\n" for x in records))
        return path

    def test_builds_one_loss_sample_per_task_and_one_visual_per_suite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            split = self._write_split(Path(tmp))
            payload = build_fixed_validation_manifest(
                split,
                selection_seed=42,
                num_frames=17,
                visual_samples_per_suite=1,
            )
            self.assertEqual(payload["num_loss_samples"], 40)
            self.assertEqual(payload["num_visual_samples"], 4)
            self.assertEqual({x["task_index"] for x in payload["samples"]}, set(range(40)))
            self.assertEqual(sum(bool(x["run_visual"]) for x in payload["samples"]), 4)
            self.assertTrue(all(0 <= x["frame_index"] <= 15 for x in payload["samples"]))

    def test_loader_rejects_wrong_split_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split = self._write_split(root)
            payload = build_fixed_validation_manifest(
                split,
                selection_seed=42,
                num_frames=17,
                visual_samples_per_suite=1,
            )
            manifest = root / "windows.json"
            manifest.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                load_fixed_validation_manifest(
                    manifest,
                    expected_split_manifest_sha256="wrong",
                    val_dataset_length=payload["val_dataset_length"],
                )


if __name__ == "__main__":
    unittest.main()
