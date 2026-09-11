import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import h5py


def load_builder():
    path = Path(__file__).resolve().parents[1] / "scripts/build_robotwin_c2r_actual_ee.py"
    spec = importlib.util.spec_from_file_location("build_c2r", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(tmp_path):
    q = np.arange(4*14, dtype=np.float32).reshape(4, 14) / 100
    ee = np.zeros((4, 14), dtype=np.float64)
    ee[:, [3, 10]] = 1
    ee[:, 0] = [1, 2, 3, 4]
    ee[:, 7] = [-1, -2, -3, -4]
    raw = tmp_path / "episode0.hdf5"
    with h5py.File(raw, "w") as f:
        f["joint_action/vector"] = q
        f["endpose/left_endpose"] = ee[:, :7]
        f["endpose/right_endpose"] = ee[:, 7:]
        f["endpose/left_gripper"] = q[:, 6]
        f["endpose/right_gripper"] = q[:, 13]
    table = pa.table({"observation.state": q[:-1].tolist(), "action": q[1:].tolist(),
                      "episode_index": [2200]*3, "frame_index": [0, 1, 2],
                      "index": [999, 1000, 1001], "task_index": [30, 31, 30],
                      "timestamp": [0., .02, .04]})
    source = tmp_path / "source.parquet"
    pq.write_table(table, source)
    return source, raw, ee


def check_actual_ee_alignment_and_source_preserved(tmp_path):
    b = load_builder();b.initialize({30: 0, 31: 1})
    source, raw, ee = inputs(tmp_path)
    before = source.read_bytes()
    out = tmp_path / "new/episode_000200.parquet"
    metadata, provenance = b.build_episode((source, raw, out, 2200, 200, 100, {}, "click_alarmclock"))
    result = pq.read_table(out).to_pydict()
    np.testing.assert_array_equal(result["observation.state.ee_pose_wxyz"], ee[:-1])
    np.testing.assert_array_equal(result["action.ee_pose_wxyz"], ee[1:])
    assert result["index"] == [100, 101, 102]
    assert result["episode_index"] == [200]*3
    assert result["task_index"] == [0, 1, 0]
    assert result["frame_index"] == [0, 1, 2]
    assert source.read_bytes() == before
    assert metadata["stats"]["action.ee_pose_wxyz"]["count"] == [3]
    assert provenance["source_episode_index"] == 2200


def check_reject_wrong_official_trajectory(tmp_path):
    b = load_builder();b.initialize({30: 0, 31: 1})
    source, raw, _ = inputs(tmp_path)
    with h5py.File(raw, "r+") as f:
        f["joint_action/vector"][1, 0] += .1
    out = tmp_path / "bad.parquet"
    with unittest.TestCase().assertRaisesRegex(ValueError, "alignment"):
        b.build_episode((source, raw, out, 2200, 200, 100, {}, "click_alarmclock"))
    assert not out.exists()


def check_new_config_does_not_use_original_550_episode_mapping():
    from hydra import compose, initialize_config_dir
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
        for model in ("wan22_5b", "wan21_1_3b"):
            c = compose(config_name="train", overrides=[f"task=robotwin_c2r_rothko_centerfrac05_full_{model}_1e-4"])
            assert c.data.train.robotwin_data_variant == "all"
            assert c.data.train.robotwin_task_names is None
            assert c.data.train.robotwin_ee_pose_key == "ee_pose_wxyz"
            assert c.data.train.raw_action_meta[1].key == "ee_pose_wxyz"
            assert c.data.train.raw_state_meta[1].key == "ee_pose_wxyz"
            assert c.data.val is None
            assert c.model.proprio_dim is None


class TestC2RActualEE(unittest.TestCase):
    def test_alignment_and_preservation(self):
        parent = Path(__file__).resolve().parents[1] / "runs/robotwin_c2r_actual_ee_dataset_check/unit"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            check_actual_ee_alignment_and_source_preserved(Path(directory))

    def test_wrong_trajectory_rejected(self):
        parent = Path(__file__).resolve().parents[1] / "runs/robotwin_c2r_actual_ee_dataset_check/unit"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            check_reject_wrong_official_trajectory(Path(directory))

    def test_configs(self):
        check_new_config_does_not_use_original_550_episode_mapping()


if __name__ == "__main__":
    unittest.main()
