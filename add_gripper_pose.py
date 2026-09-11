#!/usr/bin/env python
"""Recompute dual-arm gripper (end-effector) pose from joint positions and add
it to a RoboTwin LeRobot dataset in-place.

For each frame the script computes, via SAPIEN forward kinematics on the
aloha-agilex URDF (the exact same kinematics RoboTwin uses), the world-frame
pose of the left/right end-effector, faithfully reproducing RoboTwin's native
`endpose` convention (`Robot.get_left_ee_pose` -> `_trans_endpose(is_endpose=False)`
in third_party/RoboTwin/envs/robot/robot.py):

    ee_pose          = sapien global pose of joint fl_joint6 / fr_joint6
    R                = quat2mat(ee_pose.q) @ global_trans_matrix @ delta_matrix
    dis              = gripper_bias - 0.12            # 0 for is_endpose=False
    pose_position    = ee_pose.p + R @ [dis, 0, 0]
    pose_orientation = mat2quat(R)                    # wxyz (transforms3d)

The result is a 14-dim vector per frame: [left (pos3 + quat_wxyz4), right (...)].
It is written to two new columns in every episode parquet:
    - `action.endpose`            (from the `action` joint targets)
    - `observation.state.endpose` (from the `observation.state` joint values)
and registered in meta/info.json and meta/episodes_stats.jsonl.

For the released FastWAM RoboTwin data, `observation.state` matches the
previous recorded joint drive targets, not measured joint positions. Its FK
therefore need not equal the native, physically observed RoboTwin endpose.
This script preserves that source-column meaning; it does not reconstruct
measured end-effector poses from images or rename targets as measurements.

Run with the RoboTwin environment (and ensure that it has a parquet engine such
as pyarrow), e.g.:
    conda run -n RoboTwin python add_gripper_pose.py \
        --limit 5 --num-workers 1 --no-meta-update       # small validation run
    conda run -n RoboTwin python add_gripper_pose.py \
        --num-workers 8 --skip-existing                  # resumable full run
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_ROOT = os.path.join(
    _REPO_ROOT, "data/robotwin2.0/robotwin2.0"
)
DEFAULT_EMBODIMENT_DIR = os.path.join(
    _REPO_ROOT, "third_party/RoboTwin/assets/embodiments/aloha-agilex"
)
DEFAULT_URDF = os.path.join(DEFAULT_EMBODIMENT_DIR, "urdf/arx5_description_isaac.urdf")
DEFAULT_CONFIG = os.path.join(DEFAULT_EMBODIMENT_DIR, "config.yml")

# Source joint columns -> arm joint slices (gripper entries 6 and 13 ignored).
LEFT_JOINT_SLICE = slice(0, 6)
RIGHT_JOINT_SLICE = slice(7, 13)

ENDPOSE_NAMES = [
    "left_x", "left_y", "left_z", "left_qw", "left_qx", "left_qy", "left_qz",
    "right_x", "right_y", "right_z", "right_qw", "right_qx", "right_qy", "right_qz",
]
ENDPOSE_DIM = 14


# ----------------------------------------------------------------------------
# SAPIEN forward kinematics (mirrors RoboTwin's Robot class)
# ----------------------------------------------------------------------------
class SapienEndposeFK:
    """Loads the aloha-agilex URDF in SAPIEN and reproduces RoboTwin's endpose.

    SAPIEN objects are not picklable, so instances are created inside each
    worker process (see `_worker_init`).
    """

    def __init__(self, urdf_path: str, config_path: str):
        import sapien
        import transforms3d as t3d
        import yaml

        self._t3d = t3d

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.left_ee_name = cfg["ee_joints"][0]
        self.right_ee_name = cfg["ee_joints"][1]
        self.left_arm_joints = cfg["arm_joints_name"][0]
        self.right_arm_joints = cfg["arm_joints_name"][1]
        self.gripper_bias = float(cfg["gripper_bias"])
        self.global_trans = np.array(
            cfg.get("global_trans_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            dtype=np.float64,
        )
        self.delta = np.array(
            cfg.get("delta_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            dtype=np.float64,
        )
        robot_pose = cfg.get("robot_pose", [[0, -0.65, 0, 1, 0, 0, 1]])[0]
        self._root_pose = sapien.Pose(robot_pose[:3], robot_pose[-4:])

        # Headless PhysX-only scene.  ``sapien.Scene()`` also constructs a
        # RenderSystem by default in SAPIEN 3, which requires a rendering
        # device and can abort the whole worker on a headless node.  Forward
        # kinematics only needs the CPU PhysX system.
        self._scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True

        # SAPIEN's URDF loader constructs RenderMaterial objects while parsing
        # <visual> elements even when the scene itself has no RenderSystem.
        # Geometry is irrelevant to FK, so load a temporary kinematics-only
        # copy of the URDF with both visual and collision elements removed.
        # With those elements gone, the copy contains no relative asset paths
        # and can safely live in the system temporary directory.
        tree = ET.parse(urdf_path)
        for link in tree.getroot().iter("link"):
            for tag in ("visual", "collision"):
                for element in list(link.findall(tag)):
                    link.remove(element)
        fd, kinematic_urdf_path = tempfile.mkstemp(
            prefix=".tmp_fk_", suffix=".urdf"
        )
        os.close(fd)
        try:
            tree.write(kinematic_urdf_path, encoding="utf-8", xml_declaration=True)
            self._robot = loader.load(kinematic_urdf_path)
        finally:
            if os.path.exists(kinematic_urdf_path):
                os.remove(kinematic_urdf_path)
        self._robot.set_root_pose(self._root_pose)

        self._dof = self._robot.dof
        active_names = [j.get_name() for j in self._robot.get_active_joints()]
        name_to_idx = {n: i for i, n in enumerate(active_names)}
        self._left_idx = np.array([name_to_idx[n] for n in self.left_arm_joints])
        self._right_idx = np.array([name_to_idx[n] for n in self.right_arm_joints])

        self._left_ee = self._robot.find_joint_by_name(self.left_ee_name)
        self._right_ee = self._robot.find_joint_by_name(self.right_ee_name)

        self._qpos = np.zeros(self._dof, dtype=np.float64)

    def _trans_endpose(self, ee_joint) -> np.ndarray:
        """RoboTwin `_trans_endpose(is_endpose=False)` for one arm -> (7,)."""
        t3d = self._t3d
        ee_pose = ee_joint.global_pose
        R = t3d.quaternions.quat2mat(ee_pose.q) @ self.global_trans @ self.delta
        dis = self.gripper_bias - 0.12  # is_endpose=False
        pos = ee_pose.p + R @ np.array([dis, 0.0, 0.0])
        quat = t3d.quaternions.mat2quat(R)  # wxyz
        return np.concatenate([pos, quat])

    def compute_endpose(self, joints: np.ndarray) -> np.ndarray:
        """joints: (N, 14) -> (N, 14) endpose (left 7 + right 7) float32."""
        n = joints.shape[0]
        out = np.zeros((n, ENDPOSE_DIM), dtype=np.float64)
        left_theta = np.asarray(joints[:, LEFT_JOINT_SLICE], dtype=np.float64)
        right_theta = np.asarray(joints[:, RIGHT_JOINT_SLICE], dtype=np.float64)
        qpos = self._qpos
        for k in range(n):
            qpos[self._left_idx] = left_theta[k]
            qpos[self._right_idx] = right_theta[k]
            self._robot.set_qpos(qpos)
            out[k, 0:7] = self._trans_endpose(self._left_ee)
            out[k, 7:14] = self._trans_endpose(self._right_ee)
        return out.astype(np.float32)


# ----------------------------------------------------------------------------
# Stats
# ----------------------------------------------------------------------------
def feature_stats(arr: np.ndarray) -> Dict[str, Any]:
    """arr: (N, D) -> lerobot per-dim stats dict (lists), count=[N]."""
    a = arr.astype(np.float64)
    return {
        "min": a.min(axis=0).tolist(),
        "max": a.max(axis=0).tolist(),
        "mean": a.mean(axis=0).tolist(),
        "std": a.std(axis=0, ddof=0).tolist(),
        "count": [int(a.shape[0])],
    }


# ----------------------------------------------------------------------------
# Per-episode worker
# ----------------------------------------------------------------------------
_FK: SapienEndposeFK = None  # type: ignore[assignment]


def _worker_init(urdf_path: str, config_path: str) -> None:
    global _FK
    _FK = SapienEndposeFK(urdf_path, config_path)


def _stack_column(series: pd.Series) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float64) for v in series.values])


def process_episode(parquet_path: str, skip_existing: bool) -> Tuple[int, Dict[str, Any]]:
    """Add endpose columns to one parquet (atomic write). Returns (ep_idx, stats)."""
    fk = _FK
    df = pd.read_parquet(parquet_path)

    ep_idx = int(df["episode_index"].iloc[0]) if "episode_index" in df.columns else -1

    if skip_existing and "action.endpose" in df.columns and "observation.state.endpose" in df.columns:
        action_ep = _stack_column(df["action.endpose"]).astype(np.float32)
        state_ep = _stack_column(df["observation.state.endpose"]).astype(np.float32)
        return ep_idx, {
            "action.endpose": feature_stats(action_ep),
            "observation.state.endpose": feature_stats(state_ep),
        }

    action_joints = _stack_column(df["action"])
    state_joints = _stack_column(df["observation.state"])

    action_ep = fk.compute_endpose(action_joints)        # (N, 14) float32
    state_ep = fk.compute_endpose(state_joints)

    df["action.endpose"] = list(action_ep)
    df["observation.state.endpose"] = list(state_ep)

    out_dir = os.path.dirname(parquet_path)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_endpose_", suffix=".parquet", dir=out_dir)
    os.close(fd)
    try:
        df.to_parquet(tmp_path)
        os.replace(tmp_path, parquet_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return ep_idx, {
        "action.endpose": feature_stats(action_ep),
        "observation.state.endpose": feature_stats(state_ep),
    }


# ----------------------------------------------------------------------------
# Metadata updates
# ----------------------------------------------------------------------------
def _atomic_write_text(path: str, text: str) -> None:
    out_dir = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_meta_", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def update_info_json(info_path: str) -> None:
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    features = info["features"]
    feature_def = {
        "dtype": "float32",
        "shape": [ENDPOSE_DIM],
        "names": [list(ENDPOSE_NAMES)],
    }
    features["action.endpose"] = json.loads(json.dumps(feature_def))
    features["observation.state.endpose"] = json.loads(json.dumps(feature_def))
    _atomic_write_text(info_path, json.dumps(info, indent=4))


def update_episodes_stats(stats_path: str, new_stats: Dict[int, Dict[str, Any]]) -> None:
    """Merge new per-episode endpose stats into meta/episodes_stats.jsonl."""
    lines_out: List[str] = []
    with open(stats_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ep = rec["episode_index"]
            if ep in new_stats:
                rec["stats"].update(new_stats[ep])
            lines_out.append(json.dumps(rec))
    _atomic_write_text(stats_path, "\n".join(lines_out) + "\n")


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def validate_parquet(parquet_path: str, root_p: np.ndarray) -> None:
    df = pd.read_parquet(parquet_path)
    for col in ("action.endpose", "observation.state.endpose"):
        assert col in df.columns, f"missing column {col} in {parquet_path}"
        sample = np.asarray(df[col].iloc[0])
        assert sample.shape == (ENDPOSE_DIM,), f"{col} bad shape {sample.shape}"
        assert sample.dtype == np.float32, f"{col} bad dtype {sample.dtype}"
    arr = _stack_column(df["action.endpose"])
    left_pos, right_pos = arr[:, 0:3], arr[:, 7:10]
    left_quat, right_quat = arr[:, 3:7], arr[:, 10:14]
    qn_l = np.linalg.norm(left_quat, axis=1)
    qn_r = np.linalg.norm(right_quat, axis=1)
    assert np.allclose(qn_l, 1.0, atol=1e-4) and np.allclose(qn_r, 1.0, atol=1e-4), \
        "quaternions are not unit-norm"
    dist_l = np.linalg.norm(left_pos - root_p, axis=1)
    dist_r = np.linalg.norm(right_pos - root_p, axis=1)
    print(f"  validate {os.path.basename(parquet_path)}: N={len(df)} "
          f"left_pos[0]={np.round(left_pos[0], 3)} right_pos[0]={np.round(right_pos[0], 3)} "
          f"max_dist_l={dist_l.max():.3f} max_dist_r={dist_r.max():.3f} "
          f"mean_left_y={left_pos[:,1].mean():.3f} mean_right_y={right_pos[:,1].mean():.3f}")
    assert dist_l.max() < 1.5 and dist_r.max() < 1.5, "end-effector implausibly far from base"
    assert left_pos[:, 1].mean() > right_pos[:, 1].mean(), \
        "expected left arm on +y side relative to right arm"


# ----------------------------------------------------------------------------
# Episode enumeration
# ----------------------------------------------------------------------------
def enumerate_episodes(dataset_root: str, info: Dict[str, Any]) -> List[Tuple[int, str]]:
    chunks_size = info["chunks_size"]
    total_episodes = info["total_episodes"]
    data_path_tmpl = info["data_path"]
    eps = []
    for ep in range(total_episodes):
        chunk = ep // chunks_size
        rel = data_path_tmpl.format(episode_chunk=chunk, episode_index=ep)
        eps.append((ep, os.path.join(dataset_root, rel)))
    return eps


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="aloha-agilex embodiment config.yml (endpose params).")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N episodes (dry-run).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip parquet that already have the endpose columns "
                             "(recompute their stats only). Makes re-runs resumable.")
    parser.add_argument("--no-meta-update", action="store_true",
                        help="Do not touch info.json / episodes_stats.jsonl.")
    args = parser.parse_args()

    info_path = os.path.join(args.dataset_root, "meta", "info.json")
    stats_path = os.path.join(args.dataset_root, "meta", "episodes_stats.jsonl")
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    episodes = enumerate_episodes(args.dataset_root, info)
    if args.limit is not None:
        episodes = episodes[: args.limit]

    # robot root position (for validation only); read without spinning up sapien.
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        root_p = np.array(yaml.safe_load(f).get("robot_pose", [[0, -0.65, 0, 1, 0, 0, 1]])[0][:3])

    print(f"Processing {len(episodes)} episodes from {args.dataset_root}")
    print(f"URDF: {args.urdf}")
    print(f"Config: {args.config}")
    print(f"Workers: {args.num_workers}  skip_existing={args.skip_existing}")

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(x, **kwargs):
            return x

    new_stats: Dict[int, Dict[str, Any]] = {}
    errors: List[Tuple[str, str]] = []

    if args.num_workers <= 1:
        _worker_init(args.urdf, args.config)
        for _ep, path in tqdm(episodes, desc="episodes"):
            try:
                idx, stats = process_episode(path, args.skip_existing)
                new_stats[idx] = stats
            except Exception as e:  # noqa: BLE001  (isolate per-episode failures)
                errors.append((path, repr(e)))
    else:
        # spawn: each worker builds its own SAPIEN scene in a fresh interpreter.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=ctx,
                                 initializer=_worker_init,
                                 initargs=(args.urdf, args.config)) as ex:
            futs = {ex.submit(process_episode, path, args.skip_existing): path
                    for _, path in episodes}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="episodes"):
                path = futs[fut]
                try:
                    idx, stats = fut.result()
                    new_stats[idx] = stats
                except Exception as e:  # noqa: BLE001
                    errors.append((path, repr(e)))

    print(f"Done parquet: {len(new_stats)} ok, {len(errors)} failed")
    if errors:
        for path, err in errors[:20]:
            print(f"  FAILED {path}: {err}")
        raise SystemExit(f"{len(errors)} episodes failed; metadata not updated.")

    print("Validating sample episodes...")
    for _, path in episodes[: min(3, len(episodes))]:
        validate_parquet(path, root_p)

    if not args.no_meta_update:
        print("Updating meta/episodes_stats.jsonl ...")
        update_episodes_stats(stats_path, new_stats)
        print("Updating meta/info.json ...")
        update_info_json(info_path)
        print("Metadata updated.")
    else:
        print("Skipping metadata update (--no-meta-update).")

    print("All done.")


if __name__ == "__main__":
    main()
