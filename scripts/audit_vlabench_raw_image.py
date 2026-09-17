"""Extract one complete raw episode from an existing gzip prefix for audit.

Does not download anything, mutate training data, or extract arbitrary tar paths.
"""

import json
import io
import shutil
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/vlabench_image_video_audit/raw_sample"


def extract_sample():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "source.json"
    if manifest.exists():
        record = json.loads(manifest.read_text())
        target = OUT / record["local_file"]
        if target.is_file() and target.stat().st_size == record["size"]:
            return target
        raise RuntimeError("Existing raw audit sample does not match source manifest")
    for prefix in (ROOT / "data/vlabench_raw_primitive/.cache").rglob("*.incomplete"):
        with prefix.open("rb") as f:
            if f.read(2) != b"\x1f\x8b":
                continue
        with tarfile.open(prefix, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".hdf5"):
                    continue
                if member.size > 512 * 1024**2:
                    raise RuntimeError("First HDF5 exceeds the 512 MiB audit sample budget")
                target = OUT / (Path(member.name).parent.name + "_" + Path(member.name).name)
                part = target.with_suffix(".partial")
                with archive.extractfile(member) as source, part.open("wb") as dest:
                    shutil.copyfileobj(source, dest, length=1024**2)
                if part.stat().st_size != member.size:
                    raise RuntimeError("Raw prefix does not contain a complete HDF5 sample")
                part.rename(target)
                manifest.write_text(json.dumps({
                    "source_prefix": str(prefix), "tar_member": member.name,
                    "local_file": target.name, "size": member.size,
                    "note": "Complete tar member; outer multi-part archive is incomplete",
                }, indent=2))
                return target
    raise FileNotFoundError("No local gzip prefix with a complete HDF5 member")


def compare_sample(target):
    image_root = ROOT / "data/vlabench_primitive_ft_lerobot"
    episodes = [json.loads(line) for line in
                (image_root / "meta/episodes.jsonl").read_text().splitlines()]
    with h5py.File(target, "r") as file:
        if len(file["data"]) != 1:
            raise ValueError("This audit expects one recording in the raw sample")
        group = next(iter(file["data"].values()))
        language = group["instruction"][0].decode()
        config = json.loads(group["meta_info/episode_config"][()].decode())
        base = np.asarray(config.get("robot", {}).get("position", [0, -0.4, 0.78]))
        ee = group["observation/ee_state"][:]
        trajectory = group["trajectory"][:]
        # Reproduce the current official conversion, including float32 in-place
        # subtraction, then cast exported pose fields to their declared dtype.
        xyz = ee[:, :3].copy()
        xyz -= base
        rot = Rotation.from_quat(ee[:, [4, 5, 6, 3]])
        state = np.column_stack([xyz, rot.as_euler("xyz"), ee[:, 7]]).astype(np.float32)
        action = np.column_stack([trajectory[:, :6], trajectory[:, -1] > 0.03]).astype(np.float32)
        candidates = [e for e in episodes if e["length"] == len(ee) and language in e["tasks"]]
        report = {"raw_file": str(target), "language": language,
                  "robot_translation": base.tolist(), "length": len(ee), "candidates": []}
        for episode in candidates:
            eid = episode["episode_index"]
            path = image_root / f"data/chunk-{eid // 1000:03d}/episode_{eid:06d}.parquet"
            table = pq.read_table(path).to_pydict()
            actual_state = np.asarray(table["state"], dtype=np.float32)
            actual_action = np.asarray(table["actions"], dtype=np.float32)
            result = {"image_episode": eid,
                      "state_max_abs_error": float(np.max(np.abs(actual_state - state))),
                      "action_max_abs_error": float(np.max(np.abs(actual_action - action))),
                      "camera_matches": {}, "lag_metrics": {}}
            for camera in ("image", "second_image", "wrist_image"):
                comparisons = []
                for camera_id in range(group["observation/rgb"].shape[1]):
                    errors = []
                    for index in (0, len(ee) // 2, len(ee) - 1):
                        im = np.asarray(Image.open(io.BytesIO(table[camera][index]["bytes"])).convert("RGB"))
                        raw_im = group["observation/rgb"][index, camera_id]
                        delta = np.abs(im.astype(np.int16) - raw_im.astype(np.int16))
                        errors.append((float(delta.mean()), int(delta.max())))
                    comparisons.append({"raw_camera_id": camera_id,
                                        "mean_abs_pixel_error": float(np.mean([e[0] for e in errors])),
                                        "max_abs_pixel_error": max(e[1] for e in errors)})
                result["camera_matches"][camera] = comparisons
            # Correlation diagnostic only: lag closeness cannot alone determine
            # whether the raw recording is pre-action or post-action.
            for lag in (-1, 0, 1):
                lo, hi = max(0, -lag), min(len(ee), len(ee) - lag)
                obs = actual_state[lo + lag:hi + lag]
                act = actual_action[lo:hi]
                angle = (Rotation.from_euler("xyz", obs[:, 3:6]).inv()
                         * Rotation.from_euler("xyz", act[:, 3:6])).magnitude()
                result["lag_metrics"][str(lag)] = {
                    "meaning": f"state[t+({lag})] compared with action[t]",
                    "position_l2_mean_mm": float(np.linalg.norm(obs[:, :3] - act[:, :3], axis=1).mean() * 1000),
                    "rotation_geodesic_mean_deg": float(np.degrees(angle).mean()),
                    "gripper_equal_fraction": float(np.mean(obs[:, 6] == act[:, 6])),
                    "gripper_inverted_equal_fraction": float(np.mean(1 - obs[:, 6] == act[:, 6])),
                }
            result["gripper_transitions"] = [
                {"frame": i, "state_gripper": float(actual_state[i, 6]),
                 "action_gripper": float(actual_action[i, 6]),
                 "raw_command": trajectory[i, -2:].tolist()}
                for i in range(len(ee)) if i == 0 or
                actual_state[i, 6] != actual_state[i-1, 6] or
                actual_action[i, 6] != actual_action[i-1, 6]
            ]
            report["candidates"].append(result)
    (OUT / "comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    target = extract_sample()
    print("Sample:", target, flush=True)
    with h5py.File(target, "r") as f:
        def show(name, value):
            if isinstance(value, h5py.Dataset):
                print(name, value.shape, value.dtype, flush=True)
        f.visititems(show)
    compare_sample(target)
