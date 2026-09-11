"""Read-only alignment audit of official clean HDF5 against released LeRobot.

Writes a JSON report only; never modifies either input dataset. RGB MAE uses
PyAV/OpenCV BGR decoding and area resizing to 160x120 for a temporal alignment test,
not a pixel-identity test across JPEG and MP4 compression.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import cv2
import av
import h5py
import numpy as np
import pandas as pd


def audit(job):
    raw_path, root, episode = job
    cv2.setNumThreads(1)
    root = Path(root)
    chunk = f"chunk-{episode // 1000:03d}"
    name = f"episode_{episode:06d}"
    d = pd.read_parquet(root / "data" / chunk / f"{name}.parquet")
    state = np.stack(d["observation.state"])
    action = np.stack(d["action"])
    state_ee = np.stack(d["observation.state.endpose"]).astype(np.float64)
    action_ee = np.stack(d["action.endpose"]).astype(np.float64)
    result = {"hdf5": str(raw_path), "lerobot_episode": episode, "rows": len(d)}
    with h5py.File(raw_path) as f:
        joints = f["joint_action/vector"][:]
        native = np.concatenate([f["endpose/left_endpose"][:],
                                 f["endpose/right_endpose"][:]], axis=1)
        assert len(joints) == len(d) + 1, (raw_path, len(joints), len(d))
        result["raw_frames"] = len(joints)
        result["joint_max_abs_error"] = {
            "state_vs_raw_t": float(np.abs(state - joints[:-1]).max()),
            "action_vs_raw_t_plus_1": float(np.abs(action - joints[1:]).max()),
            "action_vs_next_state": float(np.abs(action[:-1] - state[1:]).max()),
        }
        assert result["joint_max_abs_error"]["state_vs_raw_t"] < 1e-6
        assert result["joint_max_abs_error"]["action_vs_raw_t_plus_1"] < 1e-6
        result["fk_action_vs_next_state_max_abs"] = float(
            np.abs(action_ee[:-1] - state_ee[1:]).max())
        result["ee_offsets"] = {}
        indices = np.arange(2, len(d) - 2)
        for offset in range(-2, 3):
            errors = []
            for k in (0, 7):
                x, y = state_ee[indices, k:k+7], native[indices+offset, k:k+7]
                q = x[:, 3:] / np.linalg.norm(x[:, 3:], axis=1, keepdims=True)
                r = y[:, 3:] / np.linalg.norm(y[:, 3:], axis=1, keepdims=True)
                errors.extend(np.stack([
                    np.linalg.norm(x[:, :3] - y[:, :3], axis=1) * 1000,
                    np.degrees(2 * np.arccos(np.clip(np.abs((q*r).sum(1)), 0, 1))),
                ], axis=1).tolist())
            result["ee_offsets"][str(offset)] = errors
        result["rgb"] = {}
        for raw_camera, camera in (("head_camera", "cam_high"),
                                   ("left_camera", "cam_left_wrist"),
                                   ("right_camera", "cam_right_wrist")):
            raw = []
            for encoded in f[f"observation/{raw_camera}/rgb"]:
                frame = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
                assert frame is not None
                raw_shape = list(frame.shape)
                raw.append(cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA))
            raw = np.asarray(raw, dtype=np.float32)
            path = root / "videos" / chunk / f"observation.images.{camera}" / f"{name}.mp4"
            video = []
            with av.open(str(path)) as capture:
                stream = capture.streams.video[0]
                stream.codec_context.thread_count = 1
                for decoded in capture.decode(stream):
                    frame = decoded.to_ndarray(format="bgr24")
                    video_shape = list(frame.shape)
                    video.append(cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA))
            assert len(video) >= len(d), (path, len(video), len(d))
            video = np.asarray(video, dtype=np.float32)
            maes = np.stack([np.abs(video[indices] - raw[indices+o]).mean((1, 2, 3))
                             for o in range(-2, 3)], axis=1)
            result["rgb"][camera] = {
                "raw_shape": raw_shape, "video_shape": video_shape,
                "video_frames": len(video), "compared_frames": len(indices),
                "mae_by_offset": maes.mean(0).tolist(),
                "best_offset": int(np.argmin(maes.mean(0))) - 2,
                "frame_best_offset_counts": np.bincount(maes.argmin(1), minlength=5).tolist(),
            }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", required=True, type=Path)
    p.add_argument("--lerobot-root", required=True, type=Path)
    p.add_argument("--first-episode", required=True, type=int)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    files = sorted(args.raw_dir.glob("episode*.hdf5"))
    assert files, args.raw_dir
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(audit, [(f, args.lerobot_root, args.first_episode+i)
                                 for i, f in enumerate(files)]):
            results.append(r)
            print(f"Audited {len(results)}/{len(files)}", flush=True)
    summary = {"episodes": len(results), "offsets": [-2, -1, 0, 1, 2],
               "rgb_note": "BGR MAE on 0..255, area-resized to 160x120; common interior frames",
               "ee": {}, "rgb": {}}
    for offset in summary["offsets"]:
        errors = np.concatenate([r["ee_offsets"][str(offset)] for r in results])
        summary["ee"][str(offset)] = {
            "arm_frames": len(errors), "mean_mm_deg": errors.mean(0).tolist(),
            "p95_mm_deg": np.quantile(errors, .95, axis=0).tolist(),
            "max_mm_deg": errors.max(0).tolist()}
    for camera in results[0]["rgb"]:
        rows = [r["rgb"][camera] for r in results]
        summary["rgb"][camera] = {
            "mae_by_offset": np.average([r["mae_by_offset"] for r in rows], axis=0,
                                        weights=[r["compared_frames"] for r in rows]).tolist(),
            "episode_best_offset_counts": {str(o): sum(r["best_offset"] == o for r in rows)
                                           for o in summary["offsets"]}}
    # Keep summaries per episode, not thousands of redundant pose-error rows.
    for r in results:
        del r["ee_offsets"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "episodes": results}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
