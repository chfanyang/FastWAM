# Per-action RGB / Rothko comparison

Run `third_party/FastAPI/client_openpi0_v3.py` as `client_fast_ee.py` on the workstation. This version performs a synchronous move for each of the first eight predicted actions. After the move and any gripper command, it waits for a left-camera timestamp newer than a sample read after completion. It saves the raw RGB, training-preprocessed RGB and timing/pose metadata before executing the next action. Gripper settling is not guaranteed; its measured state is recorded. A failed move has no corresponding successful capture. A camera timeout stops the run instead of silently misaligning frames.

Automatic per-action recording does not require pressing r. The r/s independent continuous recorder remains available.

- Workstation: `inference_video/paired_frames/<session>/request_<id>/step_<i>_actual.png`
- Server: `evaluate_results/realworld/pick_place_server_gpu7_rgb20/predicted_rgb/<session>/request_<id>/step_<i>_pred_rgb.png`
- Same server folder: `step_<i>_pred_rothko.png` (full representation, not camera-cropped).
- `_pred_rgb_full.png` retains both cameras; `_pred_rgb.png` crops the left camera. Only steps 01–08 are saved by the current client. Neither reconstructed frame zero nor future steps 09–16 are paired.

Session is a new UUID on each client run. Request and step IDs align predictions and actual outcomes, not wall-clock time: inference, retimed motion and camera waiting all add latency. A move returning successfully is not proof of zero tracking error; actual/target poses and position error are recorded.

After a rollout, from the workstation FastAPI directory:

```bash
scp -r -P 39107 inference_video/paired_frames cfy@218.200.126.238:/mnt/hwdata/cfy/FastWAM/evaluate_results/realworld/paired_actual/
```

Then on the server, from `/mnt/hwdata/cfy/FastWAM`:

```bash
/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin/python scripts/realworld/compare_step_images.py \
  --actual-root evaluate_results/realworld/paired_actual/paired_frames \
  --prediction-root evaluate_results/realworld/pick_place_server_gpu7_rgb20/predicted_rgb \
  --output evaluate_results/realworld/step_comparisons
```

The tool writes three-panel PNGs: actual left RGB, predicted left RGB, full predicted Rothko. It skips incomplete pairs and reports their count. Workstation and server scripts need no access to each other's robot hardware.
