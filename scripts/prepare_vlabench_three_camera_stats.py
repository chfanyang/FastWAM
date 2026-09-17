"""Rebuild spatial stats from existing ALL-task quantiles; never refit on one task."""
import json
from pathlib import Path
import torch
from fastwam.representations.rothko import RothkoNormStats, _center_and_read_masks, _border_mask
from fastwam.representations.vlabench_rothko import VLABenchRothkoCodec, VLABenchRothkoCodecConfig


def main():
    root = Path("data/vlabench_primitive_ft_lerobot/norm_stats")
    source = root / "vlabench_rothko_q99p95_h16_224x448_centerfrac05_train99.pt"
    output = root / "vlabench_rothko_q99p95_h16_192x576_centerfrac05_train99.pt"
    old = RothkoNormStats.load(source)
    VLABenchRothkoCodec(norm_stats=old, expected_action_horizon=16)
    cfg = VLABenchRothkoCodecConfig(image_height=192, image_width=576,
                                  tile_height=192, tile_width=192, horizontal_copies=3)
    lo = torch.full((1, 3, 192, 192), -cfg.dir_scale)
    hi = -lo
    mask, _, _ = _center_and_read_masks(192, 192, center_frac=cfg.center_frac,
                                      boundary_margin=0, outer_margin=0, device=lo.device)
    bounds = torch.tensor(old.metadata["translation_abs_bounds_xyz_m"])
    # Verify the reused physical ranges against the actual old tensors.
    assert torch.equal(bounds * cfg.center_scale, old.hi[0, :, 112, 112])
    lo[..., mask] = -bounds[None, :, None] * cfg.center_scale
    hi[..., mask] = bounds[None, :, None] * cfg.center_scale
    border = _border_mask(192, 192, cfg.outer_margin, lo.device)
    lo[..., border], hi[..., border] = -1, 1
    metadata = {**old.metadata, **VLABenchRothkoCodec(config=cfg).metadata(),
                "image_size": [192, 576], "tile_size": [192, 192],
                "source_stats_fingerprint": old.fingerprint(),
                "geometry_rebuilt_without_refitting": True}
    payload = dict(lo=torch.cat([lo]*3, -1), hi=torch.cat([hi]*3, -1), metadata=metadata)
    if output.exists():
        existing = torch.load(output, weights_only=False, map_location="cpu")
        if existing["metadata"] != metadata or any(not torch.equal(existing[k], payload[k]) for k in ("lo", "hi")):
            raise ValueError("Existing stats differ; refusing to overwrite")
    else:
        temporary = output.with_suffix(".partial")
        torch.save(payload, temporary)
        temporary.replace(output)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
    VLABenchRothkoCodec(config=cfg, norm_stats=output, expected_action_horizon=16)
    print(output, "physical bounds unchanged:", bounds.tolist(), flush=True)


if __name__ == "__main__":
    main()
