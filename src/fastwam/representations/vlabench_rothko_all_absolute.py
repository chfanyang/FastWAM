"""Opt-in full absolute VLABench poses; independent of relative/mixed codecs."""
import torch
from .vlabench_rothko import VLABenchRothkoCodec
from .rothko import _center_and_read_masks, quaternion_wxyz_to_matrix, matrix_to_quaternion_wxyz


class VLABenchAllAbsoluteRothkoCodec(VLABenchRothkoCodec):
    representation = 'vlabench_rothko_all_absolute'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.frame0_pose_mode != 'absolute':
            raise ValueError('Full absolute codec requires absolute RAY0 and explicit position bounds')
        if self.decode_mode != 'legacy' or self.decode_anchor_alpha != 0:
            raise ValueError('Full absolute codec requires legacy decoding with anchor=0')

    def metadata(self):
        return {**super().metadata(), 'pose_mode': 'all_absolute',
                'future_pose_mode': 'absolute', 'position_bounds_scope': 'all_17_frames'}

    def _validate_stats_metadata(self, metadata):
        super()._validate_stats_metadata(metadata)
        for key, expected in dict(pose_mode='all_absolute', future_pose_mode='absolute',
                                  frame0_pose_mode='absolute', position_bounds_scope='all_17_frames').items():
            if metadata.get(key) != expected:
                raise ValueError(f'Full absolute stats mismatch: {key}')
        for key in ('absolute_position_min', 'absolute_position_max'):
            if metadata.get(key) != list(getattr(self.config, key) or ()):
                raise ValueError(f'Full absolute stats mismatch: {key}')
        lo, hi = self._absolute_frame_bounds(self.norm_stats.lo)
        if not torch.equal(lo, self.norm_stats.lo) or not torch.equal(hi, self.norm_stats.hi):
            raise ValueError('Full absolute stats tensors disagree with position/direction bounds')

    def _encode_tile_raw(self, pose7):
        cfg = self.config
        rotation = quaternion_wxyz_to_matrix(pose7[..., 3:])
        canonical = self._canonical_directions(pose7.device, pose7.dtype)
        tile = torch.einsum('btij,hwj->btihw', rotation, canonical) * cfg.dir_scale
        center, _, _ = _center_and_read_masks(cfg.tile_height, cfg.tile_width,
            center_frac=cfg.center_frac, boundary_margin=0, outer_margin=0, device=pose7.device)
        tile[..., center] = (pose7[..., :3] * cfg.center_scale)[..., None]
        return tile

    def normalize_raw(self, raw):
        squeeze = raw.ndim == 4
        if squeeze: raw = raw.unsqueeze(0)
        if raw.ndim != 5 or raw.shape[2:] != (3, self.config.image_height, self.config.image_width):
            raise ValueError('Full absolute raw shape mismatch')
        lo, hi = self._expanded_stats(device=raw.device, dtype=raw.dtype)
        value = (2 * (raw-lo[:, None]) / (hi-lo)[:, None] - 1).clamp(-1, 1)
        return value[0] if squeeze else value

    def denormalize(self, normalized):
        squeeze = normalized.ndim == 4
        if squeeze: normalized = normalized.unsqueeze(0)
        if normalized.ndim != 5 or normalized.shape[2:] != (3, self.config.image_height, self.config.image_width):
            raise ValueError('Full absolute normalized shape mismatch')
        lo, hi = self._expanded_stats(device=normalized.device, dtype=normalized.dtype)
        raw = (normalized+1)*.5*(hi-lo)[:, None]+lo[:, None]
        return raw[0] if squeeze else raw

    def decode(self, video, current_pose7=None):
        # Every frame is decoded independently against canonical directions.
        # Physical current pose and reconstructed RAY0 do not anchor future poses.
        squeeze = video.ndim == 4
        if squeeze: video = video.unsqueeze(0)
        cfg = self.config
        if video.ndim != 5 or video.shape[1] != 3 or video.shape[-2:] != (cfg.image_height, cfg.image_width):
            raise ValueError('Full absolute video shape mismatch')
        normalized = video.permute(0, 2, 1, 3, 4)
        grip = self.read_gripper(normalized)
        raw = self.denormalize(normalized)
        tile = torch.stack(raw.split(cfg.tile_width, -1)).mean(0)
        _, center, direction = _center_and_read_masks(cfg.tile_height, cfg.tile_width,
            center_frac=cfg.center_frac, boundary_margin=cfg.boundary_margin,
            outer_margin=cfg.outer_margin, device=raw.device)
        values = tile.permute(0, 1, 3, 4, 2).double()
        xyz = values[:, :, center].median(2).values / cfg.center_scale
        rays = values[:, :, direction]
        rays = rays / rays.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        canonical = self._canonical_directions(raw.device, torch.float64)[direction]
        corr = torch.einsum('btpi,pj->btij', rays, canonical) / canonical.shape[0]
        u, _, vh = torch.linalg.svd(corr)
        fix = torch.eye(3, device=raw.device, dtype=torch.float64).expand(*u.shape[:-2], 3, 3).clone()
        fix[..., 2, 2] = torch.where(torch.linalg.det(u @ vh) >= 0, 1., -1.)
        pose = torch.cat((xyz, matrix_to_quaternion_wxyz(u @ fix @ vh)), -1).to(video.dtype)
        return (pose[0], grip[0]) if squeeze else (pose, grip)
