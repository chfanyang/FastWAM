"""Fixed RGB-pixel permutation, independently within each repeated LIBERO tile."""
import hashlib
import torch


class ShuffledRaymapCodec:
    def __init__(self, base):
        if (base.config.tile_height, base.config.tile_width, base.config.image_width) != (224, 224, 448):
            raise ValueError('Shuffled ablation requires two 224x224 tiles')
        if base.config.frame0_pose_mode != 'absolute':
            raise ValueError('This ablation requires absolute RAY0')
        self.base = base
        self.permutation = torch.randperm(224 * 224, generator=torch.Generator().manual_seed(42))
        self.inverse = self.permutation.argsort()

    def __getattr__(self, name):
        if name == 'base':
            raise AttributeError(name)
        return getattr(self.base, name)

    def transform(self, video, inverse=False):
        if video.shape[-2:] != (224, 448):
            raise ValueError(f'Unexpected canvas {video.shape}')
        perm = (self.inverse if inverse else self.permutation).to(video.device)
        return torch.cat([tile.flatten(-2)[..., perm].reshape_as(tile)
                          for tile in video.split(224, -1)], -1)

    def encode(self, *args, **kwargs):
        return self.transform(self.base.encode(*args, **kwargs))

    def decode(self, video, *args, **kwargs):
        return self.base.decode(self.transform(video, inverse=True), *args, **kwargs)

    def read_gripper(self, normalized):
        return self.base.read_gripper(self.transform(normalized, inverse=True))

    def set_decode_anchor_video(self, video):
        return self.base.set_decode_anchor_video(self.transform(video, inverse=True))

    def metadata(self):
        return dict(self.base.metadata(), spatial_permutation='fixed_uv_within_tile_rgb_together_v1',
                    permutation_seed=42,
                    permutation_sha256=hashlib.sha256(self.permutation.numpy().tobytes()).hexdigest())
