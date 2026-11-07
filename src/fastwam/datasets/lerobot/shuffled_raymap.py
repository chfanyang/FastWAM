"""Opt-in shuffled pixels or shuffled VAE cache; baseline datasets are unchanged."""
from .robot_video_dataset import RobotVideoDataset
from ..latent_cache import LatentCacheReader
from fastwam.representations.shuffled_raymap import ShuffledRaymapCodec


class ShuffledRaymapDataset(RobotVideoDataset):
    def __init__(self, *args, latent_cache_dir=None, latent_cache_only=False,
                 source_latent_cache_dir=None, **kwargs):
        # The existing cache validates the original data ordering and lets the
        # base class configure cached image/processor IO without decoding RGB.
        super().__init__(*args, latent_cache_dir=source_latent_cache_dir if latent_cache_only else None,
                         latent_cache_only=latent_cache_only, **kwargs)
        if self.raymap_representation != 'libero_rothko':
            raise ValueError('Only LIBERO absolute-RAY0 / relative-future baseline is supported')
        self.raymap_codec = ShuffledRaymapCodec(self.raymap_codec)
        self.latent_cache_dataset_contract = dict(self.latent_cache_dataset_contract,
            raymap_codec_metadata=self.raymap_codec.metadata())
        self.latent_cache = None
        self.latent_cache_metadata = None
        self.latent_cache_dir = latent_cache_dir
        if latent_cache_dir:
            self.latent_cache = LatentCacheReader(latent_cache_dir,
                expected_dataset_contract=self.latent_cache_dataset_contract)
            self.latent_cache_metadata = self.latent_cache.metadata
            if len(self.latent_cache) != len(self.lerobot_dataset) or self.latent_cache_metadata.get('benchmark_only'):
                raise ValueError('Partial diagnostic cache cannot be used for training')
        if latent_cache_only and self.latent_cache is None:
            raise ValueError('Shuffled training needs a completed shuffled action cache')
