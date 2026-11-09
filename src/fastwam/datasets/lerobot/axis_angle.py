"""Opt-in Goal axis-angle dataset; baseline loaders and caches are unchanged."""
from pathlib import Path

from .robot_video_dataset import RobotVideoDataset
from ..latent_cache import LatentCacheReader
from fastwam.representations.libero_axis_angle import LiberoAxisAngleCodec


class AxisAngleDataset(RobotVideoDataset):
    def __init__(self, *args, latent_cache_dir=None, latent_cache_only=False,
                 source_latent_cache_dir=None, **kwargs):
        if latent_cache_dir and source_latent_cache_dir and Path(latent_cache_dir).resolve()==Path(source_latent_cache_dir).resolve():
            raise ValueError('Axis-angle cache must be separate from the baseline cache.')
        if latent_cache_only and not source_latent_cache_dir:
            raise ValueError('Cached axis-angle dataset requires the authenticated source RGB cache.')
        super().__init__(*args,
            latent_cache_dir=source_latent_cache_dir if latent_cache_only else None,
            latent_cache_only=latent_cache_only, **kwargs)
        if self.raymap_representation!='libero_rothko' or self.num_frames!=17:
            raise ValueError('This ablation requires LIBERO h16 absolute-RAY0 / relative-future data.')
        self.raymap_codec=LiberoAxisAngleCodec(self.raymap_codec)
        self.latent_cache_dataset_contract=dict(self.latent_cache_dataset_contract,
            raymap_representation=self.raymap_codec.representation,
            raymap_codec_metadata=self.raymap_codec.metadata())
        self.latent_cache=None
        self.latent_cache_metadata=None
        self.latent_cache_dir=latent_cache_dir
        if latent_cache_dir:
            self.latent_cache=LatentCacheReader(latent_cache_dir,
                expected_dataset_contract=self.latent_cache_dataset_contract)
            self.latent_cache_metadata=self.latent_cache.metadata
            if len(self.latent_cache)!=len(self.lerobot_dataset) or self.latent_cache_metadata.get('benchmark_only'):
                raise ValueError('Partial diagnostic axis-angle cache cannot be used for training.')
        if latent_cache_only and self.latent_cache is None:
            raise ValueError('Completed axis-angle action cache is required before cached training.')
