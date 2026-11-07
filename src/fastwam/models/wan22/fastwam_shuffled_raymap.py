"""Opt-in model with identical DiT parameters and invertible pixel permutation."""
from .fastwam_visual_action import FastWAMVideoOnlyRaymap
from fastwam.representations.shuffled_raymap import ShuffledRaymapCodec


class FastWAMShuffledRaymap(FastWAMVideoOnlyRaymap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.raymap_representation != 'libero_rothko' or self.rothko_decode_anchor_alpha != 0:
            raise ValueError('Shuffled ablation requires LIBERO codec and anchor=0')
        self.raymap_codec = ShuffledRaymapCodec(self.raymap_codec)
        # Distinct checkpoint identity also makes baseline loaders reject these weights.
        self.raymap_representation = 'libero_rothko_shuffled'

    def validate_latent_cache_metadata(self, metadata):
        super().validate_latent_cache_metadata(metadata)
        if metadata['dataset_contract']['raymap_codec_metadata'] != self.raymap_codec.metadata():
            raise ValueError('Shuffled cache codec/permutation identity mismatch')

    def _validate_visual_action_checkpoint_config(self, visual_config, *, checkpoint_path):
        expected = self.raymap_codec.metadata()
        for key in ('spatial_permutation', 'permutation_seed', 'permutation_sha256'):
            if visual_config.get(key) != expected[key]:
                raise ValueError(f'Shuffled checkpoint identity mismatch: {key}')
        super()._validate_visual_action_checkpoint_config(visual_config, checkpoint_path=checkpoint_path)
