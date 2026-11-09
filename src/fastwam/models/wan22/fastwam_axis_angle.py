"""Independent codec adapter; DiT architecture, loss, mask and inference stay inherited."""
from .fastwam_visual_action import FastWAMVideoOnlyRaymap
from fastwam.representations.libero_axis_angle import LiberoAxisAngleCodec


class FastWAMAxisAngle(FastWAMVideoOnlyRaymap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.raymap_representation!='libero_rothko':
            raise ValueError('Axis-angle adapter requires baseline LIBERO pose semantics.')
        self.raymap_codec=LiberoAxisAngleCodec(self.raymap_codec)
        self.raymap_representation=self.raymap_codec.representation

    def validate_latent_cache_metadata(self, metadata):
        super().validate_latent_cache_metadata(metadata)
        contract=metadata.get('dataset_contract',{})
        if (contract.get('raymap_representation')!=self.raymap_representation
                or contract.get('raymap_codec_metadata')!=self.raymap_codec.metadata()):
            raise ValueError('Axis-angle cache representation/geometry/conversion identity mismatch.')
        if metadata.get('benchmark_only'):
            raise ValueError('Diagnostic cache cannot be used for formal training.')

    def _validate_visual_action_checkpoint_config(self, visual_config, *, checkpoint_path):
        expected=self.raymap_codec.metadata()
        required=('rotation_encoding','rotation_scale_rad','rotation_conversion',
            'rotation_write_region','rotation_read_region','future_rotation_frame',
            'rotation_decoder','rotation_decoder_reference')
        for key in required:
            if visual_config.get(key)!=expected[key]:
                raise ValueError(f'Axis-angle checkpoint identity mismatch: {key}')
        super()._validate_visual_action_checkpoint_config(visual_config,checkpoint_path=checkpoint_path)
