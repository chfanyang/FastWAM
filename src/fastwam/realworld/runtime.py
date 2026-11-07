"""Model factory scoped exclusively to realworld configs."""
from dataclasses import asdict
from .dataset import make_codec


def create_model(rothko_norm_stats, rothko_config=None, **kwargs):
    # Absolute bounds live in the versioned realworld stats artifact, never LIBERO stats.
    from fastwam.runtime import create_fastwam_video_only_raymap
    codec = make_codec(rothko_norm_stats, rothko_config)
    return create_fastwam_video_only_raymap(
        rothko_norm_stats=rothko_norm_stats,
        rothko_config=asdict(codec.config), **kwargs)
