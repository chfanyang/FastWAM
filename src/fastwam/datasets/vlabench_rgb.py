"""Opt-in three-camera VLABench canvas shared by dataset and deployment."""
import torch
from .libero_rgb import _to_chw_float, _resize

CAMERA_KEYS = ("image", "second_image", "wrist_image")


def build_vlabench_rgb_canvas(images, camera_size=192):
    tiles = [_resize(_to_chw_float(images[key]), camera_size, camera_size)
             for key in CAMERA_KEYS]
    if any(tile.shape != tiles[0].shape for tile in tiles):
        raise ValueError("VLABench camera shapes must match")
    return torch.cat(tiles, -1).mul(2).sub(1)
