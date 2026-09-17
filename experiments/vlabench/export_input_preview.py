"""Export the exact RGB canvas from an existing diagnostic observation pair."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from fastwam.datasets.libero_rgb import build_libero_rgb_canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('observation_pair', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    pair = np.array(Image.open(args.observation_pair).convert('RGB'))
    width = pair.shape[1]
    if width % 2:
        raise ValueError('Expected equally wide camera2 and camera3 images')
    canvas = build_libero_rgb_canvas(pair[:, :width//2], pair[:, width//2:])
    pixels = ((canvas + 1) * 127.5).round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(args.output)
    print(args.output, pixels.shape)


if __name__ == '__main__':
    main()
