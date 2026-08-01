import unittest

import numpy as np
import torch

from fastwam.datasets.libero_rgb import build_libero_rgb_canvas


class LiberoRgbTest(unittest.TestCase):
    def test_numpy_and_tensor_paths_match(self) -> None:
        rng = np.random.default_rng(11)
        agent = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
        wrist = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
        numpy_path = build_libero_rgb_canvas(agent, wrist)
        tensor_path = build_libero_rgb_canvas(
            torch.from_numpy(agent).permute(2, 0, 1),
            torch.from_numpy(wrist).permute(2, 0, 1),
        )
        torch.testing.assert_close(numpy_path, tensor_path)
        self.assertEqual(tuple(numpy_path.shape), (3, 224, 448))


if __name__ == "__main__":
    unittest.main()
