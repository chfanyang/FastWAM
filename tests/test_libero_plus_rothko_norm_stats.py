import unittest

import numpy as np

from scripts.compute_libero_plus_rothko_norm_stats import (
    relative_positions_with_tail_replication,
)


class LiberoPlusRothkoNormStatsTest(unittest.TestCase):
    def test_every_frame_is_a_start_and_tail_replicates(self) -> None:
        state = np.zeros((3, 7), dtype=np.float64)
        state[:, 3] = 1.0
        target = state.copy()
        target[:, 0] = (1.0, 2.0, 3.0)
        relative = relative_positions_with_tail_replication(state, target, horizon=2)
        self.assertEqual(relative.shape, (3, 2, 3))
        np.testing.assert_allclose(relative[:, :, 0], ((1, 2), (2, 3), (3, 3)))

    def test_positions_are_expressed_in_current_ee_frame(self) -> None:
        state = np.zeros((1, 7), dtype=np.float64)
        # +90 degrees around z in wxyz order.
        state[0, 3:7] = (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5))
        target = state.copy()
        target[0, :3] = (1.0, 0.0, 0.0)
        relative = relative_positions_with_tail_replication(state, target, horizon=1)
        np.testing.assert_allclose(relative[0, 0], (0.0, -1.0, 0.0), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
