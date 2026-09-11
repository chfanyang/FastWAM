import pytest
from fastwam.datasets.lerobot.robotwin_c2r_subset import select_c2r_windows


def test_subset_preserves_full_cache_indices():
    episodes = [dict(episode_index=i, task_name=t, rows=n)
                for i, (t, n) in enumerate([('a', 2), ('b', 3), ('a', 1)])]
    indices, counts, total = select_c2r_windows(episodes, ['a'])
    assert indices == [0, 1, 5]
    assert counts == {'a': 3}
    assert total == 6
    with pytest.raises(ValueError):
        select_c2r_windows(episodes, ['missing'])
