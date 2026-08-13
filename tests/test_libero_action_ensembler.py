import numpy as np

from experiments.libero.action_ensembler import AbsolutePoseActionEnsembler


def _pose(x: float, quaternion=(1.0, 0.0, 0.0, 0.0)) -> np.ndarray:
    return np.asarray([x, 0.0, 0.0, *quaternion], dtype=np.float32)


def test_absolute_pose_ensembler_prefers_newer_chunk():
    ensembler = AbsolutePoseActionEnsembler(decay=np.log(2.0))
    ensembler.add_actions(
        np.stack((_pose(-1.0), _pose(0.0))),
        np.asarray([[0.0], [0.0]], dtype=np.float32),
        start_timestamp=0,
    )
    ensembler.add_actions(
        np.stack((_pose(3.0),)),
        np.asarray([[1.0]], dtype=np.float32),
        start_timestamp=1,
    )

    pose, gripper = ensembler.get_action(1)

    # The newer prediction has twice the weight of the one-step-old prediction.
    np.testing.assert_allclose(pose[:3], [2.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(gripper, [2.0 / 3.0], atol=1e-6)


def test_absolute_pose_ensembler_aligns_equivalent_quaternion_signs():
    ensembler = AbsolutePoseActionEnsembler(decay=0.01)
    ensembler.add_actions(
        np.stack((_pose(0.0), _pose(0.0, (1.0, 0.0, 0.0, 0.0)))),
        np.asarray([[0.0], [0.0]], dtype=np.float32),
        start_timestamp=0,
    )
    ensembler.add_actions(
        np.stack((_pose(0.0, (-1.0, 0.0, 0.0, 0.0)),)),
        np.asarray([[0.0]], dtype=np.float32),
        start_timestamp=1,
    )

    pose, _ = ensembler.get_action(1)

    np.testing.assert_allclose(np.abs(pose[3]), 1.0, atol=1e-6)
    np.testing.assert_allclose(pose[4:], 0.0, atol=1e-6)


def test_absolute_pose_ensembler_cleanup_removes_expired_targets():
    ensembler = AbsolutePoseActionEnsembler()
    ensembler.add_actions(
        np.stack((_pose(0.0), _pose(1.0), _pose(2.0))),
        np.zeros((3, 1), dtype=np.float32),
        start_timestamp=4,
    )

    ensembler.cleanup(6)

    assert set(ensembler.action_cache) == {6}
