from experiments.libero.libero_utils import (
    evaluation_video_stem,
    sanitize_task_name,
)


def test_libero_rollout_video_name_matches_compact_format():
    assert evaluation_video_stem(
        "Open the middle drawer of the cabinet.",
        "task0_trial34",
        False,
    ) == "open_the_middle_drawer_of_the_cabinet--task0_trial34--success=False"


def test_libero_task_name_sanitization_collapses_punctuation():
    assert sanitize_task_name("  Put bowl / plate... now!  ") == "put_bowl_plate_now"
