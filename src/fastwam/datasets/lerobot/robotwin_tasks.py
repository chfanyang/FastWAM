"""RoboTwin task-to-episode mapping for the released FastWAM dataset."""

from collections.abc import Sequence


EPISODES_PER_ROBOTWIN_TASK = 550

# The released dataset contains 27,500 episodes ordered in these 50 contiguous
# task blocks. Each block contains exactly 550 episodes.
ROBOTWIN_TASK_NAMES = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
)

ROBOTWIN_TASK_TO_EPISODE_RANGE = {
    task_name: range(
        task_index * EPISODES_PER_ROBOTWIN_TASK,
        (task_index + 1) * EPISODES_PER_ROBOTWIN_TASK,
    )
    for task_index, task_name in enumerate(ROBOTWIN_TASK_NAMES)
}


def resolve_robotwin_episode_indices(task_names: Sequence[str] | None, variant: str = "all") -> list[int]:
    """Resolve released RoboTwin task names to ordered episode indices."""
    if variant not in {"all", "clean", "randomized"}:
        raise ValueError(f"Unknown RoboTwin data variant: {variant!r}")
    if task_names is None:
        task_names = ROBOTWIN_TASK_NAMES
    if isinstance(task_names, str):
        raise TypeError("robotwin_task_names must be a sequence of task names, not a string")

    normalized_names = [str(name).strip() for name in task_names]
    if not normalized_names or any(not name for name in normalized_names):
        raise ValueError("robotwin_task_names must contain at least one non-empty task name")

    unknown = sorted(set(normalized_names).difference(ROBOTWIN_TASK_TO_EPISODE_RANGE))
    if unknown:
        available = ", ".join(ROBOTWIN_TASK_NAMES)
        raise ValueError(f"Unknown RoboTwin task names: {unknown}. Available tasks: {available}")

    episode_indices: list[int] = []
    seen = set()
    for task_name in normalized_names:
        if task_name in seen:
            continue
        seen.add(task_name)
        block = ROBOTWIN_TASK_TO_EPISODE_RANGE[task_name]
        selected = block[:50] if variant == "clean" else block[50:] if variant == "randomized" else block
        episode_indices.extend(selected)
    return episode_indices
