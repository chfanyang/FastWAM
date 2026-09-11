"""Opt-in episode-boundary resume; no simulator or ML dependencies."""
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

IDENTITY_KEYS = (
    'task_name', 'task_config', 'ckpt_setting', 'seed', 'instruction_type',
    'policy_name', 'sim_task', 'finetune_method', 'vae_safetensors_path',
    'allow_vae_mismatch', 'mixed_precision', 'dataset_stats_path',
    'action_horizon', 'replan_steps', 'num_inference_steps', 'sigma_shift',
    'text_cfg_scale', 'negative_prompt', 'rand_device', 'tiled',
    'skip_get_obs_within_replan', 'eval_step_limit',
    'resume_config_fingerprint',
)


def identity(options):
    return {key: options.get(key) for key in IDENTITY_KEYS}


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def progress_path(directory, phase):
    return Path(directory) / f'progress_{phase}.json'


def load_progress(path, expected, total, start_seed):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f'Resume requires {path}. Import legacy logs first, or enable '
            'resume_tracking for a fresh evaluation.'
        )
    state = json.loads(path.read_text())
    if state.get('version') != 1 or state.get('identity') != expected:
        differences = [k for k in IDENTITY_KEYS
                       if state.get('identity', {}).get(k) != expected.get(k)]
        raise ValueError(f'Resume settings mismatch: {differences}')
    completed, successes = state['completed'], state['successes']
    if not (0 <= successes <= completed <= total):
        raise ValueError('Invalid resume counts or requested total below completed count')
    if state['next_seed'] < start_seed:
        raise ValueError('Resume seed precedes initial seed')
    return state


def archive_uncommitted(directory, phase, completed):
    """Keep interrupted/uncommitted output, including opposite success suffixes."""
    directory = Path(directory)
    candidates = []
    randomized = phase == 'random'
    for p in directory.glob('episode*.mp4'):
        match = re.match(r'episode(\d+)(.*)\.mp4$', p.name)
        if not match or int(match[1]) < completed:
            continue
        if match[2] and f'randomized-{str(randomized).lower()}' not in match[2]:
            continue
        candidates.append(p)
    predictions = directory / 'predictions' / ('demo_randomized' if randomized else 'demo_clean')
    if predictions.exists():
        candidates.extend(p for p in predictions.glob('episode_*')
                          if p.name[8:].isdigit() and int(p.name[8:]) >= completed)
    if not candidates:
        return
    archive = directory / 'interrupted' / (phase + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    for p in candidates:
        target = archive / p.relative_to(directory)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(target))


def import_legacy(log_path, directory, phase, options):
    text = re.sub(r'\x1b\[[0-9;]*m', '', Path(log_path).read_text())
    rows = re.findall(r'Success rate:\s*(\d+)/(\d+).*?current seed:\s*(\d+)', text)
    successes, completed, last_seed = (map(int, rows[-1]) if rows else
                                      (0, 0, 100000 * (1 + options['seed']) - 1))
    # Validate every completed episode against the finalized outcome video.
    video_successes = 0
    for index in range(completed):
        videos = list(Path(directory).glob(
            f'episode{index}_randomized-{str(phase == "random").lower()}_success-*.mp4'))
        if len(videos) != 1 or videos[0].stat().st_size == 0:
            raise ValueError(f'Ambiguous/missing completed video: episode {index}')
        video_successes += videos[0].name.endswith('success-true.mp4')
    if video_successes != successes:
        raise ValueError('Log/video success counts disagree')
    state = dict(version=1, identity=identity(options), completed=completed,
                 successes=successes, next_seed=last_seed + 1,
                 imported_from=str(Path(log_path).resolve()))
    path = progress_path(directory, phase)
    if path.exists():
        raise FileExistsError(path)
    atomic_save(path, state)
    return state
