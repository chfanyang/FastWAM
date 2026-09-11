"""Import this repository's legacy RoboTwin per-task logs without touching videos.

Use --options-json containing the original deploy_policy options shared by tasks.
The import refuses ambiguous video outcomes and existing progress files.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'third_party/RoboTwin/script'))
from eval_resume import import_legacy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--options-json', required=True, type=Path)
    parser.add_argument('--phase', required=True, choices=['clean', 'random'])
    parser.add_argument('--tasks', nargs='+', required=True)
    args = parser.parse_args()
    options = json.loads(args.options_json.read_text())
    for task in args.tasks:
        logs = sorted(args.output_dir.glob(f'eval_{task}_*.log'))
        if len(logs) != 1:
            raise ValueError(f'Expected one original log for {task}, found {len(logs)}')
        per_task = dict(options, task_name=task,
                        task_config='demo_clean' if args.phase == 'clean' else 'demo_randomized')
        state = import_legacy(logs[0], args.output_dir / task, args.phase, per_task)
        print(f"{task}: {state['successes']}/{state['completed']}; next_seed={state['next_seed']}")


if __name__ == '__main__':
    main()
