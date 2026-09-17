"""Export one full training episode per primitive task for camera inspection."""
import argparse
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.dataset
    info = json.loads((root / 'meta/info.json').read_text())
    manifest = json.loads((root / 'manifests/episode_split_val01_seed42.json').read_text())
    languages = {r['task_index']: r['task'] for r in map(json.loads, (root / 'meta/tasks.jsonl').read_text().splitlines())}
    selected = {}
    for record in sorted(manifest['records'], key=lambda r: r['episode_index']):
        if record['split'] == 'train':
            selected.setdefault(record['base_task'], record)
    cameras = ['image', 'second_image', 'wrist_image']
    results = []
    for task, record in sorted(selected.items()):
        table = pq.read_table(root / record['data_path'], columns=cameras + ['task_index', 'frame_index'])
        assert len(table) == record['length']
        assert table['frame_index'].to_pylist() == list(range(len(table)))
        instructions = list(dict.fromkeys(languages[i] for i in table['task_index'].to_pylist()))
        stem = f"{task}--episode{record['episode_index']:06d}"
        output = args.output / (stem + '.mp4')
        command = ['ffmpeg', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                   '-s', '1440x512', '-r', str(info['fps']), '-i', 'pipe:0', '-an',
                   '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '18',
                   '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            for frame in range(len(table)):
                canvas = Image.new('RGB', (1440, 512))
                draw = ImageDraw.Draw(canvas)
                for index, camera in enumerate(cameras):
                    value = table[camera][frame].as_py()
                    with Image.open(io.BytesIO(value['bytes'])) as image:
                        image = image.convert('RGB')
                        assert image.size == (480, 480), image.size
                        canvas.paste(image, (index * 480, 32))
                    draw.text((index * 480 + 10, 10), f'{camera} | frame {frame:04d}', fill='white')
                process.stdin.write(np.asarray(canvas).tobytes())
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f'ffmpeg failed: {output}')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        (args.output / (stem + '.txt')).write_text('\n'.join(instructions) + '\n')
        results.append(dict(task=task, episode_index=record['episode_index'],
                            source=record['data_path'], split='train', instructions=instructions,
                            frames=len(table), fps=info['fps'], cameras_left_to_right=cameras,
                            video=output.name))
        print(f'{task}: {len(table)} frames -> {output}', flush=True)
    (args.output / 'manifest.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    lines = ['# VLABench three-camera episode previews', '',
             'One lowest-index training episode per task; all original frames at metadata FPS. '
             'Left to right: image, second_image, wrist_image. Native 480x480 images; labels outside images.', '']
    for result in results:
        lines.extend([f"## {result['task']} (episode {result['episode_index']})", '',
                      f"[Video]({result['video']})", '', *result['instructions'], ''])
    (args.output / 'README.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
