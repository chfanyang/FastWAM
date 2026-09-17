"""Export same-frame RGB layout comparisons without altering training data."""
import io
import json
import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def export_preview(root, output, task, episode, frame=0):
    output.mkdir(parents=True, exist_ok=False)
    keys = ['image', 'second_image', 'wrist_image']
    source = root / f'data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet'
    table = pq.read_table(source, columns=keys + ['task_index'])
    images = []
    for key in keys:
        with Image.open(io.BytesIO(table[key][frame].as_py()['bytes'])) as image:
            images.append(torch.from_numpy(np.array(image.convert('RGB'), copy=True)).permute(2, 0, 1))
    original = torch.cat(images, dim=-1).float().div(255).unsqueeze(0)
    languages = {r['task_index']: r['task'] for r in map(json.loads, (root / 'meta/tasks.jsonl').read_text().splitlines())}
    language = languages[table['task_index'][frame].as_py()]
    layouts = [(224, 672), (192, 576), (224, 448)]
    sheet = Image.new('RGB', (704, sum(h + 60 for h, _ in layouts) + 90), '#202020')
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 12), f'{task} | episode {episode} | frame {frame}', fill='white')
    draw.text((16, 32), 'Left to right: image | second_image | wrist_image', fill='white')
    y = 70
    records = []
    for h, w in layouts:
        resized = F.interpolate(original, size=(h, w), mode='bilinear', align_corners=False, antialias=True)
        array = resized[0].permute(1, 2, 0).mul(255).round().clamp(0, 255).byte().numpy()
        preview = Image.fromarray(array)
        filename = f'episode{episode:06d}_frame{frame:04d}_{h}x{w}.png'
        preview.save(output / filename)
        ratio = h * w / (224 * 448)
        draw.text((16, y), f'HxW {h}x{w} | pixels vs current 224x448: {ratio:.3f}x', fill='white')
        draw.text((16, y + 16), 'Aspect ratio preserved' if w == 3*h else 'Horizontal compression: objects become narrower', fill='white')
        sheet.paste(preview, (16, y + 38))
        y += h + 60
        records.append(dict(height=h, width=w, file=filename, pixel_ratio=ratio))
    sheet.save(output / 'comparison.png')
    metadata = dict(task=task, episode=episode, frame=frame, source=str(source), language=language,
                    cameras=keys, interpolation='torch bilinear, align_corners=False, antialias=True',
                    layouts=records, note='RGB previews only; rows shown at native pixel sizes, no model changes')
    (output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    (output / 'language.txt').write_text(language + '\n')
    print(output / 'comparison.png', flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all-tasks', action='store_true')
    args = parser.parse_args()
    root = Path('data/vlabench_primitive_ft_lerobot')
    if not args.all_tasks:
        export_preview(root, Path('evaluate_results/vlabench/select_book_three_camera_layout_comparison'), 'select_book', 4)
        return
    output = Path('evaluate_results/vlabench/ten_tasks_three_camera_layout_comparison')
    output.mkdir(parents=True, exist_ok=False)
    selected = json.loads(Path('evaluate_results/vlabench/dataset_three_camera_previews_10tasks/manifest.json').read_text())
    records = []
    lines = ['# VLABench：10 个任务的三种 RGB 布局', '',
             '每个任务取对应预览视频的第 0 帧；左到右为 image、second_image、wrist_image。', '',
             '每张对比图从上到下为 224×672、192×576、224×448（高×宽）。', '']
    for item in selected:
        task = item['task']
        record = export_preview(root, output / task, task, item['episode_index'])
        records.append(record)
        lines.extend([f'## {task}', '', record['language'], '',
                      f"Episode {record['episode']}，frame 0。", '',
                      f'![{task}]({task}/comparison.png)', ''])
    (output / 'manifest.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    (output / 'README.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
