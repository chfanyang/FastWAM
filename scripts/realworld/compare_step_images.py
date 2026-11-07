#!/usr/bin/env python3
"""Join actual/predicted images by session, request and executed action index."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actual-root', type=Path, required=True,
                        help='Workstation paired_frames directory, copied to this machine')
    parser.add_argument('--prediction-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    count, missing = 0, 0
    for metadata_path in sorted(args.actual_root.glob('*/request_*/metadata.json')):
        meta = json.loads(metadata_path.read_text())
        relative = Path(meta['session']) / f"request_{int(meta['request_id']):06d}"
        predicted = args.prediction_root / relative
        for actual_path in sorted(metadata_path.parent.glob('step_*_actual.png')):
            stem = actual_path.name.removesuffix('_actual.png')
            paths = [actual_path, predicted / f'{stem}_pred_rgb.png',
                     predicted / f'{stem}_pred_rothko.png']
            actual_meta = metadata_path.parent / f'{stem}_actual.json'
            if not all(p.is_file() for p in paths) or not actual_meta.is_file():
                missing += 1
                continue
            panels = []
            for path in paths:
                with Image.open(path) as image:
                    panels.append(image.convert('RGB'))
            height = max(im.height for im in panels)
            canvas = Image.new('RGB', (sum(im.width for im in panels), height + 48), 'white')
            draw = ImageDraw.Draw(canvas)
            x = 0
            for panel, label in zip(panels, ('Actual left', 'Predicted left', 'Predicted Rothko (full)')):
                canvas.paste(panel, (x, 48))
                draw.text((x + 4, 26), label, fill='black')
                x += panel.width
            draw.text((4, 6), f"request {meta['request_id']} / {stem} - action-aligned, not time-aligned",
                      fill='black')
            destination = args.output / relative
            destination.mkdir(parents=True, exist_ok=True)
            canvas.save(destination / f'{stem}_comparison.png')
            count += 1
    print(f'Saved {count} comparisons; skipped {missing} incomplete pairs: {args.output}')


if __name__ == '__main__':
    main()
