#!/usr/bin/env python3
"""Reproduce paper/benchmark_progress_record typography using matched robot steps."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
FONT = '/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf'
TILE, GAP, MARGIN, LABEL, ROW_GAP, HEADER = 224, 8, 20, 200, 16, 76


def square_rgb(path):
    """Apply the real-world training resize: contain, centered black padding."""
    with Image.open(path) as raw:
        resized = ImageOps.contain(raw.convert('RGB'), (TILE, TILE), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (TILE, TILE), 'black')
    canvas.paste(resized, ((TILE-resized.width)//2, (TILE-resized.height)//2))
    return canvas


def render(items, destination):
    width = MARGIN * 2 + LABEL + len(items) * TILE + (len(items)-1) * GAP
    height = HEADER + MARGIN * 2 + 3 * TILE + 2 * ROW_GAP
    canvas = Image.new('RGB', (width, height), 'white')
    scale = 4
    header = Image.new('RGB', (width*scale, HEADER*scale), 'white')
    draw = ImageDraw.Draw(header)
    def centered(text, x, size):
        font = ImageFont.truetype(FONT, size*scale)
        box = draw.textbbox((0, 0), text, font=font)
        draw.text((x*scale-(box[2]-box[0])/2-box[0],
                   HEADER*scale/2-(box[3]-box[1])/2-box[1]), text, font=font, fill='black')
    centered('Time step', MARGIN+LABEL/2-6, 44)
    for col, item in enumerate(items):
        centered(f"T = {item['executed_step']}", MARGIN+LABEL+col*(TILE+GAP)+TILE/2, 42)
    canvas.paste(header.resize((width, HEADER), Image.Resampling.LANCZOS), (0, 0))
    for row, label in enumerate(['Predicted\nRGB', 'Actual\nRGB', 'Predicted\nAction\nRaymap']):
        band = Image.new('RGB', (LABEL*scale, TILE*scale), 'white')
        pen = ImageDraw.Draw(band)
        font = ImageFont.truetype(FONT, 48*scale)
        box = pen.multiline_textbbox((0, 0), label, font=font, spacing=6*scale, align='center')
        pen.multiline_text(((LABEL*scale-(box[2]-box[0]))/2-box[0]-6*scale,
                            (TILE*scale-(box[3]-box[1]))/2-box[1]), label,
                           font=font, spacing=6*scale, align='center', fill='black')
        y = HEADER+MARGIN+row*(TILE+ROW_GAP)
        canvas.paste(band.resize((LABEL, TILE), Image.Resampling.LANCZOS), (MARGIN, y))
        for col, item in enumerate(items):
            source = Path(item['display_sources'][row] if 'display_sources' in item else item['sources'][row])
            with Image.open(ROOT / source) as raw:
                im = raw.convert('RGB')
                if row == 2:
                    assert im.size == (448, 224), im.size
                    im = im.crop((0, 0, 224, 224))  # First duplicated action tile; not a camera crop.
                assert im.size == (224, 224), im.size
                canvas.paste(im, (MARGIN+LABEL+col*(TILE+GAP), y))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, dpi=(300, 300))
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    parser.add_argument('--actual-root', type=Path, default=ROOT/'evaluate_results/realworld/paired_frames')
    parser.add_argument('--prediction-root', type=Path,
                        default=ROOT/'evaluate_results/realworld/pick_place_server_gpu7_rgb20/predicted_rgb')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--final-actual', type=Path,
                        help='Optional replacement Actual RGB for the last overview column')
    args = parser.parse_args()
    actual_root, pred_root = args.actual_root/args.session, args.prediction_root/args.session
    pairs, requests, skipped = [], [], []
    expected_before = 0
    for folder in sorted(actual_root.glob('request_*')):
        meta = json.loads((folder/'metadata.json').read_text())
        request = int(meta['request_id'])
        assert meta['session'] == args.session
        assert meta['prediction_artifact_key'] == f'{args.session}/{folder.name}'
        pred_meta = json.loads((pred_root/folder.name/'metadata.json').read_text())
        assert pred_meta['session'] == args.session and pred_meta['request_id'] == request
        local = []
        for step in range(1, meta['expected_steps']+1):
            stem = f'step_{step:02d}'
            paths = [pred_root/folder.name/f'{stem}_pred_rgb.png',
                     folder/f'{stem}_actual.png', pred_root/folder.name/f'{stem}_pred_rothko.png']
            record_path = folder/f'{stem}_actual.json'
            if not record_path.exists() or not all(p.is_file() for p in paths):
                skipped.append(dict(request_id=request, step=step))
                continue
            record = json.loads(record_path.read_text())
            assert record['step'] == step
            assert record['camera_timestamp'] > record['camera_baseline_timestamp']
            assert record['captured_monotonic_ns'] >= record['motion_done_monotonic_ns']
            item = dict(executed_step=expected_before+step, request_id=request, step=step,
                        captured_monotonic_ns=record['captured_monotonic_ns'],
                        position_error_mm=record['position_error_mm'],
                        sources=[str(p.resolve().relative_to(ROOT)) for p in paths])
            local.append(item)
        expected_before += meta['expected_steps']
        if local:
            pairs.extend(local)
            requests.append((request, local))
    if not pairs:
        raise ValueError('No complete matching frames')
    selected = [pairs[round(i*(len(pairs)-1)/11)] for i in range(12)] if len(pairs)>=12 else pairs
    final_actual = None
    if args.final_actual is not None:
        final_actual = args.final_actual.resolve()
        if not final_actual.is_file():
            raise FileNotFoundError(final_actual)
        prepared = args.output/'sources'/'final_actual_rgb.png'
        prepared.parent.mkdir(parents=True, exist_ok=True)
        square_rgb(final_actual).save(prepared)
        selected[-1] = dict(selected[-1])
        selected[-1]['display_sources'] = list(selected[-1]['sources'])
        selected[-1]['display_sources'][1] = str(prepared.resolve().relative_to(ROOT))
    overview = args.output/'realworld_pick_place.png'
    canvas = render(selected, overview)
    canvas.save(overview.with_suffix('.pdf'), resolution=300)
    for request, items in requests:
        render(items, args.output/'requests'/f'request_{request:06d}.png')
    manifest = dict(session=args.session, image=str(overview), size=list(canvas.size),
        selection='12 evenly spaced matched action steps, including first/last; no quality-based selection',
        time_unit='executed action index, not seconds; matching by session/request/step, not wall clock',
        rothko_display='left 224x224 tile from original duplicated 448x224 predicted representation; no recoloring',
        final_actual_override=(None if final_actual is None else dict(
            input=str(final_actual.relative_to(ROOT)),
            prepared=selected[-1]['display_sources'][1],
            column=len(selected), executed_step=selected[-1]['executed_step'],
            preprocessing='RGB PIL contain LANCZOS 224, centered black padding')),
        total_pairs=len(pairs), skipped=skipped, columns=selected, all_pairs=pairs)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(output=str(overview), pairs=len(pairs), request_figures=len(requests),
                         selected_steps=[x['executed_step'] for x in selected], skipped=len(skipped))))


if __name__ == '__main__':
    main()
