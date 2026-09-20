#!/usr/bin/env python3
"""Build an isolated LIBERO batch-one text cache, or verify it against online encoding.

Uses the evaluation encode_prompt implementation. Never overwrites an existing cache.
Run build with fastwam and verify with the fastwam_libero evaluation environment.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['build', 'verify'])
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    if args.mode == 'build':
        args.cache_dir.mkdir(parents=True, exist_ok=False)
    elif not (args.cache_dir / '_SUCCESS').exists():
        raise RuntimeError('Incomplete text cache')
    import torch
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from fastwam.models.wan22.helpers.loader import _load_registered_model
    from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
    from fastwam.models.wan22.fastwam_visual_action import FastWAMVideoOnlyRaymap
    encoder_path = ROOT / 'checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors'
    tokenizer_path = ROOT / 'checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl'
    identity = dict(encoder_sha256=sha(encoder_path), tokenizer_sha256={str(p.relative_to(tokenizer_path)): sha(p) for p in sorted(tokenizer_path.rglob('*')) if p.is_file()}, batch_size=1, dtype='torch.bfloat16', context_len=128, clean='whitespace', prompt_template=DEFAULT_PROMPT)
    if args.mode == 'verify':
        manifest = json.loads((args.cache_dir / 'manifest.json').read_text())
        if manifest['identity'] != identity:
            raise ValueError('Encoder/tokenizer/config identity mismatch')
    encoder = _load_registered_model(str(encoder_path), 'wan_video_text_encoder', torch_dtype=torch.bfloat16, device='cuda:0').eval().requires_grad_(False)
    tokenizer = HuggingfaceTokenizer(name=str(tokenizer_path), seq_len=128, clean='whitespace')
    model = SimpleNamespace(text_encoder=encoder, tokenizer=tokenizer, device='cuda:0')
    tasks = sorted({json.loads(line)['task'] for suite in ['spatial', 'object', 'goal', '10'] for line in (ROOT / f'data/libero_mujoco3.3.2/libero_{suite}_no_noops_lerobot/meta/tasks.jsonl').read_text().splitlines()})
    rows = []
    with torch.inference_mode():
        for task in tasks:
            prompt = DEFAULT_PROMPT.format(task=task)
            prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
            path = args.cache_dir / f'{prompt_sha}.t5_len128.wan22ti2v5b.pt'
            context, effective_mask = FastWAMVideoOnlyRaymap.encode_prompt(model, prompt)
            context, effective_mask = context[0].cpu(), effective_mask[0].cpu()
            _, token_mask = tokenizer(prompt, return_mask=True, add_special_tokens=True)
            if args.mode == 'build':
                metadata = dict(identity, format_version=2, prompt_sha256=prompt_sha, encoder_id='wan22ti2v5b')
                # Legacy suffix/encoder_id are required by the dataset reader;
                # exact encoder identity and batch size are recorded separately.
                torch.save(dict(context=context, mask=token_mask[0].bool(), cache_metadata=metadata), path)
            payload = torch.load(path, map_location='cpu', weights_only=False)
            if args.mode == 'verify' and sha(path) != manifest['files'][path.name]:
                raise ValueError(f'Cache checksum mismatch: {path}')
            cached = payload['context'].clone()
            cached[~payload['mask'].bool()] = 0
            equal = torch.equal(cached, context) and torch.equal(torch.ones_like(payload['mask'].bool()), effective_mask)
            rows.append(dict(task=task, file=path.name, bit_exact=equal, max_abs=(cached.float()-context.float()).abs().max().item()))
            print(task, equal, flush=True)
    report = dict(mode=args.mode, identity=identity, torch_version=torch.__version__, gpu=torch.cuda.get_device_name(), rows=rows, all_bit_exact=all(r['bit_exact'] for r in rows))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    if not report['all_bit_exact']:
        raise RuntimeError('Text cache differs from single-prompt online encoding; see report')
    if args.mode == 'build':
        (args.cache_dir / 'manifest.json').write_text(json.dumps(dict(identity=identity, files={r['file']:sha(args.cache_dir/r['file']) for r in rows}), indent=2)+'\n')
        (args.cache_dir / '_SUCCESS').write_text('complete\n')


if __name__ == '__main__':
    main()
