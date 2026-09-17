"""Merge disjoint full-chain precision shards with preserved single-GPU results."""
import json
from pathlib import Path
import torch

root = Path('evaluate_results/libero_decoder_offline/wan21_fullchain_precision_a800_20260913')
parts = [root/'full800'] + [root/'eight_gpu'/f'shard{i}' for i in range(8)]
rows = []
digest = None
for part in parts:
    m = json.loads((part/'manifest.json').read_text())
    digest = digest or m['manifest_sha256']
    assert digest == m['manifest_sha256']
    if part != parts[0]:
        assert (part/'summary.json').exists(), f'Incomplete shard {part}'
    rows.extend(json.loads((part/'per_window_errors.json').read_text()))
assert len(rows) == 800 and {r['index'] for r in rows} == set(range(800))
rows.sort(key=lambda r:r['index'])

def summarize(group):
    out = {}
    for mode in group[0]['errors']:
        out[mode] = {}
        for metric in ('translation_mm', 'rotation_deg'):
            x = torch.tensor([r['errors'][mode][metric] for r in group], dtype=torch.float64)
            out[mode][metric] = {'mean':x.mean().item(), 'p95':x.quantile(.95).item(),
                                 'max':x.max().item(), 'first8_mean':x[:,:8].mean().item()}
    return out

report = {'windows':800, 'manifest_sha256':digest, 'overall':summarize(rows),
          'by_suite':{s:summarize([r for r in rows if r['suite']==s]) for s in sorted({r['suite'] for r in rows})},
          'source_directories':[str(p) for p in parts]}
(root/'eight_gpu'/'per_window_errors.json').write_text(json.dumps(rows))
(root/'eight_gpu'/'summary.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
