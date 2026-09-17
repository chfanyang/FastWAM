import json
from pathlib import Path
from compare_libero_rothko_decoders import summarize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'evaluate_results/libero_decoder_offline/wan21_wan22_step7498_joint_anchor0_heldout800_20260913'
rows=[]
hashes=set()
for i in range(4):
    p=OUT/f'shard{i}'
    hashes.add(json.loads((p/'summary.json').read_text())['manifest_sha256'])
    rows.extend(json.loads((p/'per_window_errors.json').read_text()))
assert len(hashes)==1 and len(rows)==800
assert {r['window_index'] for r in rows}==set(range(800))
assert all(set(r['errors'])=={'wan21','wan22'} for r in rows)
report={'windows':800,'future_poses_per_model':12800,'decode_mode':'robust_joint','anchor_alpha':0,'manifest_sha256':next(iter(hashes)),
        'overall':summarize(rows), 'by_suite':{s:summarize([r for r in rows if r['suite']==s]) for s in sorted({r['suite'] for r in rows})}}
(OUT/'summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report['overall'],indent=2))
