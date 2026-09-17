import json
from pathlib import Path
from compare_libero_rothko_decoders import summarize

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'evaluate_results/libero_decoder_offline'
OUT=BASE/'wan21_checkpoint_curve_joint_anchor0_heldout800_20260913'
OLD=BASE/'wan21_wan22_step7498_joint_anchor0_heldout800_20260913'
rows=[]
hashes=set()
for i in range(4):
    hashes.add(json.loads((OUT/f'shard{i}/summary.json').read_text())['manifest_sha256'])
    hashes.add(json.loads((OLD/f'shard{i}/summary.json').read_text())['manifest_sha256'])
    prior={r['window_index']:r for r in json.loads((OLD/f'shard{i}/per_window_errors.json').read_text())}
    for r in json.loads((OUT/f'shard{i}/per_window_errors.json').read_text()):
        r['errors']['wan21_step007498']=prior[r['window_index']]['errors']['wan21']
        rows.append(r)
assert len(hashes)==1 and len(rows)==800
assert {r['window_index'] for r in rows}==set(range(800))
assert all(len(r['errors'])==6 for r in rows)
report={'windows':800,'future_poses':12800,'mode':'robust_joint','anchor_alpha':0,'manifest_sha256':next(iter(hashes)),
        'overall':summarize(rows),'by_suite':{s:summarize([r for r in rows if r['suite']==s]) for s in sorted({r['suite'] for r in rows})}}
(OUT/'summary.json').write_text(json.dumps(report,indent=2))
for name,metrics in report['overall'].items():
    print(name, {k:{m:v for m,v in values.items() if m in ['mean','p95','max']} for k,values in metrics.items()})
