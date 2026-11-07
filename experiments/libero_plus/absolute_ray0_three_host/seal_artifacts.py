"""Hash a transferred, isolated evaluation snapshot. Does not start evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REQUIRED = ['artifacts/dit_step021700.pt', 'artifacts/vae_step008264.safetensors',
            'artifacts/dataset_stats.json', 'artifacts/dit_training_config.yaml',
            'artifacts/vae_training_config.json',
            'data/libero_mujoco3.3.2/libero_rothko_region_symmetric_q99p95_h16_224x448.pt']

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    marker = ROOT / 'artifacts/READY.json'
    if args.verify:
        evidence = json.loads(marker.read_text())
        assert set(REQUIRED) <= set(evidence['sha256'])
        for rel, expected in evidence['sha256'].items():
            assert digest(ROOT / rel) == expected, rel
        print('SHA256 verified; no evaluation started')
    else:
        assert not marker.exists(), 'Existing manifest must not be overwritten'
        for rel in REQUIRED:
            assert (ROOT / rel).is_file() and (ROOT / rel).stat().st_size > 0, rel
        files = {ROOT / rel for rel in REQUIRED}
        for directory in ['src', 'configs', 'experiments/libero', 'experiments/libero_plus']:
            files.update(p for p in (ROOT / directory).rglob('*')
                         if p.is_file() and p.suffix in {'.py', '.sh', '.yaml', '.json'})
        evidence = {'dit_step': 21700, 'vae_step': 8264,
                    'sha256': {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}}
        # Exclusive creation: never silently replace a prior identity.
        with marker.open('x') as f:
            json.dump(evidence, f, indent=2)
            f.write('\n')
        print('Sealed snapshot; checkpoint provenance must be checked against the recorded runs before launch')
