"""Download the pinned image dataset; preserve the separate video dataset."""
import hashlib
import json
import shutil
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout

REPO = 'VLABench/vlabench_primitive_ft_lerobot'
REVISION = '460a2a4dc0bf29bef92b3e195a0f4e87f2d026a6'
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'data/vlabench_primitive_ft_lerobot'


def main():
    info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
    reused = []
    for item in info.siblings:
        if item.rfilename not in [f'data/chunk-000/episode_{i:06d}.parquet' for i in range(3)]:
            continue
        source = ROOT / 'data/vlabench_image_video_audit' / Path(item.rfilename).name
        sha = getattr(item.lfs, 'sha256', None)
        if source.is_file() and sha and hashlib.sha256(source.read_bytes()).hexdigest() == sha:
            target = OUTPUT / item.rfilename
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)
            if hashlib.sha256(target.read_bytes()).hexdigest() == sha:
                reused.append(item.rfilename)
    print(f'Downloading {REPO}@{REVISION}; max_workers=4; reused={reused}', flush=True)
    snapshot_download(repo_id=REPO, repo_type='dataset', revision=REVISION,
                      local_dir=str(OUTPUT), max_workers=4, ignore_patterns=reused or None)
    bad = [x.rfilename for x in info.siblings
           if not (OUTPUT / x.rfilename).is_file()
           or (x.size is not None and (OUTPUT / x.rfilename).stat().st_size != x.size)]
    if bad:
        raise RuntimeError(f'Missing/incorrectly sized files: {bad[:20]}')
    result = {'repo': REPO, 'revision': REVISION, 'files': len(info.siblings),
              'bytes': sum(x.size or 0 for x in info.siblings),
              'verification': 'All repository files present with matching sizes'}
    (OUTPUT / 'download_verification.json').write_text(json.dumps(result, indent=2))
    print(result, flush=True)


if __name__ == '__main__':
    for attempt in range(1, 6):
        try:
            main()
            break
        except (ChunkedEncodingError, ConnectionError, Timeout, LocalEntryNotFoundError) as exc:
            if attempt == 5:
                raise
            delay = min(30 * attempt, 60)
            print(f'Transient download failure ({attempt}/5): {exc}; resume in {delay}s', flush=True)
            time.sleep(delay)
