"""Download original HDF5 archive volumes without extracting or deleting data."""
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = 'VLABench/raw_primitive_datasets'
REVISION = '765ca79fcb078f3b272d49580069ce17ef107134'
OUTPUT = Path(__file__).resolve().parents[1] / 'data/vlabench_raw_primitive'


def main():
    print(f'Downloading {REPO}@{REVISION} -> {OUTPUT}; max_workers=2', flush=True)
    snapshot_download(repo_id=REPO, repo_type='dataset', revision=REVISION,
                      local_dir=str(OUTPUT), max_workers=2)
    info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
    bad = [x.rfilename for x in info.siblings
           if not (OUTPUT / x.rfilename).is_file()
           or (x.size is not None and (OUTPUT / x.rfilename).stat().st_size != x.size)]
    if bad:
        raise RuntimeError(f'Missing/incorrectly sized files: {bad}')
    result = {'repo': REPO, 'revision': REVISION, 'files': len(info.siblings),
              'bytes': sum(x.size or 0 for x in info.siblings),
              'verification': 'All repository files present with matching sizes; archives not yet extracted'}
    (OUTPUT / 'download_verification.json').write_text(json.dumps(result, indent=2))
    print(result, flush=True)


if __name__ == '__main__':
    main()
