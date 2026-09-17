"""Download a pinned, complete VLABench Primitive FT video dataset snapshot."""
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "VLABench/vlabench_primitive_ft_lerobot_video"
REVISION = "9846a2f6bead3873251dc4fe3079359d57326b7c"
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/vlabench_primitive_ft_lerobot_video"


def main():
    print(f"Downloading {REPO_ID}@{REVISION} to {OUTPUT}", flush=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        local_dir=str(OUTPUT),
        max_workers=4,
    )
    info = HfApi().dataset_info(REPO_ID, revision=REVISION, files_metadata=True)
    invalid = []
    for entry in info.siblings:
        path = OUTPUT / entry.rfilename
        if not path.is_file() or (entry.size is not None and path.stat().st_size != entry.size):
            invalid.append(entry.rfilename)
    if invalid:
        raise RuntimeError(f"Missing or incorrectly sized files: {invalid[:20]}")
    result = dict(repo_id=REPO_ID, revision=REVISION, file_count=len(info.siblings),
                  total_bytes=sum(entry.size or 0 for entry in info.siblings),
                  verification="all repository files present with matching sizes")
    (OUTPUT / "download_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Download verified: {result}", flush=True)


if __name__ == "__main__":
    main()
