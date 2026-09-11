"""Download and extract official clean50 only, retaining ZIPs and provenance.

One download at a time with resumable .part files, bounded retries, pinned
Hugging Face revision, SHA256 verification and ZIP CRC validation. Existing
archives/extracted files are verified before reuse, not silently overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import time
import urllib.request
import zipfile
import zlib

REPO = "TianxingChen/RoboTwin2.0"
ARCHIVE = "aloha-agilex_clean_50.zip"


def log(message):
    print(time.strftime("[%Y-%m-%d %H:%M:%S UTC]", time.gmtime()), message, flush=True)


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def get_json(url):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response), response.headers.get("Link", "")
        except Exception:
            if attempt == 4:
                raise
            time.sleep(min(30, 5 * 2**attempt))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_verified(archive, destination):
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        for info in members:
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"Unsafe ZIP path: {info.filename}")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"ZIP symlink not supported: {info.filename}")
        hdf5 = [x for x in members if x.filename.endswith(".hdf5")]
        if len(hdf5) != 50:
            raise ValueError(f"Expected 50 HDF5 episodes, got {len(hdf5)}")
        destination.mkdir(parents=True, exist_ok=True)
        for info in members:
            target = destination / info.filename
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                crc = 0
                with target.open("rb") as f:
                    for block in iter(lambda: f.read(8 * 1024**2), b""):
                        crc = zlib.crc32(block, crc)
                if target.stat().st_size != info.file_size or crc != info.CRC:
                    raise ValueError(f"Existing extracted file differs; preserved: {target}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".extracting")
            # ZipExtFile verifies CRC while reading through EOF.
            with z.open(info) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024**2)
            temporary.replace(target)
        return sum(x.file_size for x in members)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/robotwin2.0_official_aloha_clean50"))
    p.add_argument("--tasks", nargs="+", help="Only process these tasks, preserving other task statuses")
    args = p.parse_args()
    root = args.root.resolve()
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests / "clean50_download_manifest.json"
    names = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                "src/fastwam/datasets/lerobot/robotwin_tasks.py"))["ROBOTWIN_TASK_NAMES"]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        log("Fetching pinned Hugging Face clean50 manifest")
        metadata, _ = get_json(f"https://huggingface.co/api/datasets/{REPO}")
        revision = metadata["sha"]
        url = f"https://huggingface.co/api/datasets/{REPO}/tree/{revision}/dataset?recursive=true&limit=1000"
        entries = []
        while url:
            page, links = get_json(url)
            entries.extend(page)
            url = None
            for link in links.split(","):
                if 'rel="next"' in link:
                    url = link.split("<", 1)[1].split(">", 1)[0]
        by_path = {x["path"]: x for x in entries if x["type"] == "file"}
        files = []
        for name in names:
            entry = by_path[f"dataset/{name}/{ARCHIVE}"]
            files.append({"task": name, "path": entry["path"], "size": entry["size"],
                          "sha256": entry["lfs"]["oid"]})
        manifest = {"repo": REPO, "revision": revision, "files": files}
        write_json(manifest_path, manifest)
    assert [x["task"] for x in manifest["files"]] == list(names)
    if args.tasks and set(args.tasks) - set(names):
        p.error(f"Unknown tasks: {sorted(set(args.tasks) - set(names))}")
    total = sum(x["size"] for x in manifest["files"])
    log(f"50 tasks, ZIP total {total / 1024**3:.2f} GiB; one transfer at a time, limit 20 MiB/s")
    status_path = manifests / "clean50_download_status.json"
    status = {"revision": manifest["revision"], "tasks": {}}
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status["revision"] != manifest["revision"]:
            raise ValueError("Status revision differs from pinned manifest")
    failures = []
    for index, entry in enumerate(manifest["files"], 1):
        task = entry["task"]
        if args.tasks and task not in args.tasks:
            continue
        state = {"status": "downloading", "zip_bytes": entry["size"]}
        status["tasks"][task] = state
        write_json(status_path, status)
        try:
            archive = root / "raw_zips" / task / ARCHIVE
            archive.parent.mkdir(parents=True, exist_ok=True)
            log(f"[{index}/50] {task}: {entry['size'] / 1024**2:.1f} MiB")
            if not archive.exists():
                part = archive.with_suffix(".zip.part")
                if not part.exists() or part.stat().st_size != entry["size"]:
                    url = f"https://huggingface.co/datasets/{REPO}/resolve/{manifest['revision']}/{entry['path']}"
                    subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                                    "--continue-at", "-", "--retry", "5", "--retry-delay", "10",
                                    "--retry-max-time", "600", "--connect-timeout", "30",
                                    "--max-time", "7200", "--speed-limit", "1024", "--speed-time", "120",
                                    "--limit-rate", "20M", "--output", str(part), url], check=True)
                if part.stat().st_size != entry["size"] or sha256(part) != entry["sha256"]:
                    raise ValueError(f"Downloaded ZIP checksum/size mismatch: {part}")
                part.replace(archive)
            if archive.stat().st_size != entry["size"] or sha256(archive) != entry["sha256"]:
                raise ValueError(f"Existing ZIP checksum/size mismatch, preserved: {archive}")
            state["status"] = "extracting"
            write_json(status_path, status)
            state["extracted_bytes"] = extract_verified(archive, root / "extracted" / task)
            state["status"] = "complete"
            state["episodes"] = 50
            log(f"[{index}/50] {task}: verified and extracted; ZIP retained")
        except Exception as error:
            # Do not print signed redirect URLs or proxy credentials.
            state["status"] = "failed"
            state["error_type"] = type(error).__name__
            if not isinstance(error, subprocess.CalledProcessError):
                state["error"] = str(error)
            failures.append(task)
            log(f"[{index}/50] {task}: failed ({type(error).__name__}); continuing other tasks")
        write_json(status_path, status)
    completed = sum(v["status"] == "complete" for v in status["tasks"].values())
    log(f"Finished: {completed}/50 complete; failures_this_run={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
