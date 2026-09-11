"""Remove only the two derived FK columns from RoboTwin and its metadata.

Requires --apply. Each parquet is read back and its retained values checked
before atomic replacement. This intentionally makes old FK-based dataset
configs unavailable; it does not touch videos, language or old run artifacts.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import stat
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

DROP = {"action.endpose", "observation.state.endpose"}


def clean_metadata(metadata):
    metadata = dict(metadata or {})
    if b"pandas" in metadata:
        p = json.loads(metadata[b"pandas"])
        p["columns"] = [x for x in p["columns"] if x["field_name"] not in DROP]
        metadata[b"pandas"] = json.dumps(p).encode()
    if b"huggingface" in metadata:
        h = json.loads(metadata[b"huggingface"])
        for key in DROP:
            h.get("info", {}).get("features", {}).pop(key, None)
        metadata[b"huggingface"] = json.dumps(h).encode()
    return metadata


def remove_columns(path):
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    table = pq.read_table(path, use_threads=False)
    removed = DROP.intersection(table.column_names)
    if not removed:
        return {"file": str(path), "rows": table.num_rows, "removed": []}
    retained = table.drop(sorted(removed))
    retained = retained.replace_schema_metadata(clean_metadata(table.schema.metadata))
    fd, name = tempfile.mkstemp(prefix=".remove_fk_", suffix=".parquet", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        pq.write_table(retained, temporary, compression="snappy")
        checked = pq.read_table(temporary, use_threads=False)
        if not retained.equals(checked, check_metadata=False):
            raise ValueError(f"Retained values changed during write: {path}")
        if checked.schema.metadata != retained.schema.metadata:
            raise ValueError(f"Retained metadata changed during write: {path}")
        temporary.chmod(stat.S_IMODE(path.stat().st_mode))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"file": str(path), "rows": table.num_rows, "removed": sorted(removed)}


def atomic_text(path, text):
    fd, name = tempfile.mkstemp(prefix=".remove_fk_meta_", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        if path.exists():
            temporary.chmod(stat.S_IMODE(path.stat().st_mode))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--apply", action="store_true", required=True)
    args = p.parse_args()
    root = args.root.resolve()
    paths = sorted((root / "data").glob("chunk-*/*.parquet"))
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    if len(paths) != 27500 or info["total_episodes"] != 27500:
        raise ValueError("Expected the 27,500-episode released RoboTwin dataset")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(remove_columns, paths, chunksize=16):
            results.append(result)
            if len(results) % 500 == 0:
                print(f"Verified and processed {len(results)}/{len(paths)}", flush=True)
    # Only publish metadata changes after every parquet succeeds.
    for key in DROP:
        info["features"].pop(key, None)
    atomic_text(info_path, json.dumps(info, indent=4) + "\n")
    stats_path = root / "meta/episodes_stats.jsonl"
    lines = []
    with stats_path.open() as f:
        for line in f:
            record = json.loads(line)
            for key in DROP:
                record["stats"].pop(key, None)
            lines.append(json.dumps(record))
    atomic_text(stats_path, "\n".join(lines) + "\n")
    global_stats = root / "meta/stats.json"
    if global_stats.exists():
        values = json.loads(global_stats.read_text())
        for key in DROP:
            values.pop(key, None)
        atomic_text(global_stats, json.dumps(values, indent=2) + "\n")
    summary = {"root": str(root), "episodes": len(results),
               "total_rows": sum(x["rows"] for x in results),
               "changed_parquets": sum(bool(x["removed"]) for x in results),
               "removed_columns": sorted(DROP), "retained_values_verified": True,
               "metadata_stats_records": len(lines)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(args.report, json.dumps({"summary": summary, "files": results}, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
