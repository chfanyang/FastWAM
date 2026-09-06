"""Sharded, bit-exact BF16 latent cache for visual-action datasets."""

from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


LATENT_CACHE_VERSION = 1
LATENT_CACHE_METADATA = "metadata.json"
LATENT_CACHE_SUCCESS = "_SUCCESS"
LATENT_CACHE_DTYPE = "bfloat16_raw_uint16"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset_contract(
    *,
    dataset_dirs: list[str],
    dataset_length: int,
    num_frames: int,
    video_size: list[int],
    raymap_representation: str | None,
    raymap_codec_metadata: Mapping[str, Any] | None,
    norm_stats_sha256: str | None,
) -> dict[str, Any]:
    roots = []
    for raw_root in dataset_dirs:
        root = Path(raw_root).expanduser().resolve()
        metadata_files = {}
        for name in ("info.json", "episodes.jsonl", "tasks.jsonl"):
            path = root / "meta" / name
            if not path.is_file():
                raise FileNotFoundError(
                    f"Cannot identify latent-cache source dataset; missing {path}"
                )
            metadata_files[name] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        roots.append(
            {
                "path": str(root),
                "metadata_files": metadata_files,
            }
        )
    return {
        "dataset_roots": roots,
        "dataset_length": int(dataset_length),
        "num_frames": int(num_frames),
        "video_size": [int(value) for value in video_size],
        "raymap_representation": raymap_representation,
        "raymap_codec_metadata": (
            None
            if raymap_codec_metadata is None
            else dict(raymap_codec_metadata)
        ),
        "norm_stats_sha256": norm_stats_sha256,
    }


class LatentCacheReader:
    """Random-access reader for sharded RGB/Rothko latent pairs."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        expected_dataset_contract: Mapping[str, Any],
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        metadata_path = self.cache_dir / LATENT_CACHE_METADATA
        success_path = self.cache_dir / LATENT_CACHE_SUCCESS
        if not metadata_path.is_file() or not success_path.is_file():
            raise FileNotFoundError(
                "Latent cache is incomplete; expected metadata.json and _SUCCESS "
                f"under {self.cache_dir}"
            )
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if int(metadata.get("cache_version", -1)) != LATENT_CACHE_VERSION:
            raise ValueError(
                "Unsupported latent cache version: "
                f"{metadata.get('cache_version')!r}"
            )
        if metadata.get("dtype") != LATENT_CACHE_DTYPE:
            raise ValueError(
                f"Unsupported latent cache dtype: {metadata.get('dtype')!r}"
            )
        if metadata.get("dataset_contract") != dict(expected_dataset_contract):
            raise ValueError(
                "Latent cache dataset contract mismatch. The dataset roots/order, "
                "metadata, Rothko config, norm stats, frame count and canvas size "
                "must exactly match the cache."
            )
        if not bool(metadata.get("complete", False)):
            raise ValueError(f"Latent cache metadata is not marked complete: {metadata_path}")

        latent_shape = tuple(int(value) for value in metadata["latent_shape"])
        if len(latent_shape) != 4:
            raise ValueError(
                f"Expected latent shape [C,T,H,W], got {latent_shape}."
            )
        self.latent_shape = latent_shape
        self.metadata = metadata
        self._shards = sorted(metadata["shards"], key=lambda item: int(item["start"]))
        self._starts = [int(item["start"]) for item in self._shards]
        self._maps: dict[int, np.memmap] = {}

        expected_start = 0
        item_values = 2 * int(np.prod(self.latent_shape))
        for shard in self._shards:
            start = int(shard["start"])
            end = int(shard["end"])
            if start != expected_start or end <= start:
                raise ValueError(f"Invalid/non-contiguous latent shard metadata: {shard}")
            path = self.cache_dir / shard["file"]
            expected_bytes = (end - start) * item_values * np.dtype(np.uint16).itemsize
            if not path.is_file() or path.stat().st_size != expected_bytes:
                raise ValueError(
                    f"Latent shard size mismatch: path={path}, "
                    f"expected={expected_bytes}, "
                    f"actual={path.stat().st_size if path.exists() else None}"
                )
            expected_start = end
        self.length = int(metadata["num_samples"])
        if expected_start != self.length:
            raise ValueError(
                f"Latent shards cover {expected_start} samples, expected {self.length}."
            )

    def __len__(self) -> int:
        return self.length

    def _open_shard(self, shard_index: int) -> np.memmap:
        cached = self._maps.get(shard_index)
        if cached is not None:
            return cached
        shard = self._shards[shard_index]
        count = int(shard["end"]) - int(shard["start"])
        array = np.memmap(
            self.cache_dir / shard["file"],
            mode="r",
            dtype=np.uint16,
            shape=(count, 2, *self.latent_shape),
        )
        self._maps[shard_index] = array
        return array

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(index)
        if index < 0:
            index += self.length
        if index < 0 or index >= self.length:
            raise IndexError(index)
        shard_index = bisect.bisect_right(self._starts, index) - 1
        shard = self._shards[shard_index]
        local_index = index - int(shard["start"])
        # Copy out of the read-only memmap. DataLoader collation then operates on
        # ordinary writable CPU storage and cannot mutate the cache file.
        raw = np.array(self._open_shard(shard_index)[local_index], copy=True)
        pair = torch.from_numpy(raw).view(torch.bfloat16)
        return pair[0], pair[1]

