"""Document-isolated FineWeb windows with revision-keyed row-group caches."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from nnmodelling_runtime import Batch, DatasetAdapter
from tokenizers import Tokenizer


class Dataset(DatasetAdapter[str, str]):
    def __init__(self, dataset_dir: Path):
        super().__init__(dataset_dir)
        self.config = json.loads((self.directory / "config.json").read_text(encoding="utf-8"))
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        self.tokenizer_path = self.directory / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self.eos = self.tokenizer.token_to_id(self.config["eos_token"])
        if self.eos is None:
            raise ValueError("Tokenizer is missing the declared EOS token.")
        self.context = self.config["context"]
        if type(self.context) is not int or self.context < 1:
            raise ValueError("Dataset context must be a positive integer.")
        self._tokenizer_sha256 = self._sha256(self.tokenizer_path)
        remote = self.manifest.get("remoteFiles")
        if not isinstance(remote, dict) or not isinstance(remote.get("files"), list):
            raise ValueError("Dataset manifest has no pinned remote corpus files.")
        self._remote = remote
        self._evaluation: dict[str, list[np.ndarray]] = {}

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _split(identifier: str) -> str:
        bucket = int.from_bytes(hashlib.sha256(identifier.encode("utf-8")).digest()[:8], "big") % 1000
        if bucket == 0:
            return "validation"
        if bucket == 1:
            return "test"
        return "train"

    def tokenize(self, value: str) -> torch.Tensor:
        if not isinstance(value, str):
            raise TypeError("Input must be text.")
        ids = self.tokenizer.encode(value, add_special_tokens=False).ids
        if not ids:
            ids = [self.eos]
        return torch.tensor([ids[-self.context:]], dtype=torch.int64)

    def untokenize(self, tensor: torch.Tensor) -> str:
        ids = tensor.argmax(-1) if tensor.is_floating_point() else tensor
        return self.tokenizer.decode(ids.detach().cpu().reshape(-1).tolist())

    def _cache_identity(self, source_record: dict, split: str, row_group: int) -> dict:
        return {
            "version": 1,
            "repository": self._remote["repo"],
            "revision": self._remote["revision"],
            "source_path": source_record["path"],
            "destination": source_record["destination"],
            "source_sha256": source_record["sha256"],
            "split": split,
            "row_group": row_group,
            "config": self.config,
            "tokenizer_sha256": self._tokenizer_sha256,
        }

    def _cache_location(self, identity: dict) -> tuple[Path, str] | None:
        root = self.cache_directory
        if root is None:
            return None
        if root.is_symlink():
            raise ValueError("Dataset cache root must not be a symlink.")
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("Dataset cache root is not a safe directory.")
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return root / key, key

    def _load_cached_row_group(
        self, cache_dir: Path, identity: dict, width: int,
    ) -> np.ndarray:
        if cache_dir.is_symlink() or not cache_dir.is_dir():
            raise ValueError("Dataset token cache is not a safe directory.")
        entries = list(cache_dir.iterdir())
        if {entry.name for entry in entries} != {"manifest.json", "windows.bin"}:
            raise ValueError("Dataset token cache contains unexpected files.")
        if any(entry.is_symlink() for entry in entries):
            raise ValueError("Dataset token cache contains a symlink.")
        manifest_path = cache_dir / "manifest.json"
        data_path = cache_dir / "windows.bin"
        if manifest_path.is_symlink() or data_path.is_symlink():
            raise ValueError("Dataset token cache contains a symlink.")
        if not manifest_path.is_file() or not data_path.is_file():
            raise ValueError("Dataset token cache is incomplete.")
        if manifest_path.stat().st_size > 64 * 1024:
            raise ValueError("Dataset token cache metadata is too large.")
        try:
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Dataset token cache metadata is invalid.") from error
        shape = metadata.get("shape")
        if (metadata.get("version") != 1 or metadata.get("identity") != identity
                or metadata.get("dtype") != "<i8" or not isinstance(shape, list)
                or len(shape) != 2 or shape[1] != width
                or type(shape[0]) is not int or shape[0] < 0):
            raise ValueError("Dataset token cache identity, shape or dtype is invalid.")
        expected_size = shape[0] * width * np.dtype("<i8").itemsize
        if metadata.get("size") != expected_size or data_path.stat().st_size != expected_size:
            raise ValueError("Dataset token cache size does not match its shape.")
        if self._sha256(data_path) != metadata.get("sha256"):
            raise ValueError("Dataset token cache digest is invalid.")
        if shape[0] == 0:
            return np.empty((0, width), dtype=np.int64)
        return np.memmap(data_path, dtype="<i8", mode="r", shape=(shape[0], width))

    def _iter_row_group_windows(self, table, split: str) -> Iterator[list[int]]:
        for batch in table.to_batches(max_chunksize=256):
            rows = batch.select(["id", "text"]).to_pydict()
            for identifier, text in zip(rows["id"], rows["text"], strict=True):
                if not isinstance(identifier, str) or not isinstance(text, str):
                    raise ValueError("FineWeb rows must contain string id and text values.")
                if self._split(identifier) != split:
                    continue
                tokens = self.tokenizer.encode(text, add_special_tokens=False).ids
                tokens.append(self.eos)
                width = self.context + 1
                for start in range(0, len(tokens) - width + 1, self.context):
                    yield tokens[start:start + width]

    def _write_row_group_cache(
        self, table, split: str, identity: dict, cache_dir: Path, key: str,
    ) -> np.ndarray:
        cache_root = cache_dir.parent
        temporary = Path(tempfile.mkdtemp(prefix=f".{key}.tmp-", dir=cache_root))
        try:
            data_path = temporary / "windows.bin"
            digest = hashlib.sha256()
            row_count = 0
            with data_path.open("xb") as stream:
                for window in self._iter_row_group_windows(table, split):
                    encoded = np.asarray(window, dtype="<i8").tobytes()
                    stream.write(encoded)
                    digest.update(encoded)
                    row_count += 1
            metadata = {
                "version": 1,
                "identity": identity,
                "dtype": "<i8",
                "shape": [row_count, self.context + 1],
                "size": row_count * (self.context + 1) * np.dtype("<i8").itemsize,
                "sha256": digest.hexdigest(),
            }
            (temporary / "manifest.json").write_text(
                json.dumps(metadata, sort_keys=True, separators=(",", ":")), encoding="utf-8",
            )
            try:
                os.replace(temporary, cache_dir)
            except OSError:
                if not cache_dir.exists():
                    raise
                # A concurrent writer may have committed the same key first.
                shutil.rmtree(temporary, ignore_errors=True)
            return self._load_cached_row_group(cache_dir, identity, self.context + 1)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def _windows(self, source: Path, source_record: dict, split: str) -> Iterator[np.ndarray]:
        import pyarrow.parquet as pq

        width = self.context + 1
        parquet = pq.ParquetFile(source)
        try:
            for row_group in range(parquet.num_row_groups):
                identity = self._cache_identity(source_record, split, row_group)
                location = self._cache_location(identity)
                if location is None:
                    table = parquet.read_row_group(row_group, columns=["id", "text"])
                    yield from self._iter_row_group_windows(table, split)
                    continue
                cache_dir, key = location
                if cache_dir.is_symlink():
                    raise ValueError("Dataset token cache path must not be a symlink.")
                if cache_dir.exists():
                    windows = self._load_cached_row_group(cache_dir, identity, width)
                else:
                    table = parquet.read_row_group(row_group, columns=["id", "text"])
                    windows = self._write_row_group_cache(table, split, identity, cache_dir, key)
                yield from windows
        finally:
            parquet.close()

    def _samples(self, split: str) -> Iterator[np.ndarray]:
        if split == "train":
            limit = self.config["train_token_budget"] // self.context
        else:
            limit = self.config[f"{split}_windows"]
        if type(limit) is not int or limit < 1:
            raise ValueError(f"Dataset {split} budget must be a positive integer.")

        count = 0
        data_root = self.data_directory.resolve()
        for source_record in self._remote["files"]:
            destination_text = source_record["destination"]
            if not isinstance(destination_text, str):
                raise ValueError("Acquired corpus destination is unsafe.")
            destination = Path(destination_text)
            if "\\" in destination_text or destination.is_absolute() or ".." in destination.parts:
                raise ValueError("Acquired corpus destination is unsafe.")
            source = data_root / destination
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"Acquired corpus file is missing or unsafe: {destination}.")
            try:
                source.resolve(strict=True).relative_to(data_root)
            except ValueError as error:
                raise ValueError(f"Acquired corpus path escapes its data root: {destination}.") from error
            current = data_root
            for part in destination.parts:
                current = current / part
                if current.is_symlink():
                    raise ValueError(f"Acquired corpus path contains a symlink: {destination}.")
            for window in self._windows(source, source_record, split):
                if count >= limit:
                    return
                count += 1
                yield window
                if count >= limit:
                    return
        if count < limit:
            raise ValueError(f"Corpus exhausted after {count} {split} windows; expected {limit}.")

    def load(self, split: str, batch_size: int) -> Iterator[Batch]:
        if split not in {"train", "validation", "test"} or batch_size < 1:
            raise ValueError("Invalid dataset split or batch size.")
        if split != "train":
            if split not in self._evaluation:
                self._evaluation[split] = list(self._samples(split))
            samples = iter(self._evaluation[split])
        else:
            samples = self._samples(split)

        group = []
        for window in samples:
            group.append(window)
            if len(group) == batch_size:
                data = torch.tensor(np.asarray(group), dtype=torch.int64)
                yield Batch({"tokens": data[:, :-1]}, {"target": data[:, 1:]})
                group = []
        if group:
            data = torch.tensor(np.asarray(group), dtype=torch.int64)
            yield Batch({"tokens": data[:, :-1]}, {"target": data[:, 1:]})
