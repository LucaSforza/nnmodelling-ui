from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers


DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples/models/miniDeepSeekv4/datasets/dsv4.fineweb-edu-1.0.0/dataset.py"
)
SPEC = importlib.util.spec_from_file_location("dsv4_fineweb_dataset", DATASET_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
Dataset = MODULE.Dataset


def _ids_for_each_split(count: int) -> dict[str, list[str]]:
    ids = {"train": [], "validation": [], "test": []}
    candidate = 0
    while min(map(len, ids.values())) < count:
        identifier = f"doc-{candidate}"
        split = Dataset._split(identifier)
        if len(ids[split]) < count:
            ids[split].append(identifier)
        candidate += 1
    return ids


def _fixture(
    tmp_path: Path,
    *,
    train_windows: int = 4,
    evaluation_windows: int = 2,
    row_group_size: int = 32,
    train_documents: int | None = None,
) -> tuple[Path, dict[str, list[str]], dict[str, int]]:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True)
    ids = _ids_for_each_split(max(train_windows, evaluation_windows, train_documents or 0))
    all_ids = ids["train"] + ids["validation"] + ids["test"]
    words = [f"word{index}" for index in range(len(all_ids))]
    vocab = {"[UNK]": 0, **{word: index + 1 for index, word in enumerate(words)}, "<eos>": len(words) + 1}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(dataset_dir / "tokenizer.json"))
    config = {
        "context": 2,
        "train_token_budget": train_windows * 2,
        "validation_windows": evaluation_windows,
        "test_windows": evaluation_windows,
        "tokenizer_repo": "test/tokenizer",
        "tokenizer_revision": "a" * 40,
        "eos_token": "<eos>",
    }
    (dataset_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    texts = {identifier: f"{word} {word}" for identifier, word in zip(all_ids, words, strict=True)}
    table = pa.table({"id": all_ids, "text": [texts[identifier] for identifier in all_ids]})
    source = dataset_dir / "corpus.parquet"
    pq.write_table(table, source, row_group_size=row_group_size)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {
        "schemaVersion": 1,
        "id": "dsv4.fineweb-edu",
        "version": "1.0.0",
        "remoteFiles": {
            "repo": "HuggingFaceFW/fineweb-edu",
            "revision": "b" * 40,
            "files": [{
                "path": "sample/10BT/corpus.parquet",
                "destination": "corpus.parquet",
                "size": source.stat().st_size,
                "sha256": source_hash,
            }],
        },
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return dataset_dir, ids, {identifier: vocab[word] for identifier, word in zip(all_ids, words, strict=True)}


def _make_dataset(tmp_path: Path, monkeypatch, **kwargs):
    dataset_dir, ids, token_ids = _fixture(tmp_path, **kwargs)
    monkeypatch.setenv("NNMODELLING_DATASET_CACHE_ROOT", str(tmp_path / "cache"))
    return Dataset(dataset_dir), ids, token_ids, dataset_dir


def test_split_windows_are_deterministic_disjoint_and_document_local(tmp_path, monkeypatch):
    dataset, ids, token_ids, _dataset_dir = _make_dataset(tmp_path, monkeypatch)

    windows = {
        split: list(dataset._samples(split))
        for split in ("train", "validation", "test")
    }
    assert set(ids["train"]).isdisjoint(ids["validation"])
    assert set(ids["train"]).isdisjoint(ids["test"])
    assert set(ids["validation"]).isdisjoint(ids["test"])
    assert len(windows["train"]) == 4
    assert len(windows["validation"]) == len(windows["test"]) == 2

    eos = dataset.eos
    for split, rows in windows.items():
        expected = np.asarray([[token_ids[identifier], token_ids[identifier], eos]
                               for identifier in ids[split][:len(rows)]], dtype=np.int64)
        np.testing.assert_array_equal(np.asarray(rows), expected)

    batch = next(iter(dataset.load("train", 3)))
    expected_batch = np.asarray(windows["train"][:3])
    np.testing.assert_array_equal(batch.inputs["tokens"].numpy(), expected_batch[:, :-1])
    np.testing.assert_array_equal(batch.targets["target"].numpy(), expected_batch[:, 1:])


def test_row_group_cache_is_published_before_budget_stop_and_reused(tmp_path, monkeypatch):
    dataset, _ids, _token_ids, _dataset_dir = _make_dataset(
        tmp_path, monkeypatch, train_windows=3, evaluation_windows=1,
        train_documents=6, row_group_size=32,
    )
    samples = dataset._samples("train")
    first = [next(samples)]
    cache_dirs = [path for path in (tmp_path / "cache").iterdir() if path.is_dir()]
    assert len(cache_dirs) == 1
    assert (cache_dirs[0] / "manifest.json").is_file()
    assert (cache_dirs[0] / "windows.bin").is_file()
    metadata = json.loads((cache_dirs[0] / "manifest.json").read_text(encoding="utf-8"))
    assert metadata["shape"][0] == 6
    first.extend(samples)
    assert len(first) == 3

    def unexpected_tokenization(*_args, **_kwargs):
        raise AssertionError("cached row group should bypass tokenization")

    dataset.tokenizer.encode = unexpected_tokenization
    second = list(dataset._samples("train"))
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


def test_cache_shape_dtype_and_digest_errors_are_visible(tmp_path, monkeypatch):
    dataset, _ids, _token_ids, _dataset_dir = _make_dataset(tmp_path, monkeypatch)
    list(dataset._samples("train"))
    cache_dir = next(path for path in (tmp_path / "cache").iterdir() if path.is_dir())
    metadata_path = cache_dir / "manifest.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["shape"][1] += 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="shape or dtype"):
        list(dataset._samples("train"))

    metadata["shape"][1] -= 1
    metadata["dtype"] = "float32"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="shape or dtype"):
        list(dataset._samples("train"))

    metadata["dtype"] = "<i8"
    metadata["identity"]["revision"] = "c" * 40
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        list(dataset._samples("train"))

    metadata["identity"]["revision"] = "b" * 40
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    data_path = cache_dir / "windows.bin"
    payload = bytearray(data_path.read_bytes())
    payload[0] ^= 0x01
    data_path.write_bytes(payload)
    with pytest.raises(ValueError, match="digest"):
        list(dataset._samples("train"))


def test_missing_acquired_file_and_symlinked_cache_root_fail_visibly(tmp_path, monkeypatch):
    dataset, _ids, _token_ids, dataset_dir = _make_dataset(tmp_path, monkeypatch)
    (dataset_dir / "corpus.parquet").unlink()
    with pytest.raises(ValueError, match="missing or unsafe"):
        list(dataset._samples("train"))

    other = tmp_path / "other"
    other.mkdir()
    cache_link = tmp_path / "cache-link"
    cache_link.symlink_to(other, target_is_directory=True)
    monkeypatch.setenv("NNMODELLING_DATASET_CACHE_ROOT", str(cache_link))
    dataset, _ids, _token_ids, _dataset_dir = _make_dataset(tmp_path / "fresh", monkeypatch)
    monkeypatch.setenv("NNMODELLING_DATASET_CACHE_ROOT", str(cache_link))
    with pytest.raises(ValueError, match="cache root must not be a symlink"):
        list(dataset._samples("train"))


def test_tokenizer_inference_uses_bundled_assets_without_cache(tmp_path, monkeypatch):
    dataset, ids, token_ids, _dataset_dir = _make_dataset(tmp_path, monkeypatch)
    monkeypatch.delenv("NNMODELLING_DATASET_CACHE_ROOT", raising=False)
    value = dataset.tokenize("word0 word1")
    assert value.dtype == torch.int64
    assert value.shape == (1, 2)
    assert value.tolist() == [[token_ids[ids["train"][0]], token_ids[ids["train"][1]]]]
    decoded = dataset.untokenize(value)
    assert decoded == "word0 word1"
