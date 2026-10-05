#!/usr/bin/env python3
"""Prepare small, verified training payloads for the bundled examples."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import struct
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
TEXT_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
TEXT_LIMIT = 65_536
MNIST_MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)
MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "nnmodelling-example-preparer/1"})
    with urllib.request.urlopen(request, timeout=30) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def _cached_file(cache: Path, filename: str, expected_md5: str) -> Path:
    path = cache / filename
    if not path.exists():
        failures: list[str] = []
        for mirror in MNIST_MIRRORS:
            try:
                _download(mirror + filename, path)
                break
            except OSError as error:
                path.unlink(missing_ok=True)
                failures.append(str(error))
        else:
            raise RuntimeError(f"unable to download {filename}: {'; '.join(failures)}")
    actual = hashlib.md5(path.read_bytes()).hexdigest()
    if actual != expected_md5:
        raise ValueError(f"{filename} MD5 mismatch: expected {expected_md5}, got {actual}")
    return path


def _idx_images(path: Path) -> tuple[int, bytes]:
    with gzip.open(path, "rb") as source:
        magic, count, rows, columns = struct.unpack(">IIII", source.read(16))
        if magic != 2051 or (rows, columns) != (28, 28):
            raise ValueError(f"invalid MNIST image IDX header in {path.name}")
        payload = source.read()
    if len(payload) != count * rows * columns:
        raise ValueError(f"truncated MNIST image IDX payload in {path.name}")
    return count, payload


def _idx_labels(path: Path) -> tuple[int, bytes]:
    with gzip.open(path, "rb") as source:
        magic, count = struct.unpack(">II", source.read(8))
        payload = source.read()
    if magic != 2049 or len(payload) != count:
        raise ValueError(f"invalid MNIST label IDX payload in {path.name}")
    return count, payload


def _selected_samples(images: bytes, labels: bytes, indices: list[int]) -> list[dict[str, object]]:
    return [
        {"image": list(images[index * 784 : (index + 1) * 784]), "label": labels[index]}
        for index in indices
    ]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_mnist(cache: Path) -> dict[str, list[int]]:
    files = {name: _cached_file(cache / "mnist", name, md5) for name, md5 in MNIST_FILES.items()}
    train_count, train_images = _idx_images(files["train-images-idx3-ubyte.gz"])
    train_label_count, train_labels = _idx_labels(files["train-labels-idx1-ubyte.gz"])
    test_count, test_images = _idx_images(files["t10k-images-idx3-ubyte.gz"])
    test_label_count, test_labels = _idx_labels(files["t10k-labels-idx1-ubyte.gz"])
    if train_count != train_label_count or test_count != test_label_count:
        raise ValueError("MNIST image and label counts differ")

    order = list(range(train_count))
    random.Random(20261005).shuffle(order)
    test_order = list(range(test_count))
    random.Random(20261006).shuffle(test_order)
    indices = {"train": order[:64], "validation": order[64:80], "test": test_order[:16]}
    source_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    for example in ("mnist-mlp", "mnist-vae"):
        payload = {
            "provenance": {
                "source": "MNIST IDX, official published MD5-verified gzip files",
                "sourceUrl": MNIST_MIRRORS[0],
                "publishedMd5": MNIST_FILES,
                "downloadSha256": source_hashes,
                "seed": 20261005,
                "splitSources": {
                    "train": "train-images-idx3-ubyte.gz/train-labels-idx1-ubyte.gz",
                    "validation": "train-images-idx3-ubyte.gz/train-labels-idx1-ubyte.gz",
                    "test": "t10k-images-idx3-ubyte.gz/t10k-labels-idx1-ubyte.gz",
                },
                "counts": {split: len(selected) for split, selected in indices.items()},
            },
            "indices": indices,
            "splits": {
                "train": _selected_samples(train_images, train_labels, indices["train"]),
                "validation": _selected_samples(train_images, train_labels, indices["validation"]),
                "test": _selected_samples(test_images, test_labels, indices["test"]),
            },
        }
        _write_json(ROOT / "examples" / example / "datasets" / "mnist" / "data.json", payload)
    return indices


def prepare_sine() -> dict[str, int]:
    import math

    # Each interval is generated independently before windows are extracted.
    counts = {"train": 64, "validation": 16, "test": 16}
    cursor = 0
    splits: dict[str, list[float]] = {}
    for split, windows in counts.items():
        size = windows * 33
        splits[split] = [
            math.sin((cursor + index) * 0.071) + 0.25 * math.sin((cursor + index) * 0.19)
            for index in range(size)
        ]
        cursor += size
    _write_json(
        ROOT / "examples/models/rnn-sine/datasets/sine.windows-1.0.0/series.json",
        {"provenance": {"generator": "two deterministic sine waves", "seed": 0, "window": 32, "counts": counts}, "splits": splits},
    )
    return counts


def verified_tiny_shakespeare(cache: Path) -> tuple[str, list[str], str]:
    path = cache / "tinyshakespeare.txt"
    if not path.exists():
        _download(TEXT_URL, path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != TEXT_SHA256:
        raise ValueError(f"Tiny Shakespeare SHA256 mismatch: expected {TEXT_SHA256}, got {digest}")
    full_text = raw.decode("utf-8")
    vocabulary = sorted(set(full_text))
    if len(vocabulary) != 65:
        raise ValueError(f"Tiny Shakespeare vocabulary changed: expected 65 characters, got {len(vocabulary)}")
    return full_text, vocabulary, digest


def prepare_shakespeare(cache: Path) -> dict[str, int]:
    full_text, vocabulary, digest = verified_tiny_shakespeare(cache)
    text = full_text[:TEXT_LIMIT]
    train_end = int(len(text) * 0.8)
    validation_end = int(len(text) * 0.9)
    segments = {
        "train": text[:train_end],
        "validation": text[train_end:validation_end],
        "test": text[validation_end:],
    }
    counts = {"train": 64, "validation": 16, "test": 16}
    context = 128
    selected: dict[str, list[int]] = {}
    for split, count in counts.items():
        available = len(segments[split]) - context
        if available < count:
            raise ValueError(f"Tiny Shakespeare {split} segment is too short")
        selected[split] = [round(index * (available - 1) / (count - 1)) for index in range(count)]
    directory = ROOT / "examples/models/tiny-decoder-llm/datasets/llm.tokens-1.0.0"
    _write_json(directory / "vocabulary.json", {"characters": vocabulary})
    _write_json(
        directory / "data.json",
        {
            "provenance": {
                "source": TEXT_URL,
                "sha256": digest,
                "sourceCharacters": len(full_text),
                "retainedCharacters": len(text),
                "splitFractions": [0.8, 0.1, 0.1],
                "counts": counts,
                "context": context,
            },
            "splits": segments,
            "starts": selected,
        },
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/nnmodelling-example-cache"))
    args = parser.parse_args()
    mnist = prepare_mnist(args.cache_dir)
    sine = prepare_sine()
    shakespeare = prepare_shakespeare(args.cache_dir)
    print(json.dumps({"mnist": {key: len(value) for key, value in mnist.items()}, "sine": sine, "tinyShakespeare": shakespeare}, indent=2))


if __name__ == "__main__":
    main()
