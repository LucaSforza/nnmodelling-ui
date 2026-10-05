from __future__ import annotations

import importlib.util
import json
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "examples/local-training": "datasets/tiny-regression",
    "examples/mnist-mlp": "datasets/mnist",
    "examples/mnist-vae": "datasets/mnist",
    "examples/rnn-sine": "datasets/sine.windows-1.0.0",
    "examples/tiny-decoder-llm": "datasets/llm.tokens-1.0.0",
}


def adapter(example: str, resource: str):
    path = ROOT / example / resource / "dataset.py"
    spec = importlib.util.spec_from_file_location(f"example_dataset_{example.replace('-', '_')}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Dataset(path.parent)


def batches(dataset, split: str, batch_size: int = 7):
    return list(dataset.load(split, batch_size))


@pytest.mark.parametrize(("example", "resource"), DATASETS.items())
def test_each_dataset_declares_a_python_adapter_and_uv_dependencies(example: str, resource: str):
    directory = ROOT / example / resource
    manifest = json.loads((directory / "manifest.json").read_text())
    project = tomllib.loads((directory / "pyproject.toml").read_text(encoding="utf-8"))
    assert manifest["entrypoints"]["python"] == {"language": "python", "file": "dataset.py"}
    dependencies = project["project"]["dependencies"]
    assert "nnmodelling-runtime>=0.1.0" in dependencies
    assert (directory / "dataset.py").is_file()
    assert ("Pillow>=11,<13" in dependencies) == ("mnist" in example)


def test_local_regression_adapter_has_explicit_splits_and_scalar_round_trip():
    dataset = adapter("examples/local-training", "datasets/tiny-regression")
    assert dataset.tokenize(2.5).shape == (1, 1)
    assert dataset.untokenize(torch.tensor([[2.5]])) == [2.5]
    assert sum(batch.inputs["x"].shape[0] for batch in batches(dataset, "train")) == 4
    assert sum(batch.inputs["x"].shape[0] for batch in batches(dataset, "validation")) == 2
    assert sum(batch.inputs["x"].shape[0] for batch in batches(dataset, "test")) == 2


@pytest.mark.parametrize(
    ("example", "target_shape"),
    [
        ("examples/mnist-mlp", (64,)),
        ("examples/mnist-vae", (64, 784)),
    ],
)
def test_mnist_payloads_are_verified_bounded_and_split_disjoint(example: str, target_shape: tuple[int, ...]):
    dataset = adapter(example, "datasets/mnist")
    payload = json.loads((ROOT / example / "datasets/mnist/data.json").read_text())
    assert {name: len(rows) for name, rows in payload["splits"].items()} == {
        "train": 64,
        "validation": 16,
        "test": 16,
    }
    assert set(payload["indices"]["train"]).isdisjoint(payload["indices"]["validation"])
    assert all(payload["provenance"]["publishedMd5"].values())
    assert payload["provenance"]["splitSources"]["test"].startswith("t10k-")
    assert payload["provenance"]["downloadSha256"]
    loaded = batches(dataset, "train", batch_size=64)
    assert len(loaded) == 1
    batch = loaded[0]
    assert batch.inputs["image"].shape == (64, 1, 28, 28)
    assert batch.inputs["image"].dtype == torch.float32
    assert batch.inputs["image"].min() >= 0 and batch.inputs["image"].max() <= 1
    assert batch.targets["target"].shape == target_shape
    assert sum(item.inputs["image"].shape[0] for item in batches(dataset, "validation")) == 16
    assert sum(item.inputs["image"].shape[0] for item in batches(dataset, "test")) == 16


def test_mnist_classification_raw_images_normalize_and_decode(tmp_path: Path):
    dataset = adapter("examples/mnist-mlp", "datasets/mnist")
    raw = np.zeros((28, 28), dtype=np.uint8)
    raw[4:9, 10:17] = 255
    png = tmp_path / "digit.png"
    Image.fromarray(raw).save(png)
    by_path = dataset.tokenize(png)
    with Image.open(png) as image:
        by_pil = dataset.tokenize(image)
    by_array = dataset.tokenize(raw)
    assert by_path.shape == (1, 1, 28, 28)
    assert by_path.dtype == torch.float32 and by_path.max() == 1
    assert torch.equal(by_path, by_pil) and torch.equal(by_path, by_array)
    assert dataset.untokenize(torch.tensor([[0.0, 1.0, 9.0, 0, 0, 0, 0, 0, 0, 0]])) == 2
    with pytest.raises(ValueError, match="shape"):
        dataset.tokenize(np.zeros((27, 28), dtype=np.uint8))
    with pytest.raises(TypeError, match="PNG path"):
        dataset.tokenize(object())


def test_mnist_vae_returns_normalized_reconstruction_array():
    dataset = adapter("examples/mnist-vae", "datasets/mnist")
    image = np.full((28, 28), 64, dtype=np.uint8)
    assert dataset.tokenize(image).shape == (1, 1, 28, 28)
    decoded = dataset.untokenize(torch.full((1, 784), 0.5))
    assert decoded.shape == (28, 28) and decoded.dtype == np.float32
    assert np.all(decoded == 0.5)
    with pytest.raises(ValueError, match="28 by 28"):
        dataset.untokenize(torch.zeros(783))


@pytest.mark.parametrize(
    ("example", "resource"),
    [
        ("examples/mnist-mlp", "datasets/mnist"),
        ("examples/mnist-vae", "datasets/mnist"),
    ],
)
def test_mnist_integer_pixels_always_scale_while_normalized_floats_are_preserved(example: str, resource: str):
    dataset = adapter(example, resource)
    integer_ones = dataset.tokenize(np.ones((28, 28), dtype=np.uint8))
    float_ones = dataset.tokenize(np.ones((28, 28), dtype=np.float32))
    assert torch.allclose(integer_ones, torch.full_like(integer_ones, 1 / 255))
    assert torch.equal(float_ones, torch.ones_like(float_ones))


def test_sine_series_is_split_before_nonoverlapping_windowing():
    dataset = adapter("examples/rnn-sine", "datasets/sine.windows-1.0.0")
    payload = json.loads((ROOT / "examples/rnn-sine/datasets/sine.windows-1.0.0/series.json").read_text())
    assert payload["provenance"]["counts"] == {"train": 64, "validation": 16, "test": 16}
    for split, count in payload["provenance"]["counts"].items():
        assert len(payload["splits"][split]) == count * 33
        split_batches = batches(dataset, split, batch_size=9)
        assert sum(batch.inputs["sequence"].shape[0] for batch in split_batches) == count
        assert all(batch.inputs["sequence"].shape[1:] == (32, 1) for batch in split_batches)
        assert all(batch.targets["target"].shape[1:] == (1,) for batch in split_batches)
    assert dataset.tokenize(list(range(32))).shape == (1, 32, 1)
    assert dataset.untokenize(torch.tensor([[1.25]])) == 1.25
    with pytest.raises(ValueError, match="32 finite"):
        dataset.tokenize([float("nan")] * 32)


def test_tiny_shakespeare_adapter_has_full_vocabulary_and_disjoint_windows():
    resource = ROOT / "examples/tiny-decoder-llm/datasets/llm.tokens-1.0.0"
    dataset = adapter("examples/tiny-decoder-llm", "datasets/llm.tokens-1.0.0")
    vocabulary = json.loads((resource / "vocabulary.json").read_text())["characters"]
    payload = json.loads((resource / "data.json").read_text())
    assert len(vocabulary) == 65 and vocabulary == sorted(set(vocabulary))
    assert payload["provenance"]["sourceCharacters"] > payload["provenance"]["retainedCharacters"]
    assert payload["provenance"]["retainedCharacters"] == 65_536
    assert payload["provenance"]["sha256"] == "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
    assert {key: len(value) for key, value in payload["starts"].items()} == {
        "train": 64,
        "validation": 16,
        "test": 16,
    }
    for split, starts in payload["starts"].items():
        count = len(starts)
        loaded = batches(dataset, split, batch_size=7)
        assert sum(batch.inputs["tokens"].shape[0] for batch in loaded) == count
        assert all(batch.inputs["tokens"].dtype == torch.int64 for batch in loaded)
        assert all(batch.inputs["tokens"].shape[1] == 128 for batch in loaded)
        assert all(torch.equal(batch.targets["target"][:, :-1], batch.inputs["tokens"][:, 1:]) for batch in loaded)
    raw = "To be, or not to be."
    tokens = dataset.tokenize(raw)
    assert tokens.shape == (1, len(raw)) and tokens.dtype == torch.int64
    logits = torch.nn.functional.one_hot(tokens, num_classes=65).float()
    assert dataset.untokenize(logits) == raw
    nonfinite_logits = logits.clone()
    nonfinite_logits[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="logits must be finite"):
        dataset.untokenize(nonfinite_logits)
    with pytest.raises(ValueError, match="empty"):
        dataset.tokenize("")
    with pytest.raises(ValueError, match="too long"):
        dataset.tokenize("x" * 129)
    with pytest.raises(ValueError, match="outside the bundled vocabulary"):
        dataset.tokenize("😀")
