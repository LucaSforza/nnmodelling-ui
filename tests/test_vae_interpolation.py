from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
CONSUMER = ROOT / "examples/implementation/vae"
_DOWNLOAD_SPEC = importlib.util.spec_from_file_location("vae_interpolation_downloader", CONSUMER / "download_wheel.py")
assert _DOWNLOAD_SPEC and _DOWNLOAD_SPEC.loader
downloader = importlib.util.module_from_spec(_DOWNLOAD_SPEC)
_DOWNLOAD_SPEC.loader.exec_module(downloader)


@pytest.fixture
def consumer_main(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(CONSUMER))
    monkeypatch.setitem(sys.modules, "download_wheel", downloader)
    spec = importlib.util.spec_from_file_location("_vae_interpolation_main_test", CONSUMER / "main.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_latent_interpolation_contains_five_ordered_interior_points(consumer_main) -> None:
    start = np.zeros((1, 32), dtype=np.float32)
    end = np.full((1, 32), 6, dtype=np.float32)

    points = consumer_main.interpolate_latents(start, end)

    assert len(points) == 5
    for index, point in enumerate(points, start=1):
        np.testing.assert_allclose(point, np.full((1, 32), index, dtype=np.float32))


def test_interpolation_encodes_endpoints_decodes_order_and_saves_labeled_sheet(consumer_main, tmp_path: Path) -> None:
    class FakeModel:
        def encode(self, image: np.ndarray) -> np.ndarray:
            return np.full((1, 32), float(image[0, 0]), dtype=np.float32)

        def decode(self, latent: np.ndarray) -> np.ndarray:
            return np.full((28, 28), float(latent[0, 0]) / 6, dtype=np.float32)

    image3 = np.full((28, 28), 0, dtype=np.uint8)
    image7 = np.full((28, 28), 6, dtype=np.uint8)
    results = consumer_main.run_interpolation(FakeModel(), image3, image7)
    np.testing.assert_allclose([item[0, 0] for item in results["latents"]], [0, 1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose([item[0, 0] for item in results["images"]], np.linspace(0, 1, 7))
    assert results["labels"] == ["Digit 3", "t = 1/6", "t = 2/6", "t = 3/6", "t = 4/6", "t = 5/6", "Digit 7"]

    consumer_main.save_interpolation(results, tmp_path)

    files = sorted(tmp_path.glob("interpolation-*.png"))
    assert len(files) == 7
    assert all(Image.open(path).size == (28, 28) for path in files)
    with Image.open(tmp_path / "interpolation.png") as sheet:
        assert sheet.size == (236 * 7, 224 + 48 + 16)
        assert sheet.mode == "RGB"
        # The caption band contains rendered dark text for every labeled output.
        assert np.asarray(sheet.crop((0, 224, sheet.width, sheet.height))).min() < 100


@pytest.mark.parametrize("digit", [3, 7])
def test_checked_in_endpoint_png_is_labeled_mnist_test_example(digit: int) -> None:
    data = json.loads((ROOT / "examples/models/mnist-vae/datasets/mnist/data.json").read_text(encoding="utf-8"))
    row = next(item for item in data["splits"]["test"] if item["label"] == digit)
    expected = np.asarray(row["image"], dtype=np.uint8).reshape((28, 28))

    with Image.open(CONSUMER / "assets" / f"mnist-test-{digit}.png") as image:
        assert image.mode == "L"
        np.testing.assert_array_equal(np.asarray(image), expected)
