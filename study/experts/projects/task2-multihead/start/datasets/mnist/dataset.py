from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from PIL import Image

from nnmodelling_runtime import Batch, DatasetAdapter


class Dataset(DatasetAdapter[object, int]):
    def __init__(self, dataset_dir: str | Path):
        super().__init__(dataset_dir)

    def tokenize(self, value: object) -> torch.Tensor:
        if isinstance(value, (str, Path)):
            with Image.open(value) as image:
                array = np.asarray(image.convert("L"))
        elif isinstance(value, Image.Image):
            array = np.asarray(value.convert("L"))
        elif isinstance(value, np.ndarray):
            array = value
        else:
            raise TypeError("MNIST input must be a PNG path, PIL image, or NumPy array")
        if array.shape == (1, 28, 28):
            array = array[0]
        if array.shape != (28, 28):
            raise ValueError(f"MNIST image must have shape (28, 28), got {array.shape}")
        integer_pixels = np.issubdtype(array.dtype, np.integer)
        pixels = np.asarray(array, dtype=np.float32)
        if not np.isfinite(pixels).all():
            raise ValueError("MNIST image pixels must be finite")
        if pixels.min() < 0 or pixels.max() > 255:
            raise ValueError("MNIST image pixels must be in [0, 255]")
        if integer_pixels or pixels.max() > 1:
            pixels = pixels / 255.0
        return torch.from_numpy(pixels.copy()).reshape(1, 1, 28, 28)

    def untokenize(self, tensor: torch.Tensor) -> int:
        values = tensor.detach().cpu()
        if values.ndim == 2:
            values = values[0]
        elif values.ndim == 1:
            pass
        else:
            raise ValueError("MNIST classifier prediction must be one logits vector")
        if values.numel() != 10 or not torch.isfinite(values).all():
            raise ValueError("MNIST classifier prediction must contain ten finite logits")
        return int(values.argmax().item())

    def load(self, split: str, batch_size: int) -> Iterator[Batch]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        payload = json.loads((self.directory / "data.json").read_text(encoding="utf-8"))
        try:
            rows = payload["splits"][split]
        except KeyError as error:
            raise ValueError(f"unknown MNIST split {split!r}") from error
        if not rows:
            raise ValueError(f"MNIST split {split!r} is empty")
        images = torch.tensor([row["image"] for row in rows], dtype=torch.float32).div_(255.0)
        labels = torch.tensor([row["label"] for row in rows], dtype=torch.int64)
        if images.shape != (len(rows), 784) or torch.any(labels < 0) or torch.any(labels > 9):
            raise ValueError(f"MNIST split {split!r} contains invalid image or label data")
        images = images.reshape(-1, 1, 28, 28)
        for start in range(0, len(rows), batch_size):
            stop = start + batch_size
            yield Batch(inputs={"image": images[start:stop]}, targets={"target": labels[start:stop]})
