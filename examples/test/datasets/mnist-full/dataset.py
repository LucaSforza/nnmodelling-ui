from __future__ import annotations

import gzip
import struct
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
        ranges = {"train": ("train", 0, 55000), "validation": ("train", 55000, 60000),
                  "test": ("t10k", 0, 10000)}
        if split not in ranges:
            raise ValueError(f"unknown MNIST split {split!r}")
        prefix, start, end = ranges[split]
        with gzip.open(self.directory / f"{prefix}-images-idx3-ubyte.gz", "rb") as source:
            header = source.read(16)
            if len(header) != 16:
                raise ValueError("truncated MNIST image header")
            magic, count, rows, columns = struct.unpack(">IIII", header)
            payload = source.read()
        expected = 10000 if prefix == "t10k" else 60000
        if (magic, count, rows, columns) != (2051, expected, 28, 28) or len(payload) != count * 784:
            raise ValueError("invalid MNIST image data")
        images = np.frombuffer(payload, dtype=np.uint8).reshape(count, 1, 28, 28)
        with gzip.open(self.directory / f"{prefix}-labels-idx1-ubyte.gz", "rb") as source:
            header = source.read(8)
            if len(header) != 8:
                raise ValueError("truncated MNIST label header")
            magic, label_count = struct.unpack(">II", header)
            labels = np.frombuffer(source.read(), dtype=np.uint8)
        if magic != 2049 or label_count != count or len(labels) != count or np.any(labels > 9):
            raise ValueError("invalid MNIST label data")
        for offset in range(start, end, batch_size):
            stop = min(offset + batch_size, end)
            batch = torch.from_numpy(images[offset:stop].copy()).to(torch.float32).div_(255.0)
            target = torch.from_numpy(labels[offset:stop].copy()).to(torch.int64)
            yield Batch(inputs={"image": batch}, targets={"target": target})
