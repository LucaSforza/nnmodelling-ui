from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Sequence

import torch

from nnmodelling_runtime import Batch, DatasetAdapter


class Dataset(DatasetAdapter[Sequence[float], float]):
    def __init__(self, dataset_dir: str | Path):
        super().__init__(dataset_dir)

    def tokenize(self, value: Sequence[float]) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.shape == (32,):
            tensor = tensor.reshape(32, 1)
        if tensor.shape != (32, 1) or not torch.isfinite(tensor).all():
            raise ValueError("sine input must contain exactly 32 finite values")
        return tensor.unsqueeze(0)

    def untokenize(self, tensor: torch.Tensor) -> float:
        values = tensor.detach().cpu()
        if values.numel() != 1 or not torch.isfinite(values).all():
            raise ValueError("sine prediction must contain one finite value")
        return float(values.item())

    def load(self, split: str, batch_size: int) -> Iterator[Batch]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        payload = json.loads((self.directory / "series.json").read_text(encoding="utf-8"))
        try:
            series = torch.tensor(payload["splits"][split], dtype=torch.float32)
        except KeyError as error:
            raise ValueError(f"unknown sine split {split!r}") from error
        if series.numel() < 33 or not torch.isfinite(series).all():
            raise ValueError(f"sine split {split!r} must contain at least 33 finite samples")
        inputs = series.unfold(0, 32, 33).contiguous().unsqueeze(-1)
        targets = series[32::33].reshape(-1, 1)
        if inputs.shape[0] != targets.shape[0] or inputs.shape[0] == 0:
            raise ValueError(f"sine split {split!r} has no complete windows")
        for start in range(0, inputs.shape[0], batch_size):
            stop = start + batch_size
            yield Batch(inputs={"sequence": inputs[start:stop]}, targets={"target": targets[start:stop]})
