import json
from pathlib import Path
from typing import Iterator

import torch

from nnmodelling_runtime import Batch, DatasetAdapter


class Dataset(DatasetAdapter[float, list[float]]):
    def __init__(self, dataset_dir: str | Path):
        super().__init__(dataset_dir)
        self._data = None

    def tokenize(self, value: float) -> torch.Tensor:
        return torch.tensor([[value]], dtype=torch.float32)

    def untokenize(self, tensor: torch.Tensor) -> list[float]:
        return tensor.detach().cpu().reshape(-1).tolist()

    def load(self, split: str, batch_size: int) -> Iterator[Batch]:
        try:
            if self._data is None:
                self._data = json.loads((self.directory / "data.json").read_text(encoding="utf-8"))
            rows = self._data[split]
        except KeyError as error:
            raise ValueError(f"missing {split!r} data split") from error
        inputs = torch.tensor(rows["x"], dtype=torch.float32)
        targets = torch.tensor(rows["target"], dtype=torch.float32)
        if inputs.shape[0] != targets.shape[0] or inputs.shape[0] == 0:
            raise ValueError(f"{split!r} inputs and targets must contain equal non-empty samples")
        for start in range(0, inputs.shape[0], batch_size):
            end = start + batch_size
            yield Batch(inputs={"x": inputs[start:end]}, targets={"target": targets[start:end]})
