from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Iterator, Mapping, TypeVar

import torch
import os

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True)
class Batch:
    inputs: Mapping[str, torch.Tensor]
    targets: Mapping[str, torch.Tensor]


class DatasetAdapter(ABC, Generic[InputT, OutputT]):
    """Dataset boundary. Adapters own tokenization and named batch loading."""

    def __init__(self, dataset_dir: str | Path):
        self.directory = Path(dataset_dir)

    @property
    def data_directory(self) -> Path:
        """Training data root, or the dataset resource directory for local use."""
        return Path(os.environ.get("NNMODELLING_DATASET_DATA_ROOT", self.directory))

    @property
    def cache_directory(self) -> Path | None:
        """Optional writable job scratch space for derived training data."""
        value = os.environ.get("NNMODELLING_DATASET_CACHE_ROOT")
        return Path(value) if value else None

    @abstractmethod
    def tokenize(self, value: InputT) -> torch.Tensor | Mapping[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def untokenize(self, tensor: torch.Tensor) -> OutputT:
        raise NotImplementedError

    @abstractmethod
    def load(self, split: str, batch_size: int) -> Iterator[Batch]:
        raise NotImplementedError
