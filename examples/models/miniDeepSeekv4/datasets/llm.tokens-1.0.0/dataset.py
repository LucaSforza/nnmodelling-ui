from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch

from nnmodelling_runtime import Batch, DatasetAdapter


class Dataset(DatasetAdapter[str, str]):
    context_length = 128

    def __init__(self, dataset_dir: str | Path):
        super().__init__(dataset_dir)
        payload = json.loads(
            (self.directory / "vocabulary.json").read_text(encoding="utf-8")
        )
        characters = payload.get("characters")
        if (
            not isinstance(characters, list)
            or len(characters) != 65
            or any(not isinstance(char, str) or len(char) != 1 for char in characters)
        ):
            raise ValueError(
                "Tiny Shakespeare vocabulary must contain 65 single characters"
            )
        if characters != sorted(set(characters)):
            raise ValueError("Tiny Shakespeare vocabulary must be sorted and unique")
        self._characters = characters
        self._token_ids = {char: index for index, char in enumerate(characters)}

    def tokenize(self, value: str) -> torch.Tensor:
        if not isinstance(value, str):
            raise TypeError("LLM input must be text")
        if not value:
            raise ValueError("LLM input text cannot be empty")
        if len(value) > self.context_length:
            raise ValueError(
                f"LLM input is too long; maximum is {self.context_length} characters"
            )
        try:
            tokens = [self._token_ids[char] for char in value]
        except KeyError as error:
            raise ValueError(
                f"LLM input contains a character outside the bundled vocabulary: {error.args[0]!r}"
            ) from error
        return torch.tensor([tokens], dtype=torch.int64)

    def untokenize(self, tensor: torch.Tensor) -> str:
        values = tensor.detach().cpu()
        if values.ndim == 3:
            if values.shape[0] != 1 or values.shape[2] != len(self._characters):
                raise ValueError("LLM prediction must have shape [1, T, 65]")
            if not torch.isfinite(values).all():
                raise ValueError("LLM prediction logits must be finite")
            values = values.argmax(dim=-1)[0]
        elif values.ndim == 2 and values.shape[0] == 1:
            values = values[0]
        else:
            raise ValueError(
                "LLM prediction must contain one batch of token logits or IDs"
            )
        ids = values.to(torch.int64).tolist()
        if any(index < 0 or index >= len(self._characters) for index in ids):
            raise ValueError("LLM prediction contains an unknown token ID")
        return "".join(self._characters[index] for index in ids)

    def load(self, split: str, batch_size: int) -> Iterator[Batch]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        payload = json.loads((self.directory / "data.json").read_text(encoding="utf-8"))
        try:
            text = payload["splits"][split]
            starts = payload["starts"][split]
        except KeyError as error:
            raise ValueError(f"unknown LLM split {split!r}") from error
        if not text or not starts:
            raise ValueError(f"LLM split {split!r} is empty")
        ids = torch.tensor([self._token_ids[char] for char in text], dtype=torch.int64)
        windows = [ids[start : start + self.context_length + 1] for start in starts]
        if any(window.numel() != self.context_length + 1 for window in windows):
            raise ValueError(
                f"LLM split {split!r} contains an incomplete context window"
            )
        all_windows = torch.stack(windows)
        for start in range(0, len(windows), batch_size):
            batch = all_windows[start : start + batch_size]
            yield Batch(
                inputs={"tokens": batch[:, :-1]}, targets={"target": batch[:, 1:]}
            )
