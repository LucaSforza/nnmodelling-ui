#!/usr/bin/env python3
"""Read-only corpus probe for the miniDeepSeekv4 FineWeb dataset adapter."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from nnmodelling_runtime import load_dataset


EXPECTED_VOCABULARY_SIZE = 129_280


def _split_summary(adapter: Any, split: str, batch_size: int, vocabulary_size: int) -> dict[str, int]:
    config = json.loads((adapter.directory / "config.json").read_text(encoding="utf-8"))
    context = config.get("context")
    if type(context) is not int or context < 2:
        raise ValueError("Dataset context must be an integer of at least two tokens.")

    if split == "train":
        budget = config.get("train_token_budget")
        if type(budget) is not int or budget < 1:
            raise ValueError("Training token budget must be a positive integer.")
        expected_tokens = (budget // context) * context
    else:
        windows = config.get(f"{split}_windows")
        if type(windows) is not int or windows < 1:
            raise ValueError(f"{split} window count must be a positive integer.")
        expected_tokens = windows * context

    batches = windows_seen = tokens_seen = 0
    minimum = vocabulary_size
    maximum = -1
    for batch in adapter.load(split, batch_size):
        inputs = batch.inputs.get("tokens")
        targets = batch.targets.get("target")
        if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
            raise ValueError(f"{split} batches must contain token input and target tensors.")
        if inputs.ndim != 2 or targets.shape != inputs.shape or inputs.shape[1] != context:
            raise ValueError(f"{split} batch has an unexpected window shape.")
        if not inputs.shape[0] or inputs.shape[0] > batch_size:
            raise ValueError(f"{split} batch has an unexpected batch size.")
        if not torch.isfinite(inputs).all().item() or not torch.isfinite(targets).all().item():
            raise ValueError(f"{split} batch contains non-finite token values.")
        if not torch.equal(inputs[:, 1:], targets[:, :-1]):
            raise ValueError(f"{split} targets are not the one-token shift of their inputs.")

        low = min(int(inputs.min().item()), int(targets.min().item()))
        high = max(int(inputs.max().item()), int(targets.max().item()))
        if low < 0 or high >= vocabulary_size:
            raise ValueError(f"{split} token ID is outside the tokenizer vocabulary.")
        minimum = min(minimum, low)
        maximum = max(maximum, high)
        batches += 1
        windows_seen += int(inputs.shape[0])
        tokens_seen += int(inputs.numel())

    if tokens_seen != expected_tokens:
        raise ValueError(
            f"{split} yielded {tokens_seen} input tokens; expected {expected_tokens} from dataset config."
        )
    return {
        "batches": batches,
        "windows": windows_seen,
        "input_tokens": tokens_seen,
        "token_min": minimum if maximum >= 0 else -1,
        "token_max": maximum,
    }


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_dir", type=Path, help="frozen project directory")
    parser.add_argument("--batch", type=int, default=16, help="batch size (default: 16)")
    parser.add_argument(
        "--split", choices=("all", "train", "validation", "test"), default="all",
        help="split to inspect (default: all)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    started = time.monotonic()
    try:
        if args.batch < 1:
            raise ValueError("Batch size must be positive.")
        adapter = load_dataset(args.project_dir)
        tokenizer = getattr(adapter, "tokenizer", None)
        if tokenizer is None or not callable(getattr(tokenizer, "get_vocab_size", None)):
            raise ValueError("Active dataset adapter does not expose its tokenizer vocabulary.")
        vocabulary_size = int(tokenizer.get_vocab_size(with_added_tokens=True))
        if vocabulary_size != EXPECTED_VOCABULARY_SIZE:
            raise ValueError(
                f"Tokenizer vocabulary has {vocabulary_size} entries; "
                f"expected {EXPECTED_VOCABULARY_SIZE}."
            )

        splits = ("train", "validation", "test") if args.split == "all" else (args.split,)
        summary = {
            split: _split_summary(adapter, split, args.batch, vocabulary_size)
            for split in splits
        }
        result = {
            "status": "ok",
            "batch_size": args.batch,
            "vocabulary_size": vocabulary_size,
            "splits": summary,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as error:
        print(json.dumps({
            "status": "error",
            "error": str(error),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }, sort_keys=True))
        return 1


if __name__ == "__main__":
    sys.exit(main())
