#!/usr/bin/env python3
"""Create a new Tiny Shakespeare project using every complete corpus window."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

try:
    from .prepare_example_data import TEXT_URL, verified_tiny_shakespeare
except ImportError:  # Direct execution as `python tools/prepare_full_llm.py`.
    from prepare_example_data import TEXT_URL, verified_tiny_shakespeare

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PROJECT = ROOT / "examples/models/tiny-decoder-llm"
RESOURCE_RELATIVE = Path("datasets/llm.tokens-1.0.0")
CONTEXT = 128
WINDOW_CHARS = CONTEXT + 1
STRIDE = CONTEXT
TRAINING_CONFIG = {
    "epochs": 20,
    "batch_size": 64,
    "learning_rate": 0.001,
    "seed": 0,
    "publish_every_steps": 100,
}
EXCLUDED = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".computer-use",
    "node_modules", "build", "dist",
}


def split_corpus(text: str) -> dict[str, dict[str, Any]]:
    """Split contiguous characters first; enumerate complete windows inside each segment."""
    train_end = len(text) * 8 // 10
    validation_end = len(text) * 9 // 10
    bounds = {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, len(text)),
    }
    result = {}
    for name, (start, end) in bounds.items():
        segment = text[start:end]
        relative_starts = list(range(0, len(segment) - CONTEXT, STRIDE))
        consumed = relative_starts[-1] + WINDOW_CHARS if relative_starts else 0
        result[name] = {
            "text": segment,
            "start": start,
            "end": end,
            "characters": len(segment),
            "starts": relative_starts,
            "windows": len(relative_starts),
            "omittedTailCharacters": len(segment) - consumed,
        }
    return result


def full_payload(text: str, source_sha256: str, vocabulary: list[str]) -> dict[str, Any]:
    splits = split_corpus(text)
    provenance_splits = {
        name: {key: value[key] for key in ("start", "end", "characters", "windows", "omittedTailCharacters")}
        for name, value in splits.items()
    }
    return {
        "provenance": {
            "source": TEXT_URL,
            "sha256": source_sha256,
            "sourceCharacters": len(text),
            "retainedCharacters": len(text),
            "vocabularySize": len(vocabulary),
            "contextLength": CONTEXT,
            "windowCharacters": WINDOW_CHARS,
            "windowStride": STRIDE,
            "splitFractions": [0.8, 0.1, 0.1],
            "splits": provenance_splits,
            "omittedTailCharacters": sum(item["omittedTailCharacters"] for item in provenance_splits.values()),
        },
        "splits": {name: item["text"] for name, item in splits.items()},
        "starts": {name: item["starts"] for name, item in splits.items()},
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _copy_project(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if any(part in EXCLUDED for part in relative.parts) or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError(f"source example contains a symlink: {relative}")

    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name in EXCLUDED or name.endswith(".pyc")}

    shutil.copytree(source, destination, ignore=ignore, dirs_exist_ok=True)


def prepare_project(output_project: Path, cache_dir: Path) -> dict[str, Any]:
    requested = output_project.expanduser().absolute()
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(f"output project already exists: {requested}")
    destination = requested.resolve(strict=False)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output project already exists: {destination}")
    source = SOURCE_PROJECT.resolve()
    if destination.resolve().is_relative_to(source):
        raise ValueError("output project must be outside the source example directory")

    text, vocabulary, digest = verified_tiny_shakespeare(cache_dir)
    if len(text) != 1_115_394 or len(vocabulary) != 65:
        raise ValueError("verified Tiny Shakespeare source no longer matches the frozen corpus dimensions")
    payload = full_payload(text, digest, vocabulary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    try:
        _copy_project(source, destination)
        resource_dir = destination / RESOURCE_RELATIVE
        _write_json(resource_dir / "data.json", payload)
        _write_json(resource_dir / "vocabulary.json", {"characters": vocabulary, "sourceSha256": digest})
        _write_json(destination / "training.json", TRAINING_CONFIG)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return {
        "project": str(destination),
        "sourceSha256": digest,
        "characters": len(text),
        "vocabulary": len(vocabulary),
        "splits": payload["provenance"]["splits"],
        "training": TRAINING_CONFIG,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-project", type=Path, required=True, help="new project directory; existing paths are rejected")
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/nnmodelling-example-cache"))
    args = parser.parse_args()
    print(json.dumps(prepare_project(args.output_project, args.cache_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
