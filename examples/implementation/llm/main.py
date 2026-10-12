from __future__ import annotations

import argparse
import importlib
from importlib import metadata
import sys
from pathlib import Path

import torch

try:
    from nnm_tiny_decoder_llm import Model
except ModuleNotFoundError as error:
    if error.name != "nnm_tiny_decoder_llm":
        raise
    installed = metadata.distribution("nnm-tiny-decoder-llm")
    legacy_packages = {
        path.parts[0]
        for path in installed.files or ()
        if len(path.parts) == 2
        and path.parts[1] == "__init__.py"
        and path.parts[0].startswith("nnmodel_")
        and path.parts[0].isidentifier()
    }
    if len(legacy_packages) != 1:
        raise ImportError("installed wheel has no unique legacy nnmodel_* import package") from error
    Model = importlib.import_module(next(iter(legacy_packages))).Model

CONTEXT_LENGTH = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Tiny Shakespeare text with the downloaded model wheel."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="ROMEO:\nThis is him\nGIULIETTA:",
        help="starting text (default: %(default)r)",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=100,
        help="number of characters to generate (0-2000)",
    )
    parser.add_argument(
        "--weights", type=Path, help="optional compatible safetensors file"
    )
    args = parser.parse_args()
    if not 0 <= args.tokens <= 2000:
        parser.error("--tokens must be between 0 and 2000")
    if not args.prompt:
        parser.error("prompt cannot be empty")
    return args


def main() -> int:
    args = parse_args()
    torch.set_num_threads(2)
    model = Model(weights_path=args.weights)
    generated = args.prompt
    for index in range(args.tokens):
        context = generated[-CONTEXT_LENGTH:]
        prediction = model.inference(context) if index == 0 else model.infer(context)
        if not prediction:
            raise RuntimeError("model returned an empty prediction")
        generated += prediction[-1]

    sys.stdout.write(generated)
    if not generated.endswith("\n"):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
