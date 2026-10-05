from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from nnmodel_a64143c8_07c2_40e7_8f89_37c33d64970b import Model


CONTEXT_LENGTH = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Tiny Shakespeare text with the downloaded model wheel.")
    parser.add_argument("prompt", nargs="?", default="ROMEO:\n", help="starting text (default: %(default)r)")
    parser.add_argument("--tokens", type=int, default=100, help="number of greedy characters to generate (0-2000)")
    parser.add_argument("--weights", type=Path, help="optional compatible safetensors file")
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
