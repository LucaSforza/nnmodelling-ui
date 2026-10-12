from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib
import importlib.resources
import json
from pathlib import Path
import sys

import torch
from tokenizers import Tokenizer


_DATASET_ASSET = Path(
    "_snapshot/project/datasets/dsv4.fineweb-edu-1.0.0"
)


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    *,
    context: int,
    eos_token_id: int,
) -> str:
    """Greedily generate text while keeping the original token-ID sequence."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if context < 1:
        raise ValueError("context must be positive")

    token_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    if not token_ids:
        raise ValueError("prompt must produce at least one tokenizer ID")

    for _ in range(max_new_tokens):
        input_ids = torch.tensor([token_ids[-context:]], dtype=torch.long)
        logits = model.logits(input_ids)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise TypeError("the logits operation must return a [B, T, V] tensor")
        if logits.shape[0] != 1 or logits.shape[1] == 0:
            raise ValueError("the logits operation returned an empty or non-singleton batch")
        next_id = int(logits[0, -1].argmax(dim=-1).item())
        token_ids.append(next_id)
        if next_id == eos_token_id:
            break

    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _bundled_assets(package) -> tuple[object, object]:
    root = importlib.resources.files(package)
    asset_dir = root.joinpath(*_DATASET_ASSET.parts)
    return asset_dir.joinpath("tokenizer.json"), asset_dir.joinpath("config.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text with a trained miniDeepSeekv4 wheel.")
    parser.add_argument(
        "--model-package",
        required=True,
        help="import name of the installed exported model wheel",
    )
    parser.add_argument("--tokenizer", type=Path, help="optional path to the frozen tokenizer.json asset")
    parser.add_argument("--weights", type=Path, help="optional trained safetensors checkpoint override")
    parser.add_argument("--device", default="cuda", help="inference device (default: cuda)")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("prompt", nargs="?", help="prompt text; when omitted, read stdin")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    if not prompt.strip():
        raise SystemExit("prompt must not be empty")
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be positive")

    package = importlib.import_module(args.model_package)
    bundled_tokenizer, bundled_config = _bundled_assets(package)
    tokenizer_path = args.tokenizer or bundled_tokenizer
    config_path = args.tokenizer.with_name("config.json") if args.tokenizer else bundled_config
    if args.tokenizer and not config_path.is_file():
        config_path = bundled_config

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"could not read miniDeepSeek tokenizer config: {error}") from error
    context = config.get("context")
    eos_text = config.get("eos_token")
    if context != 512 or not isinstance(eos_text, str):
        raise SystemExit("bundled config must declare context 512 and an EOS token")

    with ExitStack() as stack:
        tokenizer_file = stack.enter_context(importlib.resources.as_file(tokenizer_path))
        try:
            tokenizer = Tokenizer.from_file(str(tokenizer_file))
        except Exception as error:
            raise SystemExit(f"could not load miniDeepSeek tokenizer: {error}") from error
        if tokenizer.get_vocab_size() != 129280:
            raise SystemExit("the frozen miniDeepSeek tokenizer must have 129280 token IDs")
        eos_token_id = tokenizer.token_to_id(eos_text)
        if eos_token_id is None:
            raise SystemExit("configured EOS token is absent from the frozen tokenizer")
        model = package.Model(
            weights_path=args.weights,
            device=args.device,
        )
        print(
            generate(
                model,
                tokenizer,
                prompt,
                args.max_new_tokens,
                context=context,
                eos_token_id=eos_token_id,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
