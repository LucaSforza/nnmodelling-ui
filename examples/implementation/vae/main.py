from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch

from download_wheel import ARTIFACT, load_artifact


LATENT_SHAPE = (1, 32)
IMAGE_SHAPE = (28, 28)
MAX_PRIOR_SAMPLES = 64


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def image_result(value: Any, label: str) -> np.ndarray:
    image = as_numpy(value)
    if image.shape != IMAGE_SHAPE or not np.isfinite(image).all():
        raise ValueError(f"{label} must be one finite 28 by 28 image, got {image.shape}")
    return image.astype(np.float32, copy=False)


def run_model(model: Any, raw_image: np.ndarray, prior_latents: list[Any]) -> dict[str, Any]:
    """Exercise only the exported wheel's public Model inference and operation API."""
    source = model.inference(raw_image)
    alias = model.infer(raw_image)
    inference_image = image_result(source, "inference result")
    alias_image = image_result(alias, "infer result")
    if not np.allclose(inference_image, alias_image, rtol=1e-6, atol=1e-7):
        raise ValueError("infer() and inference() returned different images")

    encoded = model.encode(raw_image)
    encoded_again = model.encode(raw_image)
    encoded_array = as_numpy(encoded)
    if encoded_array.shape != LATENT_SHAPE or not np.isfinite(encoded_array).all():
        raise ValueError(f"encode must return one finite (1, 32) latent tensor, got {encoded_array.shape}")
    if not np.array_equal(encoded_array, as_numpy(encoded_again)):
        raise ValueError("encode must be deterministic in evaluation mode")

    reconstruction = image_result(model.decode(encoded), "decoded encoding")
    if not np.allclose(inference_image, reconstruction, rtol=1e-5, atol=1e-6):
        raise ValueError("decode(encode(image)) did not match model inference")
    prior_images = [image_result(model.decode(latent), "decoded prior") for latent in prior_latents]
    return {
        "inference": inference_image,
        "infer": alias_image,
        "encoded": encoded,
        "reconstruction": reconstruction,
        "prior_images": prior_images,
    }


def read_input(path: Path | None) -> np.ndarray:
    if path is None:
        return np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L"))
    if pixels.shape != IMAGE_SHAPE:
        raise ValueError(f"input PNG must be 28 by 28 pixels, got {pixels.shape}")
    return pixels


def load_model(weights: Path | None) -> Any:
    artifact = load_artifact(ARTIFACT)
    model_class = getattr(importlib.import_module(artifact["module"]), "Model")
    return model_class(weights_path=weights)


def save_image(array: np.ndarray, path: Path) -> None:
    pixels = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(pixels).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone MNIST VAE inference with an exported model wheel.")
    parser.add_argument("--input", type=Path, help="optional 28x28 PNG; default is a zero-valued NumPy image")
    parser.add_argument("--weights", type=Path, help="optional compatible safetensors file; defaults to bundled weights")
    parser.add_argument("--seed", type=int, default=0, help="prior sampling seed (0..2147483647)")
    parser.add_argument("--count", type=int, default=1, help="number of prior images to decode (1..64)")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "output")
    args = parser.parse_args()
    if not 0 <= args.seed <= 2_147_483_647:
        parser.error("--seed must be between 0 and 2147483647")
    if not 1 <= args.count <= MAX_PRIOR_SAMPLES:
        parser.error(f"--count must be between 1 and {MAX_PRIOR_SAMPLES}")
    return args


def main() -> int:
    args = parse_args()
    try:
        raw_image = read_input(args.input)
        torch.set_num_threads(2)
        model = load_model(args.weights)
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        prior_latents = [torch.randn(LATENT_SHAPE, generator=generator) for _ in range(args.count)]
        results = run_model(model, raw_image, prior_latents)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_image(results["reconstruction"], args.output_dir / "reconstruction.png")
        for index, image in enumerate(results["prior_images"], start=1):
            save_image(image, args.output_dir / f"prior-{index:02d}.png")
    except (OSError, ValueError, ImportError, AttributeError, json.JSONDecodeError) as error:
        print(f"VAE inference failed: {error}", file=sys.stderr)
        return 1
    print(f"saved reconstruction and {args.count} seeded prior image(s) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
