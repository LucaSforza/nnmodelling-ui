from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from download_wheel import ARTIFACT, load_artifact


LATENT_SHAPE = (1, 32)
IMAGE_SHAPE = (28, 28)
MAX_PRIOR_SAMPLES = 64
INTERPOLATION_STEPS = 5
ASSET_DIR = Path(__file__).parent / "assets"
INTERPOLATION_ENDPOINTS = (
    ("digit-3", ASSET_DIR / "mnist-test-3.png"),
    ("digit-7", ASSET_DIR / "mnist-test-7.png"),
)


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


def interpolate_latents(start: Any, end: Any, count: int = INTERPOLATION_STEPS) -> list[Any]:
    start_array, end_array = as_numpy(start), as_numpy(end)
    if start_array.shape != end_array.shape or start_array.shape != LATENT_SHAPE:
        raise ValueError(f"endpoint encodings must both have shape {LATENT_SHAPE}")
    if not np.isfinite(start_array).all() or not np.isfinite(end_array).all():
        raise ValueError("endpoint encodings must be finite")
    if count < 1:
        raise ValueError("interpolation count must be positive")
    # Keep the exported model's tensor type/device when its public encoder returns a tensor.
    return [start + (end - start) * ((index + 1) / (count + 1)) for index in range(count)]


def run_interpolation(model: Any, image3: np.ndarray, image7: np.ndarray) -> dict[str, Any]:
    latent3, latent7 = model.encode(image3), model.encode(image7)
    for label, latent in (("digit 3 encoding", latent3), ("digit 7 encoding", latent7)):
        values = as_numpy(latent)
        if values.shape != LATENT_SHAPE or not np.isfinite(values).all():
            raise ValueError(f"{label} must be one finite (1, 32) latent tensor, got {values.shape}")
    latents = [latent3, *interpolate_latents(latent3, latent7), latent7]
    images = [image_result(model.decode(latent), f"interpolation image {index}") for index, latent in enumerate(latents)]
    labels = ["Digit 3", "t = 1/6", "t = 2/6", "t = 3/6", "t = 4/6", "t = 5/6", "Digit 7"]
    return {"latents": latents, "images": images, "labels": labels}


def save_interpolation(results: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = [
        "interpolation-00-digit-3.png",
        "interpolation-01-t-1-6.png",
        "interpolation-02-t-2-6.png",
        "interpolation-03-t-3-6.png",
        "interpolation-04-t-4-6.png",
        "interpolation-05-t-5-6.png",
        "interpolation-06-digit-7.png",
    ]
    for image, filename in zip(results["images"], filenames, strict=True):
        save_image(image, output_dir / filename)

    cell_width, image_size, label_height, padding = 236, 224, 48, 8
    sheet = Image.new("RGB", (cell_width * len(filenames), image_size + label_height + padding * 2), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=20)
    for index, (image, label) in enumerate(zip(results["images"], results["labels"], strict=True)):
        pixels = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        tile = Image.fromarray(pixels).resize((image_size, image_size), Image.Resampling.NEAREST).convert("RGB")
        x = index * cell_width + (cell_width - image_size) // 2
        sheet.paste(tile, (x, padding))
        bounds = draw.textbbox((0, 0), label, font=font)
        text_width = bounds[2] - bounds[0]
        draw.text((index * cell_width + (cell_width - text_width) // 2, image_size + padding + 8), label, fill="black", font=font)
    sheet.save(output_dir / "interpolation.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone MNIST VAE inference with an exported model wheel.")
    parser.add_argument("--input", type=Path, help="optional 28x28 PNG; default is a zero-valued NumPy image")
    parser.add_argument("--weights", type=Path, help="optional compatible safetensors file; defaults to bundled weights")
    parser.add_argument("--seed", type=int, default=0, help="prior sampling seed (0..2147483647)")
    parser.add_argument("--count", type=int, default=1, help="number of prior images to decode (1..64)")
    parser.add_argument("--interpolate", action="store_true", help="encode bundled MNIST 3 and 7 images and save a five-step latent interpolation")
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
        torch.set_num_threads(2)
        model = load_model(args.weights)
        if args.interpolate:
            endpoints = [read_input(path) for _, path in INTERPOLATION_ENDPOINTS]
            save_interpolation(run_interpolation(model, *endpoints), args.output_dir)
            print(f"saved digit 3 to 7 latent interpolation to {args.output_dir}")
            return 0

        raw_image = read_input(args.input)
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
