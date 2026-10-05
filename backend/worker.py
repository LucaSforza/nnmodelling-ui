from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

from . import config


MAX_METRICS_BYTES = config.MAX_METRICS_BYTES


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    encoded = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_METRICS_BYTES:
        raise ValueError(f"Training metrics exceed the {MAX_METRICS_BYTES}-byte publication limit.")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _batch_size(inputs: dict[str, Any]) -> int:
    for value in inputs.values():
        if getattr(value, "ndim", 0) > 0:
            return int(value.shape[0])
    raise ValueError("Dataset adapter yielded a batch with no batched input tensor.")


def _loss_value(value: Any) -> float:
    import torch

    if not isinstance(value, torch.Tensor) or value.ndim != 0:
        raise ValueError("The graph Loss Output must produce one scalar tensor.")
    loss = float(value.detach().cpu().item())
    if not math.isfinite(loss):
        raise ValueError("Training produced a non-finite loss.")
    return loss


def _epoch(model: Any, adapter: Any, split: str, batch_size: int, optimizer: Any | None) -> float:
    import torch

    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in adapter.load(split, batch_size):
            size = _batch_size(batch.inputs)
            if training:
                optimizer.zero_grad(set_to_none=True)
            values = model(batch.inputs, targets=batch.targets, include_loss=True)
            if not isinstance(values, dict) or "loss" not in values:
                raise ValueError("The graph has no executable scalar Loss Output.")
            loss_tensor = values["loss"]
            loss = _loss_value(loss_tensor)
            if training:
                loss_tensor.backward()
                optimizer.step()
            total += loss * size
            count += size
    if count == 0:
        raise ValueError(f"Dataset split '{split}' yielded no batches.")
    mean = total / count
    if not math.isfinite(mean):
        raise ValueError(f"Dataset split '{split}' produced a non-finite mean loss.")
    return mean


def _validation_loss(model: Any, adapter: Any, batch_size: int) -> float:
    import numpy as np
    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    was_training = model.training
    try:
        return _epoch(model, adapter, "validation", batch_size, None)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        model.train(was_training)


def _train_epochs(
    model: Any,
    adapter: Any,
    optimizer: Any,
    epochs: int,
    batch_size: int,
    publish_every_steps: int,
    metrics: dict[str, Any],
    publish: Any,
) -> None:
    global_step = 0
    last_published_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_total = 0.0
        epoch_count = 0
        window_total = 0.0
        window_count = 0
        for batch in adapter.load("train", batch_size):
            size = _batch_size(batch.inputs)
            optimizer.zero_grad(set_to_none=True)
            values = model(batch.inputs, targets=batch.targets, include_loss=True)
            if not isinstance(values, dict) or "loss" not in values:
                raise ValueError("The graph has no executable scalar Loss Output.")
            loss_tensor = values["loss"]
            loss = _loss_value(loss_tensor)
            loss_tensor.backward()
            optimizer.step()
            global_step += 1
            epoch_total += loss * size
            epoch_count += size
            window_total += loss * size
            window_count += size

            if global_step % publish_every_steps == 0:
                validation = _validation_loss(model, adapter, batch_size)
                metrics["steps"].append({
                    "step": global_step, "epoch": epoch,
                    "training_loss": window_total / window_count,
                    "validation_loss": validation,
                })
                last_published_step = global_step
                window_total = 0.0
                window_count = 0
                publish()

        if epoch_count == 0:
            raise ValueError("Dataset split 'train' yielded no batches.")
        train_loss = epoch_total / epoch_count
        if not math.isfinite(train_loss):
            raise ValueError("Training produced a non-finite epoch mean loss.")

        if global_step != last_published_step:
            validation = _validation_loss(model, adapter, batch_size)
            metrics["steps"].append({
                "step": global_step, "epoch": epoch,
                "training_loss": window_total / window_count,
                "validation_loss": validation,
            })
            last_published_step = global_step

        metrics["epochs"].append({
            "epoch": epoch, "training_loss": train_loss,
            "validation_loss": metrics["steps"][-1]["validation_loss"],
        })
        publish()


def run(snapshot: Path, output: Path, job_id: str) -> None:
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from nnmodelling_runtime import GraphModule, build_wheel, load_dataset

    training = json.loads((snapshot / "training.json").read_text(encoding="utf-8"))
    random.seed(int(training["seed"]))
    np.random.seed(int(training["seed"]) % (2**32))
    torch.manual_seed(int(training["seed"]))
    project = Path(tempfile.mkdtemp(prefix="nnmodelling-project-"))
    try:
        shutil.copy2(snapshot / "project.json", project / "model.json")
        files_root = snapshot / "files"
        for source in files_root.rglob("*"):
            if source.is_file():
                destination = project / source.relative_to(files_root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        core_dir = snapshot / "core"
        adapter = load_dataset(project)
        model = GraphModule(project, core_dir=core_dir)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
        metrics: dict[str, Any] = {"epochs": [], "steps": [], "test_loss": None}
        metrics_path = output / "metrics.json"
        _write_json(metrics_path, metrics)
        _train_epochs(
            model, adapter, optimizer, int(training["epochs"]), int(training["batch_size"]),
            int(training.get("publish_every_steps", 10)), metrics,
            lambda: _write_json(metrics_path, metrics),
        )
        test_loss = _epoch(model, adapter, "test", int(training["batch_size"]), None)
        metrics["test_loss"] = test_loss
        _write_json(metrics_path, metrics)
        weights = output / "weights.safetensors"
        state = {name: value.detach().cpu().contiguous().clone() for name, value in model.state_dict().items()}
        save_file(state, str(weights))
        build_wheel(project, core_dir, weights, output, job_id)
    finally:
        shutil.rmtree(project, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    run(args.snapshot, args.output, args.job_id)


if __name__ == "__main__":
    main()
