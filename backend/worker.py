from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

from . import config


MAX_METRICS_BYTES = config.MAX_METRICS_BYTES
MAX_CHECKPOINT_JSON_BYTES = MAX_METRICS_BYTES
CHECKPOINT_EXIT_CODE = 75


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


def _rng_state() -> dict[str, Any]:
    import numpy as np
    import torch

    numpy = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy[0], numpy[1].tolist(), int(numpy[2]), int(numpy[3]), float(numpy[4])],
        "torch": torch.random.get_rng_state(),
    }


def _tuplify(value: Any) -> Any:
    return tuple(_tuplify(item) for item in value) if isinstance(value, list) else value


def _restore_rng(state: dict[str, Any]) -> None:
    import numpy as np
    import torch

    random.setstate(_tuplify(state["python"]))
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.random.set_rng_state(state["torch"])


def _pack_value(value: Any, tensors: dict[str, Any], prefix: str) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        key = f"{prefix}.{len(tensors)}"
        tensors[key] = value.detach().cpu().contiguous().clone()
        return {"tensor": key}
    if isinstance(value, dict):
        return {"__dict__": [[key, _pack_value(item, tensors, f"{prefix}.{key}")] for key, item in value.items()]}
    if isinstance(value, (list, tuple)):
        return [_pack_value(item, tensors, f"{prefix}.{index}") for index, item in enumerate(value)]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"Checkpoint contains unsupported state value: {type(value).__name__}.")


def _unpack_value(value: Any, tensors: dict[str, Any]) -> Any:
    if isinstance(value, dict) and set(value) == {"tensor"}:
        try:
            return tensors[value["tensor"]]
        except KeyError as error:
            raise ValueError("Checkpoint references a missing tensor.") from error
    if isinstance(value, dict) and set(value) == {"__dict__"}:
        return {key: _unpack_value(item, tensors) for key, item in value["__dict__"]}
    if isinstance(value, dict):
        return {key: _unpack_value(item, tensors) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack_value(item, tensors) for item in value]
    return value


def _atomic_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _checkpoint_request_path() -> Path | None:
    value = os.environ.get("NNMODELLING_CHECKPOINT_REQUEST")
    return Path(value) if value else None


def _save_checkpoint(
    output: Path, job_id: str, scheduler_job_id: str, state: dict[str, Any], model: Any, optimizer: Any,
) -> None:
    from safetensors.torch import save_file

    output.mkdir(parents=True, exist_ok=True)
    tensors = {f"model.{key}": value.detach().cpu().contiguous().clone() for key, value in model.state_dict().items()}
    optimizer_state = _pack_value(optimizer.state_dict(), tensors, "optimizer")
    rng_state = _pack_value(_rng_state(), tensors, "rng")
    packed_state = {
        key: (_pack_value(value, tensors, f"cursor.{key}") if key == "epoch_start_rng" else value)
        for key, value in state.items()
    }
    tensor_path = output / "checkpoint.safetensors"
    temporary_tensor = output / "checkpoint.safetensors.tmp"
    save_file(tensors, str(temporary_tensor))
    os.replace(temporary_tensor, tensor_path)
    manifest = {
        "version": 1,
        "job_id": job_id,
        "scheduler_id": scheduler_job_id,
        "tensor_size": tensor_path.stat().st_size,
        "tensor_sha256": hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
        "model_keys": list(model.state_dict()),
        "optimizer": optimizer_state,
        "rng": rng_state,
        **packed_state,
    }
    data = json.dumps(manifest, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(data) > MAX_CHECKPOINT_JSON_BYTES:
        raise ValueError(f"Training checkpoint metadata exceeds the {MAX_CHECKPOINT_JSON_BYTES}-byte limit.")
    _atomic_bytes(output / "checkpoint.json", data)


def _load_checkpoint(output: Path, job_id: str) -> dict[str, Any] | None:
    json_path = output / "checkpoint.json"
    tensor_path = output / "checkpoint.safetensors"
    if not json_path.exists() and not tensor_path.exists():
        return None
    if not json_path.is_file() or not tensor_path.is_file():
        raise ValueError("Training checkpoint is incomplete.")
    import torch
    from safetensors.torch import load_file

    data = json_path.read_bytes()
    if len(data) > MAX_CHECKPOINT_JSON_BYTES:
        raise ValueError("Training checkpoint metadata exceeds the size limit.")
    manifest = json.loads(data)
    scheduler_id = manifest.get("scheduler_id")
    if manifest.get("version") != 1 or manifest.get("job_id") != job_id or not str(scheduler_id).isdigit():
        raise ValueError("Training checkpoint identity or version is invalid.")
    required = {"global_step", "epoch", "batch_index", "epoch_start_rng", "epoch_total", "epoch_count", "window_total", "window_count", "metrics", "model_keys", "optimizer", "rng", "tensor_sha256", "tensor_size"}
    if not required.issubset(manifest):
        raise ValueError("Training checkpoint metadata is incomplete.")
    tensor_bytes = tensor_path.read_bytes()
    if len(tensor_bytes) != manifest.get("tensor_size") or hashlib.sha256(tensor_bytes).hexdigest() != manifest.get("tensor_sha256"):
        raise ValueError("Training checkpoint tensor digest does not match its manifest.")
    tensors = load_file(str(tensor_path), device="cpu")
    model_state = {key.removeprefix("model."): value for key, value in tensors.items() if key.startswith("model.")}
    if set(model_state) != set(manifest.get("model_keys", [])):
        raise ValueError("Training checkpoint model tensors do not match its manifest.")
    manifest["model_state"] = model_state
    manifest["optimizer_state"] = _unpack_value(manifest["optimizer"], tensors)
    manifest["rng_state"] = _unpack_value(manifest["rng"], tensors)
    del manifest["optimizer"], manifest["rng"]
    for key in ("epoch_start_rng",):
        if key in manifest:
            manifest[key] = _unpack_value(manifest[key], tensors)
    return manifest


def _train_epochs(
    model: Any,
    adapter: Any,
    optimizer: Any,
    epochs: int,
    batch_size: int,
    publish_every_steps: int,
    metrics: dict[str, Any],
    publish: Any,
    *,
    resume: dict[str, Any] | None = None,
    checkpoint_request: Path | None = None,
    checkpoint_output: Path | None = None,
    job_id: str | None = None,
    scheduler_job_id: str | None = None,
) -> None:
    global_step = int(resume["global_step"]) if resume else 0
    last_published_step = int(metrics["steps"][-1]["step"]) if metrics["steps"] else 0
    first_epoch = int(resume["epoch"]) if resume else 1
    for epoch in range(first_epoch, epochs + 1):
        if resume and epoch == first_epoch:
            epoch_state = resume
            epoch_start_rng = epoch_state["epoch_start_rng"]
            _restore_rng(epoch_start_rng)
        else:
            epoch_start_rng = _rng_state()
            epoch_state = None
        model.train()
        epoch_total = float(epoch_state["epoch_total"]) if epoch_state else 0.0
        epoch_count = int(epoch_state["epoch_count"]) if epoch_state else 0
        window_total = float(epoch_state["window_total"]) if epoch_state else 0.0
        window_count = int(epoch_state["window_count"]) if epoch_state else 0
        completed_batches = int(epoch_state["batch_index"]) if epoch_state else 0
        checkpoint_rng_restored = False
        for batch_index, batch in enumerate(adapter.load("train", batch_size), start=1):
            if batch_index <= completed_batches:
                continue
            if epoch_state and not checkpoint_rng_restored:
                _restore_rng(epoch_state["rng_state"])
                checkpoint_rng_restored = True
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

            if checkpoint_request is not None and checkpoint_request.is_file():
                publish()
                if checkpoint_output is None or job_id is None or scheduler_job_id is None:
                    raise ValueError("Checkpoint request is set without checkpoint identity and output.")
                _save_checkpoint(checkpoint_output, job_id, scheduler_job_id, {
                    "global_step": global_step,
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "epoch_start_rng": epoch_start_rng,
                    "epoch_total": epoch_total,
                    "epoch_count": epoch_count,
                    "window_total": window_total,
                    "window_count": window_count,
                    "metrics": metrics,
                }, model, optimizer)
                raise SystemExit(CHECKPOINT_EXIT_CODE)

        if epoch_state and not checkpoint_rng_restored:
            _restore_rng(epoch_state["rng_state"])

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
        resume = None


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
        checkpoint = _load_checkpoint(output, job_id)
        if checkpoint:
            model.load_state_dict(checkpoint["model_state"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        metrics: dict[str, Any] = checkpoint.get("metrics", {"epochs": [], "steps": [], "test_loss": None}) if checkpoint else {"epochs": [], "steps": [], "test_loss": None}
        metrics_path = output / "metrics.json"
        _write_json(metrics_path, metrics)
        checkpoint_request = _checkpoint_request_path()
        scheduler_job_id = os.environ.get("NNMODELLING_SCHEDULER_ID")
        _train_epochs(
            model, adapter, optimizer, int(training["epochs"]), int(training["batch_size"]),
            int(training.get("publish_every_steps", 10)), metrics,
            lambda: _write_json(metrics_path, metrics),
            resume=checkpoint, checkpoint_request=checkpoint_request, checkpoint_output=output,
            job_id=job_id, scheduler_job_id=scheduler_job_id,
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
