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


def _move_tensors(value: Any, device: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_move_tensors(item, device) for item in value)
    return value


def _batch_tokens(batch: Any) -> int:
    """Count supervised target elements, falling back to examples for non-token tasks."""
    import torch

    targets = getattr(batch, "targets", {})
    count = sum(int(value.numel()) for value in targets.values() if isinstance(value, torch.Tensor))
    return count or _batch_size(batch.inputs)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _epoch(
    model: Any, adapter: Any, split: str, batch_size: int, optimizer: Any | None,
    *, device: Any = "cpu", precision: str = "float32",
) -> float:
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
            with torch.autocast(
                device_type=torch.device(device).type, dtype=torch.bfloat16,
                enabled=precision == "bfloat16",
            ):
                values = model(
                    _move_tensors(batch.inputs, device),
                    targets=_move_tensors(batch.targets, device), include_loss=True,
                )
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


def _validation_loss(
    model: Any, adapter: Any, batch_size: int, *, device: Any = "cpu", precision: str = "float32",
) -> float:
    state = _rng_state()
    was_training = model.training
    try:
        return _epoch(model, adapter, "validation", batch_size, None, device=device, precision=precision)
    finally:
        _restore_rng(state)
        model.train(was_training)


def _rng_state() -> dict[str, Any]:
    import numpy as np
    import torch

    numpy = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": [numpy[0], numpy[1].tolist(), int(numpy[2]), int(numpy[3]), float(numpy[4])],
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _tuplify(value: Any) -> Any:
    return tuple(_tuplify(item) for item in value) if isinstance(value, list) else value


def _restore_rng(state: dict[str, Any]) -> None:
    import numpy as np
    import torch

    random.setstate(_tuplify(state["python"]))
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.random.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


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
        key: (_pack_value(value, tensors, f"cursor.{key}")
              if key in {"epoch_start_rng", "early_stopping_state"} else value)
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
        "tensor_sha256": _sha256_file(tensor_path),
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
    if tensor_path.stat().st_size != manifest.get("tensor_size") or _sha256_file(tensor_path) != manifest.get("tensor_sha256"):
        raise ValueError("Training checkpoint tensor digest does not match its manifest.")
    tensors = load_file(str(tensor_path), device="cpu")
    model_state = {key.removeprefix("model."): value for key, value in tensors.items() if key.startswith("model.")}
    if set(model_state) != set(manifest.get("model_keys", [])):
        raise ValueError("Training checkpoint model tensors do not match its manifest.")
    manifest["model_state"] = model_state
    manifest["optimizer_state"] = _unpack_value(manifest["optimizer"], tensors)
    manifest["rng_state"] = _unpack_value(manifest["rng"], tensors)
    del manifest["optimizer"], manifest["rng"]
    for key in ("epoch_start_rng", "early_stopping_state"):
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
    device: Any = "cpu",
    precision: str = "float32",
    gradient_clip: float | None = None,
    warmup_steps: int = 0,
    max_steps: int = 0,
    gradient_accumulation_steps: int = 1,
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 0.0,
) -> None:
    import torch

    device = torch.device(device)
    if precision not in {"float32", "bfloat16"}:
        raise ValueError(f"Unsupported training precision '{precision}'.")
    if gradient_accumulation_steps < 1 or warmup_steps < 0 or max_steps < 0:
        raise ValueError("Training step and accumulation settings are invalid.")
    if type(early_stopping_patience) is not int or not 0 <= early_stopping_patience <= 10000:
        raise ValueError("Early stopping patience must be an integer from 0 to 10000.")
    try:
        valid_min_delta = math.isfinite(float(early_stopping_min_delta))
    except (OverflowError, TypeError, ValueError):
        valid_min_delta = False
    if (isinstance(early_stopping_min_delta, bool)
            or not isinstance(early_stopping_min_delta, (int, float))
            or not valid_min_delta or early_stopping_min_delta < 0):
        raise ValueError("Early stopping min_delta must be finite and nonnegative.")
    early_stopping_min_delta = float(early_stopping_min_delta)
    early_stopping_enabled = early_stopping_patience > 0
    if gradient_clip is not None and (not math.isfinite(float(gradient_clip)) or gradient_clip <= 0):
        raise ValueError("Gradient clipping threshold must be a finite positive number.")
    if precision == "bfloat16" and device.type not in {"cpu", "cuda"}:
        raise ValueError("BFloat16 training is supported only on CPU and CUDA.")
    if warmup_steps:
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])

    global_step = int(resume["global_step"]) if resume else 0
    last_published_step = int(metrics["steps"][-1]["step"]) if metrics["steps"] else 0
    processed_tokens = int(metrics.get("processed_tokens", 0))
    metrics["optimizer_updates"] = global_step
    metrics["processed_tokens"] = processed_tokens
    metrics["max_steps"] = max_steps
    first_epoch = int(resume["epoch"]) if resume else 1
    early_state = None
    if early_stopping_enabled:
        early_state = (resume.get("early_stopping_state") if resume else None) or {
            "best_state": None, "best_epoch": None, "best_validation_loss": None,
            "non_improving_epochs": 0, "stopped_epoch": None,
        }
        metrics["early_stopping"] = {
            "best_epoch": early_state["best_epoch"],
            "best_validation_loss": early_state["best_validation_loss"],
            "non_improving_epochs": early_state["non_improving_epochs"],
            "stopped_epoch": early_state["stopped_epoch"],
        }

    def restore_best() -> None:
        # A max_steps cutoff before any completed epoch has no selection to restore.
        if early_state is not None and early_state["best_state"] is not None:
            model.load_state_dict(early_state["best_state"], strict=True)

    def publish_step(epoch: int, window_total: float, window_count: int) -> None:
        nonlocal last_published_step
        validation = _validation_loss(
            model, adapter, batch_size, device=device, precision=precision,
        )
        metrics["steps"].append({
            "step": global_step, "epoch": epoch,
            "training_loss": window_total / window_count,
            "validation_loss": validation,
            "processed_tokens": processed_tokens,
            "optimizer_updates": global_step,
        })
        last_published_step = global_step

    def update_learning_rate() -> None:
        if warmup_steps <= 0:
            return
        scale = min(global_step + 1, warmup_steps) / warmup_steps
        for group in optimizer.param_groups:
            group["lr"] = float(group.get("initial_lr", group["lr"])) * scale

    if max_steps and global_step >= max_steps:
        if resume and int(resume.get("window_count", 0)) and global_step != last_published_step:
            publish_step(
                int(resume["epoch"]), float(resume["window_total"]), int(resume["window_count"]),
            )
            publish()
        restore_best()
        return

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
        batches = iter(adapter.load("train", batch_size))
        batch_index = 0
        epoch_complete = False
        stop_at_budget = False
        while True:
            group: list[tuple[Any, int, int]] = []
            while len(group) < gradient_accumulation_steps:
                try:
                    batch = next(batches)
                except StopIteration:
                    epoch_complete = True
                    break
                batch_index += 1
                if batch_index <= completed_batches:
                    continue
                if epoch_state and not checkpoint_rng_restored:
                    _restore_rng(epoch_state["rng_state"])
                    checkpoint_rng_restored = True
                group.append((batch, _batch_size(batch.inputs), _batch_tokens(batch)))
            if not group:
                break

            optimizer.zero_grad(set_to_none=True)
            group_samples = 0
            group_losses: list[tuple[float, int, int]] = []
            for batch, size, tokens in group:
                moved_inputs = _move_tensors(batch.inputs, device)
                moved_targets = _move_tensors(batch.targets, device)
                autocast = torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=precision == "bfloat16",
                )
                with autocast:
                    values = model(moved_inputs, targets=moved_targets, include_loss=True)
                if not isinstance(values, dict) or "loss" not in values:
                    raise ValueError("The graph has no executable scalar Loss Output.")
                loss_tensor = values["loss"]
                loss = _loss_value(loss_tensor.float())
                (loss_tensor.float() * size).backward()
                group_samples += size
                group_losses.append((loss, size, tokens))

            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(group_samples)
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            update_learning_rate()
            optimizer.step()
            global_step += 1
            for loss, size, tokens in group_losses:
                epoch_total += loss * size
                epoch_count += size
                window_total += loss * size
                window_count += size
                processed_tokens += tokens
            metrics["processed_tokens"] = processed_tokens
            metrics["optimizer_updates"] = global_step

            if global_step % publish_every_steps == 0:
                publish_step(epoch, window_total, window_count)
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
                    **({"early_stopping_state": early_state} if early_stopping_enabled else {}),
                }, model, optimizer)
                raise SystemExit(CHECKPOINT_EXIT_CODE)

            if max_steps and global_step >= max_steps:
                stop_at_budget = True
                break

        if epoch_state and not checkpoint_rng_restored:
            _restore_rng(epoch_state["rng_state"])

        if epoch_count == 0 and epoch_complete:
            raise ValueError("Dataset split 'train' yielded no batches.")
        if epoch_count == 0:
            break
        train_loss = epoch_total / epoch_count
        if not math.isfinite(train_loss):
            raise ValueError("Training produced a non-finite epoch mean loss.")

        if window_count and (epoch_complete or stop_at_budget) and global_step != last_published_step:
            publish_step(epoch, window_total, window_count)

        should_stop = False
        if epoch_complete:
            epoch_validation_loss = None
            if early_stopping_enabled:
                if not metrics["steps"] or metrics["steps"][-1]["step"] != global_step:
                    raise ValueError("Completed epoch has no validation result for its final optimizer step.")
                epoch_validation_loss = metrics["steps"][-1]["validation_loss"]
                previous_best = early_state["best_validation_loss"]
                if previous_best is None:
                    improved_enough = strict_minimum = True
                else:
                    strict_minimum = epoch_validation_loss < previous_best
                    improved_enough = previous_best - epoch_validation_loss > early_stopping_min_delta
                if strict_minimum:
                    early_state["best_state"] = {
                        name: value.detach().cpu().contiguous().clone()
                        for name, value in model.state_dict().items()
                    }
                    early_state["best_epoch"] = epoch
                    early_state["best_validation_loss"] = epoch_validation_loss
                early_state["non_improving_epochs"] = (
                    0 if improved_enough else early_state["non_improving_epochs"] + 1
                )
            metrics["epochs"].append({
                "epoch": epoch, "training_loss": train_loss,
                "validation_loss": (epoch_validation_loss if early_stopping_enabled
                                     else metrics["steps"][-1]["validation_loss"]),
            })
            if early_stopping_enabled:
                should_stop = early_state["non_improving_epochs"] >= early_stopping_patience
                if should_stop:
                    early_state["stopped_epoch"] = epoch
                    restore_best()
                metrics["early_stopping"] = {
                    "best_epoch": early_state["best_epoch"],
                    "best_validation_loss": early_state["best_validation_loss"],
                    "non_improving_epochs": early_state["non_improving_epochs"],
                    "stopped_epoch": early_state["stopped_epoch"],
                }
        publish()
        resume = None
        if should_stop or stop_at_budget:
            break
    restore_best()


def run(snapshot: Path, output: Path, job_id: str) -> None:
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from nnmodelling_runtime import GraphModule, build_wheel, load_dataset

    training = json.loads((snapshot / "training.json").read_text(encoding="utf-8"))
    random.seed(int(training["seed"]))
    np.random.seed(int(training["seed"]) % (2**32))
    torch.manual_seed(int(training["seed"]))
    selected_device = training.get("device", os.environ.get("NNMODELLING_WORKER_DEVICE", "cpu"))
    if selected_device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported training device '{selected_device}'.")
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA training was requested, but CUDA is unavailable in this worker.")
    if selected_device == "cuda":
        torch.cuda.manual_seed_all(int(training["seed"]))
    precision = training.get("precision", "float32")
    if precision not in {"float32", "bfloat16"}:
        raise ValueError(f"Unsupported training precision '{precision}'.")
    device = torch.device(selected_device)
    previous_cache_root = os.environ.get("NNMODELLING_DATASET_CACHE_ROOT")
    cache_root = (output.resolve() / "dataset-cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    project = Path(tempfile.mkdtemp(prefix="nnmodelling-project-"))
    os.environ["NNMODELLING_DATASET_CACHE_ROOT"] = str(cache_root)
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
        model.to(device=device, dtype=torch.float32)
        from .optimizers import build_optimizer

        optimizer = build_optimizer(
            model, str(training.get("optimizer", "adam")), float(training["learning_rate"]),
            float(training.get("weight_decay", 0.0)),
        )
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
            device=device, precision=precision,
            gradient_clip=training.get("gradient_clip"),
            warmup_steps=int(training.get("warmup_steps", 0)),
            max_steps=int(training.get("max_steps", 0)),
            gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
            early_stopping_patience=int(training.get("early_stopping_patience", 0)),
            early_stopping_min_delta=training.get("early_stopping_min_delta", 0.0),
        )
        test_loss = _epoch(
            model, adapter, "test", int(training["batch_size"]), None,
            device=device, precision=precision,
        )
        metrics["test_loss"] = test_loss
        _write_json(metrics_path, metrics)
        weights = output / "weights.safetensors"
        state = {name: value.detach().cpu().contiguous().clone() for name, value in model.state_dict().items()}
        save_file(state, str(weights))
        build_wheel(project, core_dir, weights, output, job_id)
    finally:
        shutil.rmtree(project, ignore_errors=True)
        if previous_cache_root is None:
            os.environ.pop("NNMODELLING_DATASET_CACHE_ROOT", None)
        else:
            os.environ["NNMODELLING_DATASET_CACHE_ROOT"] = previous_cache_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    run(args.snapshot, args.output, args.job_id)


if __name__ == "__main__":
    main()
