#!/usr/bin/env python3
"""Calibrate a saved miniDeepSeekv4 GraphModule on a real CUDA device.

Pass the editable project snapshot whose graph was composed in the GUI:
    .venv/bin/python tests/deepseek_gpu_probe.py examples/models/miniDeepSeekv4 \
        --seq 512 --batch 1 --steps 1 --precision bfloat16
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def _bindings(graph) -> tuple[list[str], list[str], int]:
    input_names: list[str] = []
    target_names: set[str] = set()
    vocab_size: int | None = None
    for node in graph.nodes.values():
        package_id = node.get("data", {}).get("package", {}).get("id", "")
        parameters = node.get("data", {}).get("params", {})
        if graph._kind(node["id"]) == "input" and not node.get("data", {}).get("scope", ""):
            binding = parameters.get("binding")
            if not isinstance(binding, str) or not binding:
                raise ValueError(f"root input {node['id']!r} has no binding")
            input_names.append(binding)
        definition = graph._definition(node)[2]
        if package_id == "dsv4.embedding":
            parameter_definition = definition.get("parameters", {}).get("vocab_size", {})
            vocab_size = parameters.get("vocab_size", parameter_definition.get("default"))
        for external in definition.get("objective", {}).get("externalInputs", []):
            source = external.get("source", "").split(".")
            if len(source) == 3 and source[:2] == ["batch", "targets"]:
                target_names.add(source[2])

    if len(input_names) != 1:
        raise ValueError(f"expected one bound token input, found {input_names!r}")
    if not isinstance(vocab_size, int) or vocab_size <= 1:
        raise ValueError("the saved graph has no valid dsv4.embedding vocabulary size")
    if not target_names:
        raise ValueError("the saved graph declares no batch target for its objective")
    return input_names, sorted(target_names), vocab_size


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this calibration probe; no CPU fallback is used.")
    if args.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not support bfloat16.")

    project_root = args.project_root.expanduser().resolve(strict=True)
    model_path = project_root / "model.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved project snapshot has no model.json: {model_path}")

    repository_root = Path(__file__).resolve().parents[1]
    if str(repository_root / "python") not in sys.path:
        sys.path.insert(0, str(repository_root / "python"))
    from nnmodelling_runtime.graph import GraphModule

    # Count the saved graph without allocating its roughly 1B parameters on CPU.
    with torch.device("meta"):
        meta_graph = GraphModule(project_root)
    meta_parameter_total = sum(parameter.numel() for parameter in meta_graph.parameters())
    meta_trainable_parameters = sum(
        parameter.numel() for parameter in meta_graph.parameters() if parameter.requires_grad
    )
    del meta_graph

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    with torch.device(device):
        graph = GraphModule(project_root)
    if any(
        node.get("data", {}).get("package", {}).get("id") == "dsv4.decoder"
        for node in graph.nodes.values()
    ):
        raise ValueError("the saved graph still contains the rejected opaque dsv4.decoder resource")
    input_names, target_names, vocab_size = _bindings(graph)
    parameter_total = sum(parameter.numel() for parameter in graph.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in graph.parameters() if parameter.requires_grad
    )
    if (parameter_total, trainable_parameters) != (
        meta_parameter_total,
        meta_trainable_parameters,
    ):
        raise ValueError(
            "CUDA graph parameter count differs from the meta-device count: "
            f"meta=({meta_parameter_total}, {meta_trainable_parameters}), "
            f"cuda=({parameter_total}, {trainable_parameters})"
        )

    from backend.optimizers import build_optimizer

    graph.to(device=device, dtype=torch.float32)
    graph.train()
    optimizer = build_optimizer(graph, "deepseek-muon", learning_rate=1e-4, weight_decay=0.0)

    device_properties = torch.cuda.get_device_properties(device)
    losses: list[float] = []
    step_seconds: list[float] = []
    autocast_enabled = args.precision == "bfloat16"
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        tokens = torch.randint(
            vocab_size, (args.batch, args.seq), device=device, dtype=torch.int64
        )
        inputs = {input_names[0]: tokens}
        targets = {name: tokens for name in target_names}
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if autocast_enabled else nullcontext()
        )
        with autocast:
            result = graph(inputs, targets=targets, include_loss=True)
            loss = result["loss"]
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise FloatingPointError(f"GraphModule produced a non-finite scalar loss: {loss.detach()!r}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(graph.parameters(), max_norm=1.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"GraphModule produced a non-finite gradient norm: {gradient_norm!r}")
        optimizer.step()
        torch.cuda.synchronize(device)
        step_seconds.append(time.perf_counter() - started)
        losses.append(float(loss.detach().float().cpu()))

    return {
        "parameter_total": parameter_total,
        "trainable_parameters": trainable_parameters,
        "loss": losses[-1],
        "losses": losses,
        "step_seconds": step_seconds,
        "mean_step_seconds": sum(step_seconds) / len(step_seconds),
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "device_name": device_properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "precision": args.precision,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "sequence_length": args.seq,
        "batch_size": args.batch,
        "steps": args.steps,
        "project_snapshot": str(project_root),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path, help="saved miniDeepSeekv4 project snapshot")
    parser.add_argument("--seq", "--sequence-length", type=int, choices=(128, 512), default=512)
    parser.add_argument("--batch", "--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, choices=(1, 2), default=1)
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)
    if args.batch < 1:
        parser.error("--batch must be positive")
    return args


def main() -> int:
    try:
        result = run(parse_args())
    except Exception:
        traceback.print_exc()
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
