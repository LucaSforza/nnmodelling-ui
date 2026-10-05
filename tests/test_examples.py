from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import sys
import zipfile

import pytest

torch = pytest.importorskip("torch")

from nnmodelling_runtime import GraphModule
from nnmodelling_runtime import Batch, build_wheel, load_dataset


REPOSITORY = Path(__file__).resolve().parents[1]
CORE = REPOSITORY / "stereotype-packages" / "core"
EXAMPLES = (
    "local-training",
    "mnist-mlp",
    "rnn-sine",
    "tiny-decoder-llm",
    "mnist-vae",
)


def test_tiny_decoder_runs_training_and_causal_inference():
    graph = GraphModule(REPOSITORY / "examples" / "models" / "tiny-decoder-llm", CORE)
    tokens = torch.randint(0, 65, (2, 12))
    prediction = graph(tokens)["prediction"]
    assert prediction.shape == (2, 12, 65)
    target = torch.randint(0, 65, (2, 12))
    loss = graph(tokens, {"target": target}, include_loss=True)["loss"]
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in graph.parameters())

    graph.eval()
    first = torch.tensor([[4, 9, 12, 1, 2]])
    changed_suffix = torch.tensor([[4, 9, 12, 50, 51]])
    with torch.inference_mode():
        first_result = graph(first)["prediction"]
        second_result = graph(changed_suffix)["prediction"]
    torch.testing.assert_close(first_result[:, :3], second_result[:, :3])


def test_tiny_decoder_repeat_and_attention_parameters_are_independent():
    graph = GraphModule(REPOSITORY / "examples" / "models" / "tiny-decoder-llm", CORE)
    repeat = graph.node_modules[graph._node_modules["blocks"]]
    assert len(repeat.subflows) == 2
    first = repeat.subflows[0].node_modules["n4"].weight
    second = repeat.subflows[1].node_modules["n4"].weight
    assert first.data_ptr() != second.data_ptr()
    original_second = second.detach().clone()
    with torch.no_grad():
        first.add_(1)
    torch.testing.assert_close(second, original_second)

    heads = repeat.subflows[0].node_modules["n1"]
    stacked = [parameter for parameter in heads.parameters() if parameter.ndim >= 3]
    assert stacked
    parameter = next((value for value in stacked if torch.unique(value.detach(), dim=0).shape[0] == 4), None)
    assert parameter is not None
    other_heads = parameter.detach()[1:].clone()
    with torch.no_grad():
        parameter[0].add_(1)
    torch.testing.assert_close(parameter.detach()[1:], other_heads)


def test_bundled_examples_train_and_export_isolated_inference_wheels(tmp_path):
    from safetensors.torch import save_file

    installed = []
    for index, name in enumerate(EXAMPLES):
        project = REPOSITORY / "examples" / "models" / name
        adapter = load_dataset(project)
        batch = next(iter(adapter.load("train", batch_size=2)))
        assert isinstance(batch, Batch)
        assert batch.inputs and batch.targets

        graph = GraphModule(project, CORE)
        prediction = graph(batch.inputs)["prediction"]
        assert torch.isfinite(prediction).all()
        loss = graph(batch.inputs, batch.targets, include_loss=True)["loss"]
        assert loss.ndim == 0 and torch.isfinite(loss)
        before = {key: value.detach().clone() for key, value in graph.state_dict().items()}
        optimizer = torch.optim.SGD(graph.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [parameter.grad for parameter in graph.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        optimizer.step()
        assert any(not torch.equal(before[key], value.detach()) for key, value in graph.state_dict().items())

        weights = tmp_path / f"{name}.safetensors"
        save_file({key: value.detach().cpu().contiguous() for key, value in graph.state_dict().items()}, weights)
        wheel = build_wheel(project, CORE, weights, tmp_path / "wheels", f"example-{index}")
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
            package = f"nnmodel_example_{index}"
            assert f"{package}/weights.safetensors" in names
            assert any(f"{package}/_snapshot/core/" in path and path.endswith("/manifest.json") for path in names)
            model = json.loads((project / "model.json").read_text(encoding="utf-8"))
            manifest = model["manifest"]
            dataset_ref = next(ref for ref in manifest["customDatasets"] if ref["id"] == manifest["activeDataset"]["id"] and ref["version"] == manifest["activeDataset"]["version"])
            dataset_manifest = json.loads((project / dataset_ref["path"] / "manifest.json").read_text(encoding="utf-8"))
            for asset in dataset_manifest.get("inferenceAssets", []):
                relative = Path(asset)
                archive_name = f"{package}/_snapshot/project/{relative.as_posix()}"
                assert archive.read(archive_name) == (project / relative).read_bytes()
            forbidden_suffixes = {".csv", ".npy", ".npz", ".pt", ".pth", ".safetensors"}
            dataset_prefix = f"{package}/_snapshot/project/{dataset_ref['path']}/"
            assert not any(path.startswith(dataset_prefix) and (Path(path).suffix in forbidden_suffixes or Path(path).name == "data.json") for path in names)

        target = tmp_path / f"install-{index}"
        result = subprocess.run(
            ["uv", "pip", "install", "--python", sys.executable, "--target", str(target), "--no-deps", str(wheel)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        installed.append(target)

    script = '''import importlib, importlib.util, sys
import numpy as np
import torch
sys.path = [p for p in sys.path if "/python/nnmodelling-runtime/src" not in p]
sys.path[:0] = sys.argv[1:]
assert importlib.util.find_spec("nnmodelling_runtime") is None
samples = [0.25, np.zeros((28, 28), dtype=np.uint8), [0.0] * 32, "To be, or not to be.", np.zeros((28, 28), dtype=np.uint8)]
for index, value in enumerate(samples):
    module = importlib.import_module(f"nnmodel_example_{index}")
    model = module.Model()
    result = model.infer(value)
    assert result is not None
    assert "nnmodelling_runtime" not in sys.modules
    if index == 0: assert np.isfinite(result).all()
    if index == 1: assert isinstance(result, int) and 0 <= result < 10
    if index == 2: assert np.isfinite(result)
    if index == 3: assert isinstance(result, str) and result
    if index == 4: assert result.shape == (28, 28) and np.isfinite(result).all()
'''
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", script, *(str(path) for path in installed)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
