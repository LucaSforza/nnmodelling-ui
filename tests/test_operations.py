from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RUNTIME_SRC = Path(__file__).resolve().parents[1] / "python" / "nnmodelling-runtime" / "src"
sys.path.insert(0, str(RUNTIME_SRC))

torch = pytest.importorskip("torch")
from nnmodelling_runtime import GraphModule, build_wheel


def _json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(root: Path, *, operations=None):
    project, core = root / "project", root / "core"
    project.mkdir(parents=True)
    for package_id, kind, outputs in (("core.input", "input", [{"id": "out", "type": "output"}]),
                                      ("core.output", "output", []), ("core.loss-output", "loss-output", [])):
        folder = core / package_id.split(".")[-1]
        _json(folder / "manifest.json", {"id": package_id, "version": "1.0.0", "entrypoints": {"definition": "stereotype.json"}})
        _json(folder / "stereotype.json", {"kind": kind, "outputs": outputs})
    scale = core / "scale"
    _json(scale / "manifest.json", {"id": "test.scale", "version": "1.0.0", "entrypoints": {"definition": "stereotype.json", "pytorch": {"language": "python", "file": "pytorch.py"}}})
    _json(scale / "stereotype.json", {"kind": "layer", "parameters": {"factor": {"type": "real", "default": 2}}, "inputs": {"in": {}}, "outputs": [{"id": "out", "type": "output"}]})
    (scale / "pytorch.py").write_text(
        "import torch\nfrom torch import nn\n"
        "class Scale(nn.Module):\n"
        " def __init__(self, factor): super().__init__(); self.factor=nn.Parameter(torch.tensor(float(factor)))\n"
        " def forward(self, value): return value * self.factor\n"
        "def build(parameters, context, services): return Scale(parameters['factor'])\n",
        encoding="utf-8",
    )
    dataset = project / "datasets" / "demo"
    _json(dataset / "manifest.json", {"id": "demo", "version": "1.0.0", "entrypoints": {"definition": "dataset.json", "python": {"language": "python", "file": "dataset.py"}}})
    _json(dataset / "dataset.json", {"batch": {"inputs": {"features": {"dtype": "float32", "shape": ["B"]}}, "targets": {}}})
    (dataset / "pyproject.toml").write_text('[project]\nname="demo"\nversion="1.0.0"\ndependencies=[]\n', encoding="utf-8")
    (dataset / "dataset.py").write_text(
        "import torch\nfrom nnmodelling_runtime import DatasetAdapter\n"
        "class Dataset(DatasetAdapter):\n"
        " def tokenize(self, value): return torch.as_tensor(value, dtype=torch.float32).reshape(-1)\n"
        " def untokenize(self, value): return value.tolist()\n"
        " def load(self, split, batch_size): return iter(())\n",
        encoding="utf-8",
    )
    manifest = {"schemaVersion": 2, "id": "op-fixture", "version": "1.0.0", "customPackages": [],
                "customDatasets": [{"id": "demo", "version": "1.0.0", "path": "datasets/demo"}],
                "activeDataset": {"id": "demo", "version": "1.0.0"}}
    if operations is not None:
        manifest["operations"] = operations
    nodes = [
        {"id": "input", "data": {"scope": "", "params": {"binding": "features"}, "package": {"id": "core.input", "version": "1.0.0"}}},
        {"id": "upstream", "data": {"scope": "", "params": {"factor": 5}, "package": {"id": "test.scale", "version": "1.0.0"}}},
        {"id": "decoder", "data": {"scope": "", "params": {"factor": 3}, "package": {"id": "test.scale", "version": "1.0.0"}}},
        {"id": "output", "data": {"scope": "", "params": {}, "package": {"id": "core.output", "version": "1.0.0"}}},
        {"id": "loss", "data": {"scope": "", "params": {}, "package": {"id": "core.loss-output", "version": "1.0.0"}}},
    ]
    edges = [{"id": "a", "source": "input", "sourceHandle": "out", "target": "upstream", "targetHandle": "in"},
             {"id": "b", "source": "upstream", "sourceHandle": "out", "target": "decoder", "targetHandle": "in"},
             {"id": "c", "source": "decoder", "sourceHandle": "out", "target": "output", "targetHandle": "in"}]
    _json(project / "model.json", {"manifest": manifest, "nodes": nodes, "edges": edges})
    return project, core


def _operations():
    return [
        {"name": "encode", "input": {"node": "input", "handle": "out", "codec": "dataset"},
         "output": {"node": "upstream", "handle": "out", "codec": "tensor"}},
        {"name": "decode", "input": {"node": "decoder", "handle": "in", "codec": "tensor"},
         "output": {"node": "decoder", "handle": "out", "codec": "dataset"}},
    ]


def _subflow_fixture(root: Path, *, calls=1):
    project, core = _fixture(root, operations=_operations())
    proxy = core / "proxy"
    _json(proxy / "manifest.json", {"id": "test.proxy", "version": "1.0.0", "entrypoints": {"definition": "stereotype.json", "pytorch": {"language": "python", "file": "pytorch.py"}}})
    _json(proxy / "stereotype.json", {"kind": "subflow", "outputs": [{"id": "out", "type": "output"}]})
    (proxy / "pytorch.py").write_text(
        "from torch import nn\n"
        "class Proxy(nn.Module):\n"
        " def __init__(self, body): super().__init__(); self.body=body\n"
        " def forward(self, value):\n"
        f"  for _ in range({calls}): value=self.body(value)\n"
        "  return value\n"
        "def build(parameters, context, services): return Proxy(services.build_subflow())\n",
        encoding="utf-8",
    )
    model = json.loads((project / "model.json").read_text(encoding="utf-8"))
    decoder = next(node for node in model["nodes"] if node["id"] == "decoder")
    decoder["data"].update(package={"id": "test.proxy", "version": "1.0.0"}, params={})
    for node_id, package_id, params in (("nested-input", "core.input", {}),
                                         ("nested-scale", "test.scale", {"factor": 3}),
                                         ("nested-output", "core.output", {})):
        model["nodes"].append({"id": node_id, "data": {"scope": "decoder", "package": {"id": package_id, "version": "1.0.0"}, "params": params}})
    model["edges"].extend([
        {"id": "nested-a", "source": "nested-input", "sourceHandle": "out", "target": "nested-scale", "targetHandle": "in"},
        {"id": "nested-b", "source": "nested-scale", "sourceHandle": "out", "target": "nested-output", "targetHandle": "in"},
    ])
    _json(project / "model.json", model)
    return project, core


def test_subflow_operation_budget_is_per_call(tmp_path):
    project, core = _subflow_fixture(tmp_path)
    graph = GraphModule(project, core)
    for _ in range(257):
        torch.testing.assert_close(graph.run_operation("decode", torch.tensor([4.0])), torch.tensor([12.0]))
    assert graph.training
    torch.testing.assert_close(graph(torch.tensor([2.0]))["prediction"], torch.tensor([30.0]))


def test_subflow_operation_still_enforces_per_call_budget(tmp_path):
    project, core = _subflow_fixture(tmp_path, calls=257)
    graph = GraphModule(project, core)
    for _ in range(2):
        with pytest.raises(ValueError, match="subflow execution exceeds 256 invocations"):
            graph.run_operation("decode", torch.zeros(1))
        assert graph.training


def test_operation_slices_at_injected_handle_and_keeps_regular_forward(tmp_path):
    project, core = _fixture(tmp_path, operations=_operations())
    graph = GraphModule(project, core)
    injected = graph.run_operation("decode", torch.tensor([4.0]))
    torch.testing.assert_close(injected, torch.tensor([12.0]))
    normal = graph(torch.tensor([2.0]))["prediction"]
    torch.testing.assert_close(normal, torch.tensor([30.0]))


def test_wheel_exposes_codec_methods_and_generic_operation(tmp_path):
    project, core = _fixture(tmp_path, operations=_operations())
    graph = GraphModule(project, core)
    from safetensors.torch import save_file
    weights = tmp_path / "weights.safetensors"
    save_file({key: value.detach().cpu().contiguous() for key, value in graph.state_dict().items()}, weights)
    wheel = build_wheel(project, core, weights, tmp_path / "wheel", "operation-test")
    import subprocess
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = f"{RUNTIME_SRC}:{env.get('PYTHONPATH', '')}"
    code = (
        "import sys,zipfile,tempfile,importlib; "
        f"z=zipfile.ZipFile({str(wheel)!r}); d=tempfile.mkdtemp(); z.extractall(d); sys.path.insert(0,d); "
        "m=importlib.import_module('nnmodel_operation_test').Model(); import torch; "
        "assert m.encode([2]).tolist()==[10.0]; assert m.decode(torch.tensor([4.0]))==[12.0]; "
        "assert m.run_operation('decode', torch.tensor([2.0]))==[6.0]; "
        "assert m.infer([2])==[30.0] and m.inference([2])==[30.0]"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


@pytest.mark.parametrize("operations", [
    [{"name": "infer", "input": {"node": "input", "handle": "out", "codec": "dataset"}, "output": {"node": "upstream", "handle": "out", "codec": "tensor"}}],
    [{"name": "bad", "input": {"node": "missing", "handle": "out", "codec": "tensor"}, "output": {"node": "upstream", "handle": "out", "codec": "tensor"}}],
    [{"name": "bad", "input": {"node": "decoder", "handle": "unknown", "codec": "tensor"}, "output": {"node": "upstream", "handle": "out", "codec": "tensor"}}],
    [{"name": "bad", "input": {"node": "input", "handle": "out", "codec": "python"}, "output": {"node": "upstream", "handle": "out", "codec": "tensor"}}],
])
def test_runtime_rejects_bad_or_stale_operation_metadata(tmp_path, operations):
    project, core = _fixture(tmp_path, operations=operations)
    with pytest.raises(ValueError, match="operation|handle"):
        GraphModule(project, core)


def test_backend_rejects_stale_operation_metadata_before_queueing(tmp_path):
    from fastapi import HTTPException
    from backend.validation import validate_project

    operations = [{**_operations()[0], "input": {"node": "deleted", "handle": "out", "codec": "dataset"}}]
    project_dir, core_dir = _fixture(tmp_path, operations=operations)
    model = json.loads((project_dir / "model.json").read_text(encoding="utf-8"))
    files = {path.relative_to(project_dir).as_posix(): path.read_bytes() for path in project_dir.rglob("*") if path.is_file() and path.name != "model.json"}
    with pytest.raises(HTTPException):
        validate_project(model, files, {folder.name: folder for folder in core_dir.iterdir()})


@pytest.mark.parametrize("role", ["input", "output"])
@pytest.mark.parametrize("codec", [None, 0, True, [], {}, "", "python"])
def test_runtime_and_backend_reject_invalid_operation_codecs(tmp_path, role, codec):
    from fastapi import HTTPException
    from backend.validation import validate_project

    operations = _operations()
    operations[0][role]["codec"] = codec
    project, core = _fixture(tmp_path, operations=operations)
    with pytest.raises(ValueError, match="invalid .* codec"):
        GraphModule(project, core)
    model = json.loads((project / "model.json").read_text(encoding="utf-8"))
    files = {path.relative_to(project).as_posix(): path.read_bytes() for path in project.rglob("*") if path.is_file() and path.name != "model.json"}
    with pytest.raises(HTTPException) as failure:
        validate_project(model, files, {folder.name: folder for folder in core.iterdir()})
    assert failure.value.status_code == 422
