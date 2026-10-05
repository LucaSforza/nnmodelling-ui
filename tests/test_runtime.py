from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

RUNTIME_SRC = Path(__file__).resolve().parents[1] / "python" / "nnmodelling-runtime" / "src"
sys.path.insert(0, str(RUNTIME_SRC))

torch = pytest.importorskip("torch")
from nnmodelling_runtime import Batch, DatasetAdapter, GraphModule, build_wheel, load_dataset


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_scalar_weights(path: Path, value: float):
    header = json.dumps({"node_modules.n0.weight": {"dtype": "F32", "shape": [], "data_offsets": [0, 4]}}).encode()
    header += b" " * ((8 - len(header) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", value))


def _package(core: Path, package_id: str, kind: str, source: str | None = None, parameters=None, outputs=None, objective=None):
    directory = core / package_id.rsplit(".", 1)[-1]
    entrypoints = {"definition": "stereotype.json"}
    if source:
        entrypoints["pytorch"] = {"language": "python", "file": "pytorch.py"}
        (directory / "pytorch.py").parent.mkdir(parents=True, exist_ok=True)
        (directory / "pytorch.py").write_text(source, encoding="utf-8")
    _write_json(directory / "manifest.json", {"schemaVersion": 1, "id": package_id, "version": "1.0.0", "dependencies": {}, "entrypoints": entrypoints})
    definition = {"name": package_id, "kind": kind, "parameters": parameters or {}}
    if outputs is not None:
        definition["outputs"] = outputs
    if objective is not None:
        definition["objective"] = objective
    _write_json(directory / "stereotype.json", definition)


def _node(node_id: str, package_id: str, *, scope="", params=None, boundary=None):
    data = {"scope": scope, "params": params or {}, "package": {"id": package_id, "version": "1.0.0"}}
    if boundary is not None:
        data["boundaryHandle"] = boundary
    return {"id": node_id, "data": data}


def _project(tmp_path: Path, core_packages: list[tuple], nodes, edges, *, dataset=False):
    root = tmp_path / "project"
    root.mkdir()
    core = tmp_path / "core"
    for package in core_packages:
        _package(core, *package)
    manifest = {"schemaVersion": 2, "id": "fixture", "version": "1.0.0", "customPackages": [], "customDatasets": [], "activeDataset": None}
    if dataset:
        (root / "datasets" / "demo").mkdir(parents=True)
        manifest["customDatasets"] = [{"id": "demo", "version": "1.0.0", "path": "datasets/demo"}]
        manifest["activeDataset"] = {"id": "demo", "version": "1.0.0"}
        _write_json(root / "datasets" / "demo" / "manifest.json", {"schemaVersion": 1, "id": "demo", "version": "1.0.0", "entrypoints": {"definition": "dataset.json", "python": {"language": "python", "file": "dataset.py"}}, "inferenceAssets": ["datasets/demo/vocab.json", "__init__.py"]})
        (root / "datasets" / "demo" / "pyproject.toml").write_text(
            '[project]\nname="demo-adapter"\nversion="1.0.0"\ndependencies=["nnmodelling-runtime >=0.1.0", "requests>=2.31"]\n', encoding="utf-8")
        _write_json(root / "datasets" / "demo" / "dataset.json", {"inputs": [{"name": "features"}], "targets": []})
        (root / "datasets" / "demo" / "dataset_helper.py").write_text(
            "from nnmodelling_runtime import DatasetAdapter\nclass BaseAdapter(DatasetAdapter): pass\n", encoding="utf-8")
        (root / "datasets" / "demo" / "dataset.py").write_text(
            "import torch\nfrom .dataset_helper import BaseAdapter\n"
            "class Dataset(BaseAdapter):\n"
            " def tokenize(self, value): return torch.as_tensor(value, dtype=torch.float32)\n"
            " def untokenize(self, tensor): return tensor\n"
            " def load(self, split, batch_size): return iter(())\n", encoding="utf-8")
        (root / "datasets" / "demo" / "vocab.json").write_text("{}", encoding="utf-8")
        (root / "datasets" / "demo" / "train.csv").write_text("secret training data", encoding="utf-8")
        (root / "__init__.py").write_text("snapshot asset, not wheel package code", encoding="utf-8")
    _write_json(root / "model.json", {"manifest": manifest, "nodes": nodes, "edges": edges})
    return root, core


def _linear_source():
    return '''import torch
from torch import nn
from stereotype_runtime.pytorch import BuildContext, NoServices
class Scale(nn.Module):
    def __init__(self, value):
        super().__init__(); self.weight = nn.Parameter(torch.tensor(float(value)))
    def forward(self, x): return x * self.weight
def build(parameters, context, services): return Scale(parameters["value"])
'''


def _input_package():
    return ("core.input", "input")


def _output_package():
    return ("core.output", "output")


def _loss_output_package():
    return ("core.loss-output", "loss-output")


def test_branches_join_order_multiple_handles_and_target_closure(tmp_path):
    pair = '''import torch
from torch import nn
from stereotype_runtime.pytorch import BuildContext, NoServices
class Pair(nn.Module):
 def forward(self, x): return {"left": x + 1, "right": x + 2}
def build(parameters, context, services): return Pair()
'''
    ordered = '''import torch
from torch import nn
from stereotype_runtime.pytorch import BuildContext, NoServices
class Ordered(nn.Module):
 def forward(self, first, second): return first * 10 + second
def build(parameters, context, services): return Ordered()
'''
    loss = '''import torch
from torch import nn
from stereotype_runtime.pytorch import BuildContext, NoServices
class Loss(nn.Module):
 def forward(self, prediction, target): return ((prediction - target) ** 2).mean()
def build(parameters, context, services): return Loss()
'''
    core_packages = [
        _input_package(), _output_package(), _loss_output_package(),
        ("test.pair", "layer", pair, None, [{"id": "left", "type": "output"}, {"id": "right", "type": "loss"}]),
        ("test.join", "join", ordered),
        ("test.loss", "loss", loss, None, [{"id": "loss", "type": "loss"}], {"externalInputs": [{"name": "target", "source": "batch.targets.target"}]}),
    ]
    nodes = [
        _node("x", "core.input", params={"binding": "features"}),
        _node("pair", "test.pair"), _node("join", "test.join"),
        _node("pred", "core.output"), _node("loss", "test.loss"), _node("lossout", "core.loss-output"),
    ]
    edges = [
        {"id": "a", "source": "x", "target": "pair", "sourceHandle": "out", "targetHandle": "in"},
        {"id": "b", "source": "pair", "sourceHandle": "right", "target": "join", "targetHandle": "in-2"},
        {"id": "c", "source": "pair", "sourceHandle": "left", "target": "join", "targetHandle": "in-1"},
        {"id": "d", "source": "join", "target": "pred", "sourceHandle": "out", "targetHandle": "in"},
        {"id": "e", "source": "join", "target": "loss", "sourceHandle": "out", "targetHandle": "in"},
        {"id": "f", "source": "loss", "target": "lossout", "sourceHandle": "loss", "targetHandle": "in"},
    ]
    root, core = _project(tmp_path, core_packages, nodes, edges)
    graph = GraphModule(root, core)
    x = torch.tensor([2.0])
    assert graph(x)["prediction"].item() == pytest.approx(34.0)
    with pytest.raises(ValueError, match="target"):
        graph(x, include_loss=True)
    result = graph(x, {"target": torch.tensor([30.0])}, include_loss=True)
    assert result["loss"].ndim == 0
    assert result["loss"].item() == pytest.approx(16.0)


def test_repeat_builds_independent_registered_subflow_modules(tmp_path):
    repeat_source = '''from torch import nn
from stereotype_runtime.pytorch import BuildContext, SubflowServices
class Repeat(nn.Module):
 def __init__(self, items): super().__init__(); self.items = nn.ModuleList(items)
 def forward(self, x):
  for item in self.items: x = item(x)
  return x
def build(parameters, context, services): return Repeat([services.build_subflow() for _ in range(parameters["times"])])
'''
    proxy_source = '''from stereotype_runtime.pytorch import BuildContext, SubflowServices
def build(parameters, context, services): return services.build_subflow()
'''
    core_packages = [
        _input_package(), _output_package(),
        ("test.repeat", "subflow", repeat_source, {"times": {"type": "integer", "minimum": 1, "default": 2}}),
        ("test.proxy", "subflow", proxy_source),
        ("test.scale", "layer", _linear_source(), {"value": {"type": "real", "default": 2.0}}, [{"id": "out", "type": "output"}]),
    ]
    nodes = [
        _node("x", "core.input", params={"binding": "x"}), _node("repeat", "test.repeat", params={"times": 2}),
        _node("pred", "core.output"), _node("nested-in", "core.input", scope="repeat", params={"binding": ""}),
        _node("inner", "test.proxy", scope="repeat"), _node("nested-out", "core.output", scope="repeat", boundary="out"),
        _node("deep-in", "core.input", scope="inner", params={"binding": ""}),
        _node("scale", "test.scale", scope="inner", params={"value": 2}),
        _node("deep-out", "core.output", scope="inner", boundary="out"),
    ]
    edges = [
        {"id": "a", "source": "x", "target": "repeat"}, {"id": "b", "source": "repeat", "target": "pred"},
        {"id": "c", "source": "nested-in", "target": "inner"},
        {"id": "d", "source": "inner", "sourceHandle": "out", "target": "nested-out"},
        {"id": "e", "source": "deep-in", "target": "scale"},
        {"id": "f", "source": "scale", "sourceHandle": "out", "target": "deep-out"},
    ]
    root, core = _project(tmp_path, core_packages, nodes, edges)
    graph = GraphModule(root, core)
    parameters = list(graph.parameters())
    assert len(parameters) == 2
    assert parameters[0].data_ptr() != parameters[1].data_ptr()
    output = graph(torch.tensor([3.0]))["prediction"]
    output.sum().backward()
    assert all(parameter.grad is not None for parameter in parameters)
    assert output.item() == pytest.approx(12.0)


def test_horizontal_repeat_builds_independent_subflows_and_join(tmp_path):
    horizontal = '''from torch import nn
from torch.func import functional_call, stack_module_state, vmap
from stereotype_runtime.pytorch import BuildContext, StereotypeServices, SubflowServices
class Horizontal(nn.Module):
 def __init__(self, items, join):
  super().__init__(); self.count=len(items); self.join=join
  params,buffers=stack_module_state(items); object.__setattr__(self,"base",items[0].to("meta"))
  self.param_names=tuple(params); self.buffer_names=tuple(buffers)
  for i,value in enumerate(params.values()): self.register_parameter(f"p{i}",nn.Parameter(value))
  for i,value in enumerate(buffers.values()): self.register_buffer(f"b{i}",value)
 def forward(self,x):
  params={name:getattr(self,f"p{i}") for i,name in enumerate(self.param_names)}
  buffers={name:getattr(self,f"b{i}") for i,name in enumerate(self.buffer_names)}
  def one(p,b,value): return functional_call(self.base,(p,b),(value,))
  outputs=vmap(one,in_dims=(0,0,None))(params,buffers,x)
  return self.join(*outputs.unbind(0))
def build(parameters,context,services):
 return Horizontal([services.build_subflow() for _ in range(parameters["times"])], services.build_stereotype(parameters["join"]))
'''
    concat = '''from torch import nn
from stereotype_runtime.pytorch import BuildContext, NoServices
class Concat(nn.Module):
 def __init__(self,dim): super().__init__(); self.dim=dim
 def forward(self,*values):
  import torch
  return torch.cat(values,dim=self.dim)
def build(parameters,context,services): return Concat(parameters["dim"])
'''
    repeat_parameters = {
        "times": {"type": "integer", "minimum": 2},
        "join": {"type": "stereotype", "kind": "join", "default": {"id": "test.concat", "version": "^1.0.0", "parameters": {"dim": -1}}},
    }
    core_packages = [
        _input_package(), _output_package(),
        ("test.horizontal", "subflow", horizontal, repeat_parameters),
        ("test.scale", "layer", _linear_source(), {"value": {"type": "real", "default": 2.0}}, [{"id": "out", "type": "output"}]),
        ("test.concat", "join", concat, {"dim": {"type": "integer", "default": -1}}),
    ]
    nodes = [
        _node("x", "core.input", params={"binding": "x"}),
        _node("horizontal", "test.horizontal", params={"times": 2}),
        _node("pred", "core.output"),
        _node("nested-in", "core.input", scope="horizontal", params={"binding": "ignored"}),
        _node("scale", "test.scale", scope="horizontal", params={"value": 3}),
        _node("nested-out", "core.output", scope="horizontal", boundary="out"),
    ]
    edges = [
        {"id": "a", "source": "x", "target": "horizontal"}, {"id": "b", "source": "horizontal", "target": "pred"},
        {"id": "c", "source": "nested-in", "target": "scale"},
        {"id": "d", "source": "scale", "sourceHandle": "out", "target": "nested-out"},
    ]
    root, core = _project(tmp_path, core_packages, nodes, edges, dataset=True)
    graph = GraphModule(root, core)
    assert len(list(graph.parameters())) == 1
    output = graph(torch.tensor([2.0]))["prediction"]
    assert output.tolist() == pytest.approx([6.0, 6.0])
    output.sum().backward()
    assert graph.node_modules[graph._node_modules["horizontal"]].p0.grad.tolist() == pytest.approx([2.0, 2.0])

    from safetensors.torch import save_file
    import zipfile
    weights = tmp_path / "weights.safetensors"
    save_file({name: value.detach().cpu().contiguous() for name, value in graph.state_dict().items()}, weights)
    wheel = build_wheel(root, core, weights, tmp_path / "wheel", "reference-join")
    with zipfile.ZipFile(wheel) as archive:
        assert "nnmodel_reference_join/_snapshot/core/concat/manifest.json" in archive.namelist()


def test_horizontal_repeat_eval_mode_reaches_unregistered_subflow_template(tmp_path):
    dropout = '''import torch
from torch import nn
from stereotype_runtime.pytorch import BuildContext, NoServices
class Drop(nn.Module):
 def __init__(self): super().__init__(); self.weight=nn.Parameter(torch.ones(())); self.dropout=nn.Dropout(p=0.75)
 def forward(self,x): return self.dropout(x)*self.weight
def build(parameters,context,services): return Drop()
'''
    root, _ = _project(
        tmp_path,
        [_input_package(), _output_package()],
        [
            _node("x", "core.input", params={"binding": "x"}),
            _node("horizontal", "core.horizontal-repeat", params={"times": 2}),
            _node("pred", "core.output"),
            _node("inner", "core.input", scope="horizontal"),
            _node("drop", "test.dropout", scope="horizontal"),
            _node("inner-out", "core.output", scope="horizontal", boundary="out"),
        ],
        [
            {"id": "a", "source": "x", "target": "horizontal"},
            {"id": "b", "source": "horizontal", "target": "pred"},
            {"id": "c", "source": "inner", "target": "drop"},
            {"id": "d", "source": "drop", "target": "inner-out"},
        ],
    )
    package_root = root / "packages"
    _package(package_root, "test.dropout", "layer", dropout)
    model = json.loads((root / "model.json").read_text(encoding="utf-8"))
    model["manifest"]["customPackages"] = [{"id": "test.dropout", "version": "1.0.0", "path": "packages/dropout"}]
    for node in model["nodes"]:
        if node["data"]["package"]["id"].startswith("core."):
            node["data"]["package"]["version"] = "0.1.0"
    _write_json(root / "model.json", model)

    repository_core = Path(__file__).resolve().parents[1] / "stereotype-packages" / "core"
    graph = GraphModule(root, repository_core).eval()
    result = graph(torch.ones((8, 1)))["prediction"]
    torch.testing.assert_close(result, torch.ones((8, 2)))


def test_subflow_maps_prediction_and_loss_outputs_with_target_closure(tmp_path):
    proxy = '''from torch import nn
from stereotype_runtime.pytorch import BuildContext, SubflowServices
def build(parameters,context,services): return services.build_subflow()
'''
    loss = '''from torch import nn
from stereotype_runtime.pytorch import BuildContext, NoServices
class Loss(nn.Module):
 def forward(self,prediction,target): return ((prediction-target)**2).mean()
def build(parameters,context,services): return Loss()
'''
    core_packages = [
        _input_package(), _output_package(), _loss_output_package(),
        ("test.proxy", "subflow", proxy, None, None, None),
        ("test.scale", "layer", _linear_source(), {"value": {"type": "number", "default": 2.0}}, [{"id": "out", "type": "output"}]),
        ("test.loss", "loss", loss, None, [{"id": "loss", "type": "loss"}], {"externalInputs": [{"name": "target", "source": "batch.targets.target"}]}),
    ]
    proxy_definition = {"name": "proxy", "kind": "subflow", "parameters": {}, "outputs": [{"id": "prediction", "type": "output"}, {"id": "objective", "type": "loss"}]}
    nodes = [
        _node("x", "core.input", params={"binding": "x"}), _node("sf", "test.proxy"),
        _node("pred", "core.output"), _node("lossout", "core.loss-output"),
        _node("inner-input", "core.input", scope="sf", params={"binding": "ignored"}),
        _node("scale", "test.scale", scope="sf", params={"value": 2}),
        _node("inner-output", "core.output", scope="sf", boundary="prediction"),
        _node("inner-loss", "test.loss", scope="sf"),
        _node("inner-loss-output", "core.loss-output", scope="sf", boundary="objective"),
    ]
    edges = [
        {"id": "a", "source": "x", "target": "sf"},
        {"id": "b", "source": "sf", "sourceHandle": "prediction", "target": "pred"},
        {"id": "c", "source": "sf", "sourceHandle": "objective", "target": "lossout"},
        {"id": "d", "source": "inner-input", "target": "scale"},
        {"id": "e", "source": "scale", "sourceHandle": "out", "target": "inner-output"},
        {"id": "f", "source": "scale", "sourceHandle": "out", "target": "inner-loss"},
        {"id": "g", "source": "inner-loss", "sourceHandle": "loss", "target": "inner-loss-output"},
    ]
    root, core = _project(tmp_path, core_packages, nodes, edges)
    proxy_path = core / "proxy" / "stereotype.json"
    _write_json(proxy_path, proxy_definition)
    graph = GraphModule(root, core)
    x = torch.tensor([3.0])
    assert graph(x)["prediction"].item() == pytest.approx(6.0)
    train = graph(x, {"target": torch.tensor([4.0])}, include_loss=True)
    assert train["loss"].ndim == 0
    assert train["loss"].item() == pytest.approx(4.0)


def test_bundled_mnist_graph_executes_prediction_and_training_objective():
    repository = Path(__file__).resolve().parents[1]
    graph = GraphModule(repository / "examples" / "models" / "mnist-mlp", repository / "stereotype-packages" / "core")
    images = torch.randn(3, 1, 28, 28)
    prediction = graph({"image": images})["prediction"]
    assert prediction.shape == (3, 10)
    trained = graph({"image": images}, {"target": torch.tensor([1, 2, 3])}, include_loss=True)
    assert trained["loss"].ndim == 0
    trained["loss"].backward()
    assert any(parameter.grad is not None for parameter in graph.parameters())


def test_build_context_input_arity_counts_edges_and_external_inputs(tmp_path):
    inspect_arity = '''from torch import nn
from stereotype_runtime.pytorch import BuildContext, NoServices
class Arity(nn.Module):
 def __init__(self, value): super().__init__(); self.value=value
 def forward(self, *inputs): return inputs[0] * self.value
def build(parameters, context, services):
 assert context["inputs"] == parameters["expected"]
 return Arity(parameters["value"])
'''
    objective = {"externalInputs": [{"name": "target", "source": "batch.targets.target"}]}
    packages = [
        _input_package(), _output_package(), _loss_output_package(),
        ("test.arity", "layer", inspect_arity, {"expected": {"type": "integer"}, "value": {"type": "number", "default": 1}}, [{"id": "out", "type": "output"}]),
        ("test.loss", "loss", '''from torch import nn
from stereotype_runtime.pytorch import BuildContext, NoServices
class Loss(nn.Module):
 def forward(self, prediction, target): return ((prediction-target)**2).mean()
def build(parameters, context, services): return Loss()
''', None, [{"id": "loss", "type": "loss"}], objective),
    ]
    nodes = [
        _node("x", "core.input", params={"binding": "x"}),
        _node("branch-a", "test.arity", params={"expected": 1}),
        _node("branch-b", "test.arity", params={"expected": 1}),
        _node("join", "test.arity", params={"expected": 2, "value": 2}),
        _node("prediction", "core.output"), _node("loss", "test.loss"),
        _node("loss-output", "core.loss-output"),
    ]
    edges = [
        {"id": "a", "source": "x", "target": "branch-a"},
        {"id": "b", "source": "x", "target": "branch-b"},
        {"id": "c", "source": "branch-a", "target": "join", "targetHandle": "in-1"},
        {"id": "d", "source": "branch-b", "target": "join", "targetHandle": "in-2"},
        {"id": "e", "source": "join", "target": "prediction"},
        {"id": "f", "source": "join", "target": "loss"},
        {"id": "g", "source": "loss", "sourceHandle": "loss", "target": "loss-output"},
    ]
    root, core = _project(tmp_path, packages, nodes, edges)
    graph = GraphModule(root, core)
    result = graph(torch.tensor([2.0]), {"target": torch.tensor([4.0])}, include_loss=True)
    assert result["prediction"].item() == pytest.approx(4.0)
    assert result["loss"].item() == pytest.approx(0.0)


def test_dataset_adapter_and_standalone_wheels_support_weight_override_and_isolation(tmp_path):
    core_packages = [
        _input_package(), _output_package(),
        ("test.scale", "layer", _linear_source(), {"value": {"type": "real", "default": 2.0}}, [{"id": "out", "type": "output"}]),
    ]
    nodes = [_node("x", "core.input", params={"binding": "features"}), _node("scale", "test.scale"), _node("pred", "core.output")]
    edges = [{"id": "a", "source": "x", "target": "scale"}, {"id": "b", "source": "scale", "sourceHandle": "out", "target": "pred"}]
    root, core = _project(tmp_path, core_packages, nodes, edges, dataset=True)
    adapter = load_dataset(root)
    assert isinstance(adapter, DatasetAdapter)
    assert list(adapter.load("train", 4)) == []
    graph = GraphModule(root, core)
    weights = tmp_path / "weights.safetensors"
    _write_scalar_weights(weights, 2.0)
    alternate = tmp_path / "alternate.safetensors"
    _write_scalar_weights(alternate, 5.0)
    wheels = [build_wheel(root, core, weights, tmp_path / f"out-{index}", f"job-{index}") for index in range(2)]
    import zipfile
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            assert any(name.endswith("/weights.safetensors") for name in names)
            assert any(name.endswith("/_snapshot/project/datasets/demo/vocab.json") for name in names)
            asset_name = next(name for name in names if name.endswith("/_snapshot/project/__init__.py"))
            assert archive.read(asset_name) == b"snapshot asset, not wheel package code"
            assert not any(name.endswith("/train.csv") for name in names)
            metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
            metadata = archive.read(metadata_name).decode()
            assert "Requires-Dist: requests>=2.31" in metadata
            assert "Requires-Dist: nnmodelling-runtime" not in metadata
    targets = []
    for index, wheel in enumerate(wheels):
        target = tmp_path / f"install-{index}"
        install = subprocess.run(["uv", "pip", "install", "--python", sys.executable, "--target", str(target), "--no-deps", str(wheel)], capture_output=True, text=True)
        assert install.returncode == 0, install.stderr
        targets.append(target)
    script = '''import importlib, sys, torch
sys.path = [p for p in sys.path if "/python/nnmodelling-runtime/src" not in p]
sys.path[:0] = sys.argv[1:3]
assert importlib.util.find_spec("nnmodelling_runtime") is None
first = importlib.import_module("nnmodel_job_0").Model()
second = importlib.import_module("nnmodel_job_1").Model(weights_path=sys.argv[3])
assert "nnmodelling_runtime" not in sys.modules
x = torch.tensor([3.0])
print(first.infer(x).item(), second.inference(x).item())
'''
    result = subprocess.run([sys.executable, "-c", script, str(targets[0]), str(targets[1]), str(alternate)], cwd=tmp_path, check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "6.0 15.0"
