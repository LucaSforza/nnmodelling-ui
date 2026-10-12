from __future__ import annotations

import base64
import csv
import gc
import hashlib
import json
import os
import struct
import subprocess
import sys
import sysconfig
import weakref
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
            " def untokenize(self, tensor):\n  assert tensor.device.type == 'cpu'\n  return tensor\n"
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


@pytest.mark.parametrize("times", [43, 64])
def test_repeat_43_and_64_keep_independent_trainable_causal_layers(tmp_path, times):
    repeat_source = '''from torch import nn
from stereotype_runtime.pytorch import BuildContext, SubflowServices
class Repeat(nn.Module):
 def __init__(self, items): super().__init__(); self.items=nn.ModuleList(items)
 def forward(self, x):
  for item in self.items: x=item(x)
  return x
def build(parameters, context, services): return Repeat([services.build_subflow() for _ in range(parameters["times"])])
'''
    causal_source = '''import math
import torch
from torch import nn
from stereotype_runtime.pytorch import BuildContext, NoServices
class Causal(nn.Module):
 def __init__(self):
  super().__init__(); self.query=nn.Linear(4,4,bias=False); self.key=nn.Linear(4,4,bias=False); self.value=nn.Linear(4,4,bias=False)
 def forward(self, x):
  scores=self.query(x) @ self.key(x).transpose(-2,-1) / math.sqrt(4)
  mask=torch.ones(scores.shape[-2:], dtype=torch.bool, device=x.device).triu(1)
  weights=scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
  return x + 0.01 * (weights @ self.value(x))
def build(parameters, context, services): return Causal()
'''
    packages = [
        _input_package(), _output_package(),
        ("test.repeat", "subflow", repeat_source, {"times": {"type": "integer", "minimum": 1, "default": times}}),
        ("test.causal", "layer", causal_source, None, [{"id": "out", "type": "output"}]),
    ]
    nodes = [
        _node("x", "core.input", params={"binding": "x"}), _node("repeat", "test.repeat"),
        _node("pred", "core.output"), _node("inner-in", "core.input", scope="repeat"),
        _node("causal", "test.causal", scope="repeat"), _node("inner-out", "core.output", scope="repeat", boundary="out"),
    ]
    edges = [
        {"id": "a", "source": "x", "target": "repeat"}, {"id": "b", "source": "repeat", "target": "pred"},
        {"id": "c", "source": "inner-in", "target": "causal"}, {"id": "d", "source": "causal", "target": "inner-out"},
    ]
    root, core = _project(tmp_path, packages, nodes, edges)
    graph = GraphModule(root, core)
    parameters = list(graph.parameters())
    assert len(parameters) == 3 * times
    assert len({parameter.data_ptr() for parameter in parameters}) == 3 * times
    value = torch.randn((1, 4, 4), requires_grad=True)
    result = graph(value)["prediction"]
    changed = value.detach().clone()
    changed[:, -1] += 10
    torch.testing.assert_close(graph(changed)["prediction"][:, :-1], result.detach()[:, :-1])
    result.sum().backward()
    assert all(parameter.grad is not None for parameter in parameters)
    assert torch.isfinite(result).all()


def test_subflow_runner_budget_replaces_legacy_repeat_cap(tmp_path):
    repeat_source = '''from torch import nn
from stereotype_runtime.pytorch import BuildContext, SubflowServices
class Repeat(nn.Module):
 def __init__(self, items): super().__init__(); self.items=nn.ModuleList(items)
 def forward(self, x):
  for item in self.items: x=item(x)
  return x
def build(parameters, context, services): return Repeat([services.build_subflow() for _ in range(parameters["times"])])
'''
    packages = [
        _input_package(), _output_package(),
        ("test.repeat", "subflow", repeat_source, {"times": {"type": "integer", "minimum": 1}}),
    ]
    nodes = [
        _node("x", "core.input", params={"binding": "x"}), _node("repeat", "test.repeat", params={"times": 257}),
        _node("pred", "core.output"), _node("inner-in", "core.input", scope="repeat"),
        _node("inner-out", "core.output", scope="repeat", boundary="out"),
    ]
    edges = [
        {"id": "a", "source": "x", "target": "repeat"}, {"id": "b", "source": "repeat", "target": "pred"},
        {"id": "c", "source": "inner-in", "target": "inner-out"},
    ]
    (tmp_path / "within").mkdir()
    root, core = _project(tmp_path / "within", packages, nodes, edges)
    graph = GraphModule(root, core)
    assert graph(torch.tensor([2.0]))["prediction"].item() == pytest.approx(2.0)

    nodes[1] = _node("repeat", "test.repeat", params={"times": 4097})
    (tmp_path / "excess").mkdir()
    root, core = _project(tmp_path / "excess", packages, nodes, edges)
    with pytest.raises(ValueError, match="subflow runner construction budget exceeded: 4096"):
        GraphModule(root, core)


def test_recursive_reference_builder_stops_at_construction_budgets(tmp_path, monkeypatch):
    recursive = '''from torch import nn
from stereotype_runtime.pytorch import BuildContext, StereotypeServices
def build(parameters, context, services): return services.build_stereotype(parameters["child"])
'''
    reference = {"id": "test.recursive", "version": "1.0.0", "parameters": {}}
    packages = [
        _input_package(), _output_package(),
        ("test.recursive", "layer", recursive, {"child": {"type": "stereotype", "default": reference}}, [{"id": "out", "type": "output"}]),
    ]
    root, core = _project(
        tmp_path,
        packages,
        [_node("x", "core.input", params={"binding": "x"}), _node("recursive", "test.recursive"), _node("out", "core.output")],
        [{"id": "a", "source": "x", "target": "recursive"}, {"id": "b", "source": "recursive", "target": "out"}],
    )
    with pytest.raises(ValueError, match="construction depth budget exceeded: 32"):
        GraphModule(root, core)
    monkeypatch.setattr(GraphModule, "_MAX_BUILDER_CALLS", 2)
    with pytest.raises(ValueError, match="resource builder call budget exceeded: 2"):
        GraphModule(root, core)


def test_nested_scope_depth_limit_accepts_32_and_rejects_33(tmp_path):
    proxy = '''from torch import nn
from stereotype_runtime.pytorch import BuildContext, SubflowServices
def build(parameters, context, services): return services.build_subflow()
'''
    packages = [_input_package(), _output_package(), ("test.proxy", "subflow", proxy)]

    def nested_project(base: Path, depth: int):
        base.mkdir()
        nodes = [
            _node("root-input", "core.input", params={"binding": "x"}),
            _node("scope-0", "test.proxy"), _node("root-output", "core.output"),
        ]
        edges = [
            {"id": "root-in", "source": "root-input", "target": "scope-0"},
            {"id": "root-out", "source": "scope-0", "target": "root-output"},
        ]
        for index in range(depth):
            scope = f"scope-{index}"
            input_id, output_id = f"input-{index}", f"output-{index}"
            nodes.extend([
                _node(input_id, "core.input", scope=scope),
                _node(output_id, "core.output", scope=scope),
            ])
            if index + 1 < depth:
                child_id = f"scope-{index + 1}"
                nodes.append(_node(child_id, "test.proxy", scope=scope))
                edges.extend([
                    {"id": f"inner-in-{index}", "source": input_id, "target": child_id},
                    {"id": f"inner-out-{index}", "source": child_id, "target": output_id},
                ])
            else:
                edges.append({"id": f"inner-{index}", "source": input_id, "target": output_id})
        return _project(base, packages, nodes, edges)

    root, core = nested_project(tmp_path / "depth-32", 32)
    GraphModule(root, core)
    root, core = nested_project(tmp_path / "depth-33", 33)
    with pytest.raises(ValueError, match="construction depth budget exceeded: 32"):
        GraphModule(root, core)


def test_registered_tensor_budget_deduplicates_and_includes_unregistered_modules(tmp_path, monkeypatch):
    tensor_source = '''import torch
from torch import nn
from stereotype_runtime.pytorch import BuildContext, NoServices
class Template(nn.Module):
 def __init__(self): super().__init__(); self.weight=nn.Parameter(torch.ones(2, device="meta"))
class Model(nn.Module):
 def __init__(self):
  super().__init__(); self.weight=nn.Parameter(torch.ones(2)); self.register_buffer("alias", self.weight)
  object.__setattr__(self, "template", Template())
  self.metadata={}; self.metadata["self"]=self.metadata
 def forward(self, x): return x * self.weight
def build(parameters, context, services): return Model()
'''
    root, core = _project(
        tmp_path,
        [_input_package(), _output_package(), ("test.tensor-model", "layer", tensor_source)],
        [_node("x", "core.input", params={"binding": "x"}), _node("model", "test.tensor-model"), _node("out", "core.output")],
        [{"id": "a", "source": "x", "target": "model"}, {"id": "b", "source": "model", "target": "out"}],
    )
    monkeypatch.setattr(GraphModule, "_MAX_REGISTERED_TENSOR_BYTES", 15)
    with pytest.raises(ValueError, match="registered tensor byte budget exceeded: 15 bytes"):
        GraphModule(root, core)

    monkeypatch.setattr(GraphModule, "_MAX_REGISTERED_TENSOR_BYTES", 16)
    graph = GraphModule(root, core)
    assert graph._tensor_bytes == 16
    monkeypatch.setattr(GraphModule, "_MAX_REGISTERED_TENSOR_BYTES", 100)
    baseline = graph._tensor_bytes
    ephemeral = torch.nn.Module()
    ephemeral.register_buffer("buffer", torch.ones(3))
    tensor_ref = weakref.ref(ephemeral.buffer)
    graph._account_module_tensors(ephemeral)
    assert graph._tensor_bytes == baseline + 12
    del ephemeral
    gc.collect()
    assert tensor_ref() is None
    assert graph._tensor_bytes == baseline


def test_tensor_budget_rechecks_referenced_tensor_after_dtype_change(tmp_path, monkeypatch):
    child_source = '''import torch
from torch import nn
from stereotype_runtime.pytorch import BuildContext, NoServices
class Child(nn.Module):
 def __init__(self): super().__init__(); self.weight=nn.Parameter(torch.ones(2))
 def forward(self, x): return x * self.weight
def build(parameters, context, services): return Child()
'''
    parent_source = '''from torch import nn
from stereotype_runtime.pytorch import BuildContext, StereotypeServices
def build(parameters, context, services):
 child=services.build_stereotype(parameters["child"]); child.double(); return child
'''
    child_ref = {"id": "test.tensor-child", "version": "1.0.0", "parameters": {}}
    root, core = _project(
        tmp_path,
        [
            _input_package(), _output_package(),
            ("test.tensor-child", "layer", child_source),
            ("test.tensor-parent", "layer", parent_source, {"child": {"type": "stereotype", "default": child_ref}}),
        ],
        [_node("x", "core.input", params={"binding": "x"}), _node("model", "test.tensor-parent"), _node("out", "core.output")],
        [{"id": "a", "source": "x", "target": "model"}, {"id": "b", "source": "model", "target": "out"}],
    )
    monkeypatch.setattr(GraphModule, "_MAX_REGISTERED_TENSOR_BYTES", 12)
    with pytest.raises(ValueError, match="registered tensor byte budget exceeded: 12 bytes"):
        GraphModule(root, core)


def test_forward_execution_budget_is_fresh_after_failure(tmp_path, monkeypatch):
    packages = [
        _input_package(), _output_package(),
        ("test.scale", "layer", _linear_source(), {"value": {"type": "real", "default": 2.0}}, [{"id": "out", "type": "output"}]),
    ]
    root, core = _project(
        tmp_path, packages,
        [_node("x", "core.input", params={"binding": "x"}), _node("scale", "test.scale"), _node("out", "core.output")],
        [{"id": "a", "source": "x", "target": "scale"}, {"id": "b", "source": "scale", "target": "out"}],
    )
    graph = GraphModule(root, core)
    monkeypatch.setattr(GraphModule, "_MAX_EXECUTION_WORK", 2)
    with pytest.raises(ValueError, match="graph execution budget exceeded: 2 work units"):
        graph(torch.tensor([3.0]))
    monkeypatch.setattr(GraphModule, "_MAX_EXECUTION_WORK", 100)
    assert graph(torch.tensor([3.0]))["prediction"].item() == pytest.approx(6.0)


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
        assert "nnm_fixture/_snapshot/core/concat/manifest.json" in archive.namelist()


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


def test_root_batch_inputs_survive_repeated_subflows_and_named_operations(tmp_path):
    repeat = '''from torch import nn
from stereotype_runtime.pytorch import BuildContext, SubflowServices
class Repeat(nn.Module):
 def __init__(self, blocks): super().__init__(); self.blocks=nn.ModuleList(blocks)
 def forward(self, value):
  for block in self.blocks: value=block(value)
  return value
def build(parameters, context, services): return Repeat([services.build_subflow() for _ in range(2)])
'''
    route = '''import torch
from torch import nn
from stereotype_runtime.pytorch import BuildContext, NoServices
class Route(nn.Module):
 def forward(self, value, token_ids):
  assert token_ids.dtype == torch.int64
  return value.float() + token_ids.remainder(7).float()
def build(parameters, context, services): return Route()
'''
    route_objective = {
        "externalInputs": [{"name": "token_ids", "source": "batch.inputs.tokens"}]
    }
    packages = [
        _input_package(),
        _output_package(),
        ("test.repeat", "subflow", repeat),
        ("test.route", "layer", route, None, [{"id": "out", "type": "output"}], route_objective),
    ]
    nodes = [
        _node("tokens", "core.input", params={"binding": "tokens"}),
        _node("repeat", "test.repeat"),
        _node("after", "test.route"),
        _node("prediction", "core.output"),
        _node("block-input", "core.input", scope="repeat", params={"binding": "ignored"}),
        _node("block-route", "test.route", scope="repeat"),
        _node("block-output", "core.output", scope="repeat", boundary="out"),
    ]
    edges = [
        {"id": "root-in", "source": "tokens", "target": "repeat"},
        {"id": "repeat-after", "source": "repeat", "sourceHandle": "out", "target": "after"},
        {"id": "after-out", "source": "after", "sourceHandle": "out", "target": "prediction"},
        {"id": "inner-in", "source": "block-input", "target": "block-route"},
        {"id": "inner-out", "source": "block-route", "sourceHandle": "out", "target": "block-output"},
    ]
    root, core = _project(tmp_path, packages, nodes, edges)
    model_json = json.loads((root / "model.json").read_text(encoding="utf-8"))
    model_json["manifest"]["operations"] = [
        {
            "name": "route_tokens",
            "input": {"node": "tokens", "handle": "out", "codec": "tensor"},
            "output": {"node": "repeat", "handle": "out", "codec": "tensor"},
        },
        {
            "name": "cut_after",
            "input": {"node": "after", "handle": "in", "codec": "tensor"},
            "output": {"node": "after", "handle": "out", "codec": "tensor"},
        },
    ]
    _write_json(root / "model.json", model_json)
    graph = GraphModule(root, core)
    token_ids = torch.tensor([[2, 5]], dtype=torch.int64)

    full = graph({"tokens": token_ids})["prediction"]
    torch.testing.assert_close(full, torch.tensor([[8.0, 20.0]]))
    injected = graph.run_operation("route_tokens", token_ids)
    torch.testing.assert_close(injected, torch.tensor([[6.0, 15.0]]))
    with pytest.raises(ValueError, match="missing objective input 'tokens'"):
        graph.run_operation("cut_after", token_ids.float())
    with pytest.raises(ValueError, match="missing model input binding 'tokens'"):
        graph({"other": token_ids})
    # Failed calls restore root context and do not poison a later model call.
    torch.testing.assert_close(graph({"tokens": token_ids})["prediction"], full)


def test_dataset_adapter_and_standalone_wheels_support_weight_override_and_isolation(tmp_path):
    core_packages = [
        _input_package(), _output_package(),
        ("test.scale", "layer", _linear_source(), {"value": {"type": "real", "default": 2.0}}, [{"id": "out", "type": "output"}]),
    ]
    nodes = [_node("x", "core.input", params={"binding": "features"}), _node("scale", "test.scale"), _node("pred", "core.output")]
    edges = [{"id": "a", "source": "x", "target": "scale"}, {"id": "b", "source": "scale", "sourceHandle": "out", "target": "pred"}]
    root, core = _project(tmp_path, core_packages, nodes, edges, dataset=True)
    model_json = json.loads((root / "model.json").read_text(encoding="utf-8"))
    model_json["manifest"]["operations"] = [
        {
            "name": "logits",
            "input": {"node": "x", "handle": "out", "codec": "tensor"},
            "output": {"node": "scale", "handle": "out", "codec": "tensor"},
        },
        {
            "name": "dataset_logits",
            "input": {"node": "x", "handle": "out", "codec": "dataset"},
            "output": {"node": "scale", "handle": "out", "codec": "dataset"},
        },
    ]
    _write_json(root / "model.json", model_json)
    adapter = load_dataset(root)
    assert isinstance(adapter, DatasetAdapter)
    assert list(adapter.load("train", 4)) == []
    graph = GraphModule(root, core)
    weights = tmp_path / "weights.safetensors"
    _write_scalar_weights(weights, 2.0)
    alternate = tmp_path / "alternate.safetensors"
    _write_scalar_weights(alternate, 5.0)
    wheels = [
        build_wheel(root, core, selected_weights, tmp_path / f"out-{index}", f"job-{index}")
        for index, selected_weights in enumerate((weights, alternate))
    ]
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
        assert wheel.name == "nnm_fixture-0.1.0-py3-none-any.whl"
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            dist_info = "nnm_fixture-0.1.0.dist-info"
            assert f"{dist_info}/METADATA" in names
            assert f"{dist_info}/RECORD" in names
            assert "nnm_fixture/__init__.py" in names
            assert f"nnmodel_job_{index}/__init__.py" in names
            metadata = archive.read(f"{dist_info}/METADATA").decode()
            assert "Name: nnm_fixture\n" in metadata
            records = {row[0]: row[1:] for row in csv.reader(archive.read(f"{dist_info}/RECORD").decode().splitlines())}
            for name, digest_and_size in records.items():
                if name.endswith("/RECORD"):
                    assert digest_and_size == ["", ""]
                    continue
                payload = archive.read(name)
                digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode("ascii")
                assert digest_and_size == [f"sha256={digest}", str(len(payload))]
        target = tmp_path / f"install-{index}"
        install = subprocess.run(["uv", "pip", "install", "--python", sys.executable, "--target", str(target), "--no-deps", str(wheel)], capture_output=True, text=True)
        assert install.returncode == 0, install.stderr
        targets.append(target)
    script = '''import importlib, sys, torch
sys.path = [p for p in sys.path if "/python/nnmodelling-runtime/src" not in p]
sys.path.insert(0, sys.argv[1])
assert importlib.util.find_spec("nnmodelling_runtime") is None
first = importlib.import_module("nnm_fixture").Model()
legacy = importlib.import_module(sys.argv[2]).Model()
assert "nnmodelling_runtime" not in sys.modules
x = torch.tensor([3.0])
assert first.logits(x).item() == float(sys.argv[3])
assert first.dataset_logits([3.0]).item() == float(sys.argv[3])
assert legacy.infer(x).item() == float(sys.argv[3])
if not torch.cuda.is_available():
 try: first.__class__(device="cuda")
 except RuntimeError as error: assert "CUDA is unavailable" in str(error)
 else: raise AssertionError("unavailable CUDA request succeeded")
'''
    for index, target in enumerate(targets):
        result = subprocess.run([sys.executable, "-c", script, str(target), f"nnmodel_job_{index}", str(6 + index * 9)], cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert result.stdout == ""


def test_llm_consumer_replaces_a_same_version_wheel_at_the_stable_import(tmp_path):
    root, core = _project(
        tmp_path,
        [_input_package(), _output_package(),
         ("test.scale", "layer", _linear_source(), {"value": {"type": "real", "default": 2.0}}, [{"id": "out", "type": "output"}])],
        [_node("x", "core.input", params={"binding": "features"}), _node("scale", "test.scale"), _node("pred", "core.output")],
        [{"id": "a", "source": "x", "target": "scale"}, {"id": "b", "source": "scale", "sourceHandle": "out", "target": "pred"}],
        dataset=True,
    )
    model = json.loads((root / "model.json").read_text(encoding="utf-8"))
    model["manifest"]["id"] = "tiny-decoder-llm"
    _write_json(root / "model.json", model)
    first_weights = tmp_path / "first.safetensors"
    second_weights = tmp_path / "second.safetensors"
    _write_scalar_weights(first_weights, 2.0)
    _write_scalar_weights(second_weights, 5.0)
    wheels = [
        build_wheel(root, core, weights, tmp_path / f"consumer-wheel-{index}", f"consumer-job-{index}")
        for index, weights in enumerate((first_weights, second_weights))
    ]

    environment = tmp_path / "consumer-venv"
    subprocess.run(["uv", "venv", "--system-site-packages", str(environment)], check=True, capture_output=True, text=True)
    python = environment / "bin" / "python"
    installer = Path(__file__).resolve().parents[1] / "examples" / "implementation" / "llm" / "install_wheel.py"
    env = {**os.environ, "PYTHONPATH": sysconfig.get_paths()["purelib"]}
    for index, wheel in enumerate(wheels):
        subprocess.run([str(python), str(installer), str(wheel)], check=True, capture_output=True, text=True, env=env)
        result = subprocess.run(
            [str(python), "-c", "from nnm_tiny_decoder_llm import Model; import torch; print(Model().infer(torch.tensor([3.0])).item())"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == ("6.0" if index == 0 else "15.0")


@pytest.mark.parametrize(
    ("project_id", "distribution"),
    [("llm", "nnm_llm"), ("123", "nnm_123"), ("My--Project.42", "nnm_my_project_42")],
)
def test_wheel_distribution_uses_normalized_project_id(tmp_path, project_id, distribution):
    root, core = _project(
        tmp_path,
        [_input_package(), _output_package(),
         ("test.scale", "layer", _linear_source(), {"value": {"type": "real", "default": 2.0}}, [{"id": "out", "type": "output"}])],
        [_node("x", "core.input", params={"binding": "features"}), _node("scale", "test.scale"), _node("pred", "core.output")],
        [{"id": "a", "source": "x", "target": "scale"}, {"id": "b", "source": "scale", "sourceHandle": "out", "target": "pred"}],
        dataset=True,
    )
    model = json.loads((root / "model.json").read_text(encoding="utf-8"))
    model["manifest"]["id"] = project_id
    _write_json(root / "model.json", model)
    weights = tmp_path / "weights.safetensors"
    _write_scalar_weights(weights, 2.0)

    wheel = build_wheel(root, core, weights, tmp_path / "wheels", "job-1")

    assert wheel.name == f"{distribution}-0.1.0-py3-none-any.whl"
    import zipfile
    with zipfile.ZipFile(wheel) as archive:
        assert f"{distribution}-0.1.0.dist-info/METADATA" in archive.namelist()
        metadata = archive.read(f"{distribution}-0.1.0.dist-info/METADATA").decode()
        assert f"Name: {distribution}\n" in metadata
        assert "nnmodel_job_1/__init__.py" in archive.namelist()


@pytest.mark.parametrize("project_id", [None, 7, "", "!!!"])
def test_wheel_rejects_missing_or_unreadable_project_id(tmp_path, project_id):
    root, core = _project(tmp_path, [_input_package(), _output_package()], [], [], dataset=True)
    model = json.loads((root / "model.json").read_text(encoding="utf-8"))
    if project_id is None:
        del model["manifest"]["id"]
    else:
        model["manifest"]["id"] = project_id
    _write_json(root / "model.json", model)
    weights = tmp_path / "weights.safetensors"
    _write_scalar_weights(weights, 2.0)

    with pytest.raises(ValueError, match="project ID"):
        build_wheel(root, core, weights, tmp_path / "wheels", "job-1")
