from __future__ import annotations

import copy
import contextvars
import importlib
import math
import json
import re
import uuid
import weakref
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Mapping

import torch

from ._imports import resource_module

_CURRENT_TARGETS = contextvars.ContextVar("nnmodelling_targets", default={})
_CURRENT_INCLUDE_LOSS = contextvars.ContextVar("nnmodelling_include_loss", default=False)


def _safe_relative(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts or "\\" in path:
        raise ValueError(f"unsafe project resource path: {path!r}")
    return candidate


def _project_path(root: Path, relative: str) -> Path:
    candidate = root / _safe_relative(relative)
    current = root
    for part in _safe_relative(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"project resource path traverses symlink: {relative!r}")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"project resource path escapes root: {relative!r}")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


class _Catalog:
    def __init__(self, project_dir: Path, core_dir: Path | None):
        self.project_dir = project_dir.resolve()
        if core_dir is None:
            repo_core = Path(__file__).resolve().parents[4] / "stereotype-packages" / "core"
            candidates = [self.project_dir / "stereotype-packages" / "core", repo_core]
            core_dir = next((p for p in candidates if p.is_dir()), candidates[-1])
        self.core_dir = Path(core_dir).resolve()
        self.records: dict[str, tuple[Path, dict[str, Any], dict[str, Any]]] = {}
        model = _read_json(self.project_dir / "model.json")
        manifest = model.get("manifest", {})
        refs = manifest.get("customPackages", [])
        custom_ids: set[str] = set()
        for ref in refs:
            if ref["id"] in custom_ids:
                raise ValueError(f"duplicate active package identity {ref['id']!r}")
            custom_ids.add(ref["id"])
            rel = _safe_relative(ref["path"])
            package_dir = _project_path(self.project_dir, ref["path"])
            self._add(package_dir, ref["id"], ref["version"])
        core_index: dict[str, tuple[Path, dict[str, Any]]] = {}
        for package_dir in self.core_dir.iterdir() if self.core_dir.is_dir() else ():
            if package_dir.is_dir() and (package_dir / "manifest.json").is_file():
                package_manifest = _read_json(package_dir / "manifest.json")
                package_id = package_manifest.get("id")
                if package_id in core_index:
                    raise ValueError(f"duplicate core package identity {package_id!r}")
                core_index[package_id] = (package_dir, package_manifest)
        overlap = custom_ids.intersection(core_index)
        if overlap:
            raise ValueError(f"custom packages shadow core package IDs: {sorted(overlap)!r}")
        todo: list[tuple[str, str | None]] = []
        for node in model.get("nodes", []):
            package = node.get("data", {}).get("package", {})
            todo.append((package.get("id"), package.get("version")))
        activated: set[str] = set()
        while todo:
            package_id, constraint = todo.pop()
            if package_id in activated:
                if constraint and not _version_matches(self.records[package_id][1]["version"], constraint):
                    raise ValueError(f"incompatible version constraints for package {package_id!r}")
                continue
            if package_id not in self.records:
                try:
                    directory, manifest = core_index[package_id]
                except KeyError as exc:
                    raise ValueError(f"undeclared stereotype package {package_id!r}") from exc
                self._add(directory, package_id, manifest.get("version"))
            manifest = self.records[package_id][1]
            if constraint and not _version_matches(manifest["version"], constraint):
                raise ValueError(f"package {package_id!r} does not satisfy {constraint!r}")
            activated.add(package_id)
            for dependency_id, constraint in manifest.get("dependencies", {}).items():
                todo.append((dependency_id, constraint))
            for node in model.get("nodes", []):
                node_package = node.get("data", {}).get("package", {})
                if node_package.get("id") != package_id:
                    continue
                params = node.get("data", {}).get("params", {})
                for name, spec in self.records[package_id][2].get("parameters", {}).items():
                    value = params.get(name, spec.get("default") if isinstance(spec, dict) else None)
                    if isinstance(spec, dict) and spec.get("type") == "stereotype" and isinstance(value, dict):
                        todo.append((value.get("id"), value.get("version")))
        self.model = model

    def _add(self, directory: Path, expected_id: str | None, expected_version: str | None) -> None:
        manifest = _read_json(_project_path(directory, "manifest.json"))
        definition_name = manifest.get("entrypoints", {}).get("definition", "stereotype.json")
        definition = _read_json(_project_path(directory, definition_name))
        package_id, version = manifest.get("id"), manifest.get("version")
        if package_id != expected_id or version != expected_version:
            raise ValueError(f"package identity mismatch in {directory}")
        self.records[package_id] = (directory, manifest, definition)

    def resolve(self, package_id: str, version: str | None = None) -> tuple[Path, dict[str, Any], dict[str, Any]]:
        try:
            record = self.records[package_id]
        except KeyError as exc:
            raise ValueError(f"undeclared stereotype package {package_id!r}") from exc
        if version is not None and not _version_matches(record[1].get("version", ""), version):
            raise ValueError(f"package {package_id!r} version does not match {version!r}")
        return record


def _version_matches(version: str, constraint: str) -> bool:
    if constraint in {"", "*"}:
        return True
    if constraint[0] not in "^~":
        return version == constraint
    try:
        wanted = tuple(map(int, constraint[1:].split(".")))
        actual = tuple(map(int, version.split(".")))
    except ValueError as exc:
        raise ValueError(f"unsupported version constraint {constraint!r}") from exc
    if len(wanted) != 3 or len(actual) != 3 or actual < wanted:
        return False
    if constraint.startswith("~"):
        return actual[:2] == wanted[:2]
    return actual[0] == wanted[0] if wanted[0] else actual[:2] == wanted[:2] if wanted[1] else actual == wanted

class _SubflowRunner(torch.nn.Module):
    def __init__(self, graph: "GraphModule", scope: str, modules: Mapping[str, torch.nn.Module], depth: int):
        super().__init__()
        object.__setattr__(self, "_graph_ref", weakref.ref(graph))
        self.scope = scope
        self.node_modules = torch.nn.ModuleDict({f"n{i}": module for i, module in enumerate(modules.values())})
        self._module_keys = {node: f"n{i}" for i, node in enumerate(modules)}
        self.depth = depth

    def __deepcopy__(self, memo):
        graph = self._graph_ref()
        if graph is None:
            raise RuntimeError("subflow graph is no longer available")
        modules = {
            node_id: copy.deepcopy(self.node_modules[key], memo)
            for node_id, key in self._module_keys.items()
        }
        cloned = type(self)(graph, self.scope, modules, self.depth)
        memo[id(self)] = cloned
        return cloned

    def forward(self, value: torch.Tensor, targets: Mapping[str, torch.Tensor] | None = None, include_loss: bool | None = None) -> torch.Tensor | Mapping[str, torch.Tensor]:
        graph = self._graph_ref()
        if graph is None:
            raise RuntimeError("subflow graph is no longer available")
        graph._runtime_invocations += 1
        if graph._runtime_invocations > graph._module_limit:
            raise ValueError("subflow execution exceeds 256 invocations")
        nested_targets = _CURRENT_TARGETS.get() if targets is None else targets
        nested_loss = _CURRENT_INCLUDE_LOSS.get() if include_loss is None else include_loss
        result = graph._evaluate_scope(self.scope, {"": value}, nested_targets, include_loss=nested_loss, module_overrides=self.node_modules, override_keys=self._module_keys, depth=self.depth)
        return next(iter(result.values())) if len(result) == 1 else result


class _Services:
    def __init__(self, ref: StereotypeReference, build_subflow):
        self.reference = ref
        self._build_subflow = build_subflow

    def build_subflow(self) -> torch.nn.Module:
        return self._build_subflow()

    def resolve(self, package_id: str, version: str | None = None) -> StereotypeReference:
        return self._resolve(package_id, version)

    def build_stereotype(self, reference: Any) -> torch.nn.Module:
        return self._build_stereotype(reference)

    _resolve = None


class GraphModule(torch.nn.Module):
    """Generic, handle-keyed executor for a saved NNModelling schema-v2 graph."""

    def __init__(self, project_dir: str | Path, core_dir: str | Path | None = None):
        super().__init__()
        self.project_dir = Path(project_dir).resolve()
        self.catalog = _Catalog(self.project_dir, Path(core_dir) if core_dir else None)
        self.model = self.catalog.model
        self.nodes = {node["id"]: node for node in self.model.get("nodes", [])}
        if len(self.nodes) != len(self.model.get("nodes", [])):
            raise ValueError("duplicate node identity")
        self.edges = self.model.get("edges", [])
        self.node_modules = torch.nn.ModuleDict()
        self._node_modules: dict[str, str] = {}
        self._node_module_objects: dict[str, torch.nn.Module] = {}
        self._subflow_builds = 0
        self._runtime_invocations = 0
        self._module_limit = 256
        self._scope_depths = self._compute_scope_depths()
        self._scopes = defaultdict(list)
        for node in self.nodes.values():
            scope = node.get("data", {}).get("scope", "")
            self._scopes[scope].append(node["id"])
        self._validate_scopes()
        self._build_modules()

    def _compute_scope_depths(self) -> dict[str, int]:
        scopes = {node.get("data", {}).get("scope", "") for node in self.nodes.values()} | {""}
        depths: dict[str, int] = {"": 0}
        for scope in scopes - {""}:
            current, depth, seen = scope, 0, set()
            while current:
                if current in seen:
                    raise ValueError(f"recursive subflow scope containment at {current!r}")
                seen.add(current)
                owner = self.nodes.get(current)
                if owner is None:
                    raise ValueError(f"orphan graph scope {current!r}")
                current = owner.get("data", {}).get("scope", "")
                depth += 1
                if depth > 32:
                    raise ValueError("subflow nesting exceeds maximum depth 32")
            depths[scope] = depth
        return depths

    def _definition(self, node: dict[str, Any]):
        return self.catalog.resolve(node["data"]["package"]["id"], node["data"]["package"].get("version"))

    @staticmethod
    def _reference(manifest, definition, directory):
        abi = importlib.import_module(f"{__package__}._stereotype_runtime.pytorch")
        return abi.StereotypeReference(manifest["id"], manifest["version"], str(directory), manifest, definition)

    def _effective_parameters(self, supplied: dict[str, Any], definition: dict[str, Any]) -> dict[str, Any]:
        defaults = {key: spec["default"] for key, spec in definition.get("parameters", {}).items() if isinstance(spec, dict) and "default" in spec}
        defaults.update(supplied or {})
        return defaults

    def _load_build(self, node_id: str, directory: Path, path: Path):
        module = resource_module("_nnmodel_stereotype_" + uuid.uuid4().hex, directory, path, __package__)
        if not callable(getattr(module, "build", None)):
            raise ValueError(f"Python entrypoint has no build(parameters, context, services): {path}")
        return module

    def _build_modules(self) -> None:
        ordered_nodes = sorted(self.nodes.items(), key=lambda pair: self._scope_depths[pair[1].get("data", {}).get("scope", "")], reverse=True)
        for node_id, node in ordered_nodes:
            directory, manifest, definition = self._definition(node)
            entry = manifest.get("entrypoints", {}).get("pytorch")
            kind = definition.get("kind")
            if not entry or entry.get("language") != "python":
                if kind in {"input", "output", "loss-output"}:
                    continue
                raise ValueError(f"package {manifest['id']!r} has no Python pytorch entrypoint")
            path = _project_path(directory, entry["file"])
            if not path.is_file():
                raise ValueError(f"missing Python entrypoint: {path}")
            module = self._load_build(node_id, directory, path)
            ref = self._reference(manifest, definition, directory)
            scope = node.get("data", {}).get("scope", "")

            def build_subflow(scope=scope, node_id=node_id, definition=definition):
                child_scope = node_id
                if child_scope not in self._scopes and scope:
                    child_scope = scope
                if child_scope not in self._scopes:
                    raise ValueError(f"subflow {node_id!r} has no nested graph scope")
                self._subflow_builds += 1
                if self._subflow_builds > self._module_limit:
                    raise ValueError("subflow module construction exceeds 256 instances")
                child_modules = {
                    child_id: copy.deepcopy(self._node_module_objects[child_id])
                    for child_id in self._scopes[child_scope]
                    if child_id in self._node_module_objects
                }
                return _SubflowRunner(self, child_scope, child_modules, self._scope_depths[child_scope])

            services = _Services(ref, build_subflow)
            services._resolve = self._resolve_reference
            services._build_stereotype = self._build_stereotype
            context = {"input": definition.get("inputs", {}), "output": (definition.get("outputs") or [{}])[0]}
            params = self._effective_parameters(node.get("data", {}).get("params", {}), definition)
            self._validate_parameters(manifest["id"], params, definition)
            if kind == "subflow" and isinstance(params.get("times"), int) and params["times"] > self._module_limit:
                raise ValueError("subflow repeat count exceeds 256")
            built = module.build(params, context, services)
            if not isinstance(built, torch.nn.Module):
                raise TypeError(f"build() for {manifest['id']!r} must return torch.nn.Module")
            self._node_module_objects[node_id] = built
            if not scope:
                key = f"n{len(self.node_modules)}"
                self.node_modules[key] = built
                self._node_modules[node_id] = key

    @staticmethod
    def _validate_parameters(package_id: str, params: dict[str, Any], definition: dict[str, Any]) -> None:
        declarations = definition.get("parameters", {})
        unknown = set(params) - set(declarations)
        if unknown:
            raise ValueError(f"unknown parameters for {package_id!r}: {sorted(unknown)!r}")
        for name, spec in declarations.items():
            if name not in params:
                if isinstance(spec, dict) and ("default" in spec or spec.get("optional", False)):
                    continue
                raise ValueError(f"missing required parameter {name!r} for {package_id!r}")
            if not isinstance(spec, dict) or "type" not in spec:
                continue
            value, type_name = params[name], spec["type"]
            valid = {
                "boolean": lambda: isinstance(value, bool),
                "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
                "real": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
                "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
                "dtype": lambda: isinstance(value, str) and value in {"float16", "bfloat16", "float32", "float64", "int8", "uint8", "int16", "int32", "int64", "bool"},
                "json": lambda: value is None or isinstance(value, (bool, int, float, str, list, dict)),
                "string": lambda: isinstance(value, str),
                "array": lambda: isinstance(value, list),
                "object": lambda: isinstance(value, dict),
                "stereotype": lambda: isinstance(value, dict),
            }.get(type_name)
            if valid is None or not valid():
                raise ValueError(f"invalid {type_name} parameter {name!r} for {package_id!r}")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"parameter {name!r} must be finite for {package_id!r}")
                if ("minimum" in spec and value < spec["minimum"]) or ("maximum" in spec and value > spec["maximum"]):
                    raise ValueError(f"parameter {name!r} is outside its declared range for {package_id!r}")
            choices = spec.get("choices", spec.get("enum"))
            if choices is not None and value not in choices:
                raise ValueError(f"parameter {name!r} is not an allowed value for {package_id!r}")

    def _build_stereotype(self, reference: Any) -> torch.nn.Module:
        if not isinstance(reference, Mapping) or not reference.get("id"):
            raise ValueError("stereotype parameter must be an {id, version, parameters} object")
        directory, manifest, definition = self.catalog.resolve(reference["id"], reference.get("version"))
        entry = manifest.get("entrypoints", {}).get("pytorch", {})
        if entry.get("language") != "python":
            raise ValueError(f"stereotype {reference['id']!r} has no Python entrypoint")
        module = self._load_build("reference", directory, _project_path(directory, entry["file"]))
        params = self._effective_parameters(reference.get("parameters", {}), definition)
        self._validate_parameters(manifest["id"], params, definition)
        ref = self._reference(manifest, definition, directory)
        services = _Services(ref, lambda: (_ for _ in ()).throw(ValueError("referenced join cannot build a subflow")))
        services._resolve = self._resolve_reference
        services._build_stereotype = self._build_stereotype
        built = module.build(params, {"input": {}, "output": (definition.get("outputs") or [{}])[0]}, services)
        if not isinstance(built, torch.nn.Module):
            raise TypeError(f"build() for {manifest['id']!r} must return torch.nn.Module")
        return built

    def _resolve_reference(self, package_id: str, version: str | None = None) -> StereotypeReference:
        directory, manifest, definition = self.catalog.resolve(package_id, version)
        return self._reference(manifest, definition, directory)

    def _validate_scopes(self) -> None:
        for node in self.nodes.values():
            scope = node.get("data", {}).get("scope", "")
            if scope and scope not in self.nodes:
                raise ValueError(f"orphan graph scope {scope!r}")
            if scope and self._kind(scope) != "subflow":
                raise ValueError(f"scope {scope!r} is not owned by a subflow node")
        for edge in self.edges:
            if edge.get("source") not in self.nodes or edge.get("target") not in self.nodes:
                raise ValueError(f"edge {edge.get('id')!r} references a missing node")
            source_scope = self.nodes[edge["source"]].get("data", {}).get("scope", "")
            target_scope = self.nodes[edge["target"]].get("data", {}).get("scope", "")
            if source_scope != target_scope:
                raise ValueError(f"cross-scope edge {edge.get('id')!r}")
            self._input_order(edge.get("targetHandle", "in"))
            source_definition = self._definition(self.nodes[edge["source"]])[2]
            source_outputs = [output.get("id") for output in source_definition.get("outputs", [])]
            source_kind = source_definition.get("kind", "layer")
            if not source_outputs:
                source_outputs = ["loss" if source_kind in {"loss", "loss-calculation"} else "out"]
            if edge.get("sourceHandle", "out") not in source_outputs:
                raise ValueError(f"invalid source handle on edge {edge.get('id')!r}")
        for scope, node_ids in self._scopes.items():
            self._ordered(scope, set(node_ids))

    def _kind(self, node_id: str) -> str:
        return self._definition(self.nodes[node_id])[2].get("kind", "layer")

    def _dependencies(self, node_id: str, scope: str, include_loss: bool) -> list[tuple[str, str, str]]:
        return [
            (e["source"], e.get("sourceHandle", "out"), e.get("targetHandle", "in"))
            for e in self.edges
            if e["target"] == node_id and self.nodes[e["source"]].get("data", {}).get("scope", "") == scope
        ]

    def _terminal_nodes(self, scope: str, include_loss: bool) -> list[str]:
        wanted = {"output"}
        if include_loss:
            wanted.add("loss-output")
        return [node_id for node_id in self._scopes[scope] if self._kind(node_id) in wanted]

    def _input_output_handle(self, node_id: str) -> str:
        outputs = self._definition(self.nodes[node_id])[2].get("outputs", [])
        return outputs[0]["id"] if len(outputs) == 1 else "out"

    def _closure(self, terminals: list[str], scope: str) -> set[str]:
        required: set[str] = set()
        todo = list(terminals)
        while todo:
            node_id = todo.pop()
            if node_id in required:
                continue
            required.add(node_id)
            todo.extend(source for source, _, _ in self._dependencies(node_id, scope, False))
        return required

    def _ordered(self, scope: str, required: set[str]) -> list[str]:
        indegree = {node_id: 0 for node_id in required}
        outgoing: dict[str, list[str]] = defaultdict(list)
        for edge in self.edges:
            a, b = edge["source"], edge["target"]
            if a in required and b in required and self.nodes[a].get("data", {}).get("scope", "") == scope:
                indegree[b] += 1
                outgoing[a].append(b)
        queue = deque(node_id for node_id in required if indegree[node_id] == 0)
        result = []
        while queue:
            node_id = queue.popleft()
            result.append(node_id)
            for target in outgoing[node_id]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        if len(result) != len(required):
            raise ValueError(f"cycle in graph scope {scope!r}")
        return result

    @staticmethod
    def _input_order(handle: str) -> tuple[int, str]:
        if handle == "in":
            return (0, "")
        match = re.fullmatch(r"in-(\d+)", handle)
        if match:
            suffix = int(match.group(1))
            if suffix < 1:
                raise ValueError(f"invalid ordered input handle {handle!r}")
            return (suffix, "")
        raise ValueError(f"unsupported input handle {handle!r}")

    def _evaluate_scope(self, scope: str, bindings: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor], include_loss: bool, module_overrides=None, override_keys=None, depth: int = 0) -> dict[str, torch.Tensor]:
        if depth > 32:
            raise ValueError("subflow execution exceeds depth 32")
        terminals = self._terminal_nodes(scope, include_loss)
        if not terminals:
            raise ValueError(f"scope {scope!r} has no requested output terminal")
        required = self._closure(terminals, scope)
        values: dict[str, dict[str, torch.Tensor]] = {}
        for node_id in self._ordered(scope, required):
            node = self.nodes[node_id]
            kind = self._kind(node_id)
            if kind == "input":
                name = node.get("data", {}).get("params", {}).get("binding", "")
                if scope:
                    if "" not in bindings:
                        raise ValueError(f"nested input {node_id!r} has no inherited value")
                    value = bindings[""]
                else:
                    try:
                        value = bindings[name]
                    except KeyError as exc:
                        raise ValueError(f"missing model input binding {name!r}") from exc
                values[node_id] = {self._input_output_handle(node_id): value}
                continue
            edges = self._dependencies(node_id, scope, include_loss)
            edges.sort(key=lambda item: self._input_order(item[2]))
            incoming = []
            for source, source_handle, _ in edges:
                try:
                    incoming.append(values[source][source_handle])
                except KeyError as exc:
                    raise ValueError(f"missing source handle {source!r}.{source_handle!r}") from exc
            if kind in {"output", "loss-output"}:
                if len(incoming) != 1:
                    raise ValueError(f"terminal {node_id!r} requires exactly one input")
                handle = node.get("data", {}).get("boundaryHandle") or ("loss" if kind == "loss-output" else "out")
                values[node_id] = {handle: incoming[0]}
                continue
            directory, manifest, definition = self._definition(node)
            for external in definition.get("objective", {}).get("externalInputs", []):
                parts = external.get("source", "").split(".")
                if len(parts) != 3 or parts[0] != "batch" or parts[1] != "targets":
                    raise ValueError(f"unsupported objective external input source {external.get('source')!r}")
                if parts[2] not in targets:
                    raise ValueError(f"missing objective target {parts[2]!r}")
                value = targets[parts[2]]
                transform = external.get("transform")
                if transform == "flatten_batch":
                    value = value.reshape(value.shape[0], -1)
                elif transform is not None:
                    raise ValueError(f"unsupported objective external input transform {transform!r}")
                incoming.append(value)
            if module_overrides is not None and node_id in override_keys:
                module = module_overrides[override_keys[node_id]]
            elif node_id in self._node_modules:
                module = self.node_modules[self._node_modules[node_id]]
            else:
                module = self._node_module_objects[node_id]
            result = module(*incoming)
            outputs = definition.get("outputs", [])
            if isinstance(result, Mapping):
                declared = {output["id"]: output.get("type", "output") for output in outputs}
                if not declared:
                    default_handle = "loss" if kind in {"loss", "loss-calculation"} else "out"
                    declared = {default_handle: "loss" if default_handle == "loss" else "output"}
                expected = set(declared)
                missing = expected - set(result)
                allowed_missing = {key for key in missing if declared.get(key) == "loss"} if not include_loss else set()
                if not set(result) <= expected or missing - allowed_missing or any(not isinstance(value, torch.Tensor) for value in result.values()):
                    raise ValueError(f"node {node_id!r} returned outputs that do not match its definition")
                values[node_id] = dict(result)
            else:
                if not isinstance(result, torch.Tensor):
                    raise TypeError(f"node {node_id!r} must return a Tensor or handle-keyed Tensor mapping")
                selected_outputs = [output for output in outputs if include_loss or output.get("type") != "loss"]
                if len(selected_outputs) > 1:
                    raise ValueError(f"node {node_id!r} has multiple declared outputs and must return a mapping")
                handle = selected_outputs[0]["id"] if selected_outputs else (outputs[0]["id"] if len(outputs) == 1 else ("loss" if kind in {"loss", "loss-calculation"} else "out"))
                values[node_id] = {handle: result}
        result: dict[str, torch.Tensor] = {}
        for terminal in terminals:
            kind = self._kind(terminal)
            parent_handle = self.nodes[terminal].get("data", {}).get("boundaryHandle")
            key = (parent_handle or ("loss" if kind == "loss-output" else "out")) if scope else ("loss" if kind == "loss-output" else "prediction")
            incoming = self._dependencies(terminal, scope, include_loss)
            if incoming:
                source, handle, _ = incoming[0]
                result[key] = values[source][handle]
        return result

    def forward(self, inputs: Mapping[str, torch.Tensor] | torch.Tensor, targets: Mapping[str, torch.Tensor] | None = None, include_loss: bool = False) -> dict[str, torch.Tensor]:
        self._runtime_invocations = 0
        targets_token = _CURRENT_TARGETS.set(targets or {})
        loss_token = _CURRENT_INCLUDE_LOSS.set(include_loss)
        try:
            if isinstance(inputs, torch.Tensor):
                inputs = self.single_input_mapping(inputs)
            if not isinstance(inputs, Mapping) or any(not isinstance(value, torch.Tensor) for value in inputs.values()):
                raise TypeError("model inputs must be a tensor or a named tensor mapping")
            if any(not isinstance(value, torch.Tensor) for value in (targets or {}).values()):
                raise TypeError("model targets must be a named tensor mapping")
            return self._evaluate_scope("", inputs, targets or {}, include_loss)
        finally:
            _CURRENT_TARGETS.reset(targets_token)
            _CURRENT_INCLUDE_LOSS.reset(loss_token)

    def single_input_mapping(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        names = []
        for node_id in self._scopes[""]:
            if self._kind(node_id) == "input":
                binding = self.nodes[node_id].get("data", {}).get("params", {}).get("binding", "")
                if not binding:
                    raise ValueError(f"root input {node_id!r} has no binding")
                names.append(binding)
        if len(names) != 1:
            raise ValueError(f"tensor shorthand requires exactly one bound root input, found {len(names)}")
        return {names[0]: value}
