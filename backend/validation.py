from __future__ import annotations

import json
import keyword
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import HTTPException

from . import config

_INVALID_SEGMENTS = {"", ".", "..", ".venv", "venv", "__pycache__", ".git", "node_modules"}


def safe_relative_path(value: str) -> PurePosixPath:
    if not value or "\\" in value or "\x00" in value:
        raise HTTPException(422, "File paths must be non-empty project-relative POSIX paths.")
    if any(part in _INVALID_SEGMENTS for part in value.split("/")):
        raise HTTPException(422, f"Unsafe project file path: {value!r}.")
    path = PurePosixPath(value)
    if path.is_absolute() or ":" in path.parts[0]:
        raise HTTPException(422, f"Unsafe project file path: {value!r}.")
    if value == "model.json" or value.startswith("model.json/"):
        raise HTTPException(422, "model.json is supplied by the project field, not files.")
    return path


def validate_project(project: Any, files: dict[str, bytes], core_packages: dict[str, Path]) -> dict[str, Any]:
    if not isinstance(project, dict) or not isinstance(project.get("manifest"), dict):
        raise HTTPException(422, "project must be a schema-v2 model object with a manifest.")
    manifest = project["manifest"]
    if manifest.get("schemaVersion") != 2:
        raise HTTPException(422, "Only saved schema-v2 projects are accepted.")
    if not isinstance(project.get("nodes"), list) or not isinstance(project.get("edges"), list):
        raise HTTPException(422, "project must contain nodes and edges arrays.")
    custom_packages = _resource_refs(manifest.get("customPackages", []), "customPackages")
    custom_datasets = _resource_refs(manifest.get("customDatasets", []), "customDatasets")
    _unique_resources(custom_packages, "custom package")
    _unique_resources(custom_datasets, "dataset")
    active_dataset = manifest.get("activeDataset")
    if (
        not isinstance(active_dataset, dict)
        or not isinstance(active_dataset.get("id"), str)
        or not active_dataset["id"]
        or not isinstance(active_dataset.get("version"), str)
        or not active_dataset["version"]
        or (active_dataset["id"], active_dataset["version"]) not in {
        (item["id"], item["version"]) for item in custom_datasets
        }
    ):
        raise HTTPException(422, "Training requires an active dataset that resolves to a declared dataset.")

    packages: dict[tuple[str, str], dict[str, Any]] = {}
    for folder_name, folder in core_packages.items():
        manifest_value = _read_json(folder / "manifest.json", f"Core package {folder_name}")
        definition_file = _definition_file(manifest_value, "Core package")
        definition = _read_json(folder / definition_file, f"Core package {folder_name} definition")
        packages[(manifest_value["id"], manifest_value["version"])] = {
            "kind": definition.get("kind"), "outputs": definition.get("outputs"),
            "dependencies": manifest_value.get("dependencies", {}), "objective": definition.get("objective", {}),
        }
    for ref in custom_packages:
        prefix = ref["path"].rstrip("/")
        manifest_value = _read_json(files.get(f"{prefix}/manifest.json"), f"Custom package {prefix}")
        if (manifest_value.get("id"), manifest_value.get("version")) != (ref["id"], ref["version"]):
            raise HTTPException(422, f"Custom package path {prefix!r} does not match its declared identity.")
        if ref["id"].startswith("core."):
            raise HTTPException(422, "Custom packages cannot claim a core package identity.")
        definition_file = _definition_file(manifest_value, "Custom package")
        definition = _read_json(files.get(f"{prefix}/{definition_file}"), f"Custom package {prefix} definition")
        entrypoints = manifest_value.get("entrypoints", {})
        pytorch = entrypoints.get("pytorch") if isinstance(entrypoints, dict) else None
        if not isinstance(pytorch, dict) or pytorch.get("language") != "python" or not isinstance(pytorch.get("file"), str):
            raise HTTPException(422, f"Custom package {prefix!r} has no Python PyTorch entrypoint.")
        build_path = _safe_entrypoint(pytorch["file"], f"Custom package {prefix} PyTorch entrypoint")
        if f"{prefix}/{build_path}" not in files:
            raise HTTPException(422, f"Custom package {prefix!r} is missing its PyTorch entrypoint file.")
        packages[(ref["id"], ref["version"])] = {
            "kind": definition.get("kind"), "outputs": definition.get("outputs"),
            "dependencies": manifest_value.get("dependencies", {}), "objective": definition.get("objective", {}),
        }
    _validate_package_dependencies(packages)
    datasets: dict[tuple[str, str], dict[str, Any]] = {}
    for ref in custom_datasets:
        prefix = ref["path"].rstrip("/")
        dataset_manifest = _read_json(files.get(f"{prefix}/manifest.json"), f"Dataset {prefix}")
        if (dataset_manifest.get("id"), dataset_manifest.get("version")) != (ref["id"], ref["version"]):
            raise HTTPException(422, f"Dataset path {prefix!r} does not match its declared identity.")
        definition_file = _definition_file(dataset_manifest, "Dataset")
        dataset_definition = _read_json(files.get(f"{prefix}/{definition_file}"), f"Dataset {prefix} definition")
        batch = dataset_definition.get("batch")
        if not isinstance(batch, dict) or not isinstance(batch.get("inputs"), dict) or not isinstance(batch.get("targets"), dict):
            raise HTTPException(422, f"Dataset {prefix!r} definition must declare batch.inputs and batch.targets.")
        for section in (batch["inputs"], batch["targets"]):
            for name, slot in section.items():
                if not isinstance(name, str) or not name or not isinstance(slot, dict) or not isinstance(slot.get("dtype"), str) or not isinstance(slot.get("shape"), list):
                    raise HTTPException(422, f"Dataset {prefix!r} has an invalid tensor slot declaration.")
        datasets[(ref["id"], ref["version"])] = dataset_definition
        entrypoints = dataset_manifest.get("entrypoints", {})
        python = entrypoints.get("python") if isinstance(entrypoints, dict) else None
        if not isinstance(python, dict) or python.get("language") != "python" or not isinstance(python.get("file"), str):
            raise HTTPException(422, f"Dataset {prefix!r} has no Python adapter entrypoint.")
        python_path = _safe_entrypoint(python["file"], f"Dataset {prefix} Python adapter")
        if f"{prefix}/{python_path}" not in files:
            raise HTTPException(422, f"Dataset {prefix!r} is missing its Python adapter file.")

    node_map: dict[str, dict[str, Any]] = {}
    kinds: dict[str, str] = {}
    output_types: dict[str, dict[str, str]] = {}
    for node in project["nodes"]:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str) or not node["id"]:
            raise HTTPException(422, "Each graph node must have a non-empty string id.")
        node_id = node["id"]
        if node_id in node_map:
            raise HTTPException(422, f"Duplicate graph node id: {node_id!r}.")
        data = node.get("data")
        package = data.get("package") if isinstance(data, dict) else None
        if not isinstance(package, dict):
            raise HTTPException(422, f"Node {node_id!r} has no package identity.")
        identity = (package.get("id"), package.get("version"))
        if not all(isinstance(part, str) and part for part in identity):
            raise HTTPException(422, f"Node {node_id!r} has an invalid package identity.")
        resolved = packages.get(identity)
        if resolved is None:
            raise HTTPException(422, f"Node {node_id!r} uses package {identity[0]}@{identity[1]} outside the activated project closure.")
        kind = resolved["kind"]
        if not isinstance(kind, str):
            raise HTTPException(422, f"Package {identity[0]} has no valid kind.")
        node_map[node_id] = node
        kinds[node_id] = kind
        outputs = resolved["outputs"]
        if outputs is None:
            outputs = [] if kind in {"output", "loss-output"} else ([{"id": "loss", "type": "loss"}] if kind == "loss" else [{"id": "out", "type": "output"}])
        if not isinstance(outputs, list):
            raise HTTPException(422, f"Package {identity[0]} has invalid output declarations.")
        output_types[node_id] = {}
        for item in outputs:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or item.get("type") not in {"output", "loss"}:
                raise HTTPException(422, f"Package {identity[0]} has invalid output declarations.")
            if item["id"] in output_types[node_id]:
                raise HTTPException(422, f"Package {identity[0]} has duplicate output handles.")
            output_types[node_id][item["id"]] = item["type"]
    if not node_map:
        raise HTTPException(422, "Training requires a non-empty graph.")

    edge_ids: set[str] = set()
    occupied: set[tuple[str, str]] = set()
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_map}
    scopes = {}
    for node_id, node in node_map.items():
        data = node.get("data")
        scope = data.get("scope", "") if isinstance(data, dict) else ""
        if not isinstance(scope, str):
            raise HTTPException(422, f"Node {node_id!r} has an invalid graph scope.")
        scopes[node_id] = scope
    for node_id, scope in scopes.items():
        if scope and (scope not in node_map or kinds[scope] != "subflow"):
            raise HTTPException(422, f"Node {node_id!r} belongs to an orphan or non-subflow scope {scope!r}.")
    root_inputs = [node_id for node_id, kind in kinds.items() if kind == "input" and not scopes[node_id]]
    root_outputs = [node_id for node_id, kind in kinds.items() if kind == "output" and not scopes[node_id]]
    root_losses = [node_id for node_id, kind in kinds.items() if kind == "loss-output" and not scopes[node_id]]
    if not root_inputs or len(root_outputs) != 1 or len(root_losses) != 1:
        raise HTTPException(422, "A trainable root graph requires an Input, one Output, and one Loss Output.")
    selected_dataset = datasets[(active_dataset["id"], active_dataset["version"])]
    declared_inputs = selected_dataset["batch"]["inputs"]
    declared_targets = selected_dataset["batch"]["targets"]
    for node_id, kind in kinds.items():
        data = node_map[node_id].get("data", {})
        if kind == "input" and not scopes[node_id]:
            parameters = data.get("params", {}) if isinstance(data, dict) else {}
            binding = parameters.get("binding") if isinstance(parameters, dict) else None
            if not isinstance(binding, str) or binding not in declared_inputs:
                raise HTTPException(422, f"Input node {node_id!r} does not resolve to a selected dataset input slot.")
        package = data.get("package", {}) if isinstance(data, dict) else {}
        objective = packages[(package["id"], package["version"])].get("objective", {})
        external_inputs = objective.get("externalInputs", []) if isinstance(objective, dict) else []
        if not isinstance(external_inputs, list):
            raise HTTPException(422, f"Package objective on node {node_id!r} has invalid external inputs.")
        for external in external_inputs:
            source_name = external.get("source") if isinstance(external, dict) else None
            parts = source_name.split(".") if isinstance(source_name, str) else []
            if len(parts) != 3 or parts[:2] != ["batch", "targets"] or parts[2] not in declared_targets:
                raise HTTPException(422, f"Node {node_id!r} objective references an undeclared dataset target.")
    for edge in project["edges"]:
        if not isinstance(edge, dict) or not isinstance(edge.get("id"), str) or not edge["id"]:
            raise HTTPException(422, "Each graph edge must have a non-empty string id.")
        edge_id = edge["id"]
        if edge_id in edge_ids:
            raise HTTPException(422, f"Duplicate graph edge id: {edge_id!r}.")
        edge_ids.add(edge_id)
        source, target = edge.get("source"), edge.get("target")
        source_handle, target_handle = edge.get("sourceHandle"), edge.get("targetHandle")
        if not all(isinstance(value, str) and value for value in (source, target, source_handle, target_handle)):
            raise HTTPException(422, f"Edge {edge_id!r} has invalid node or handle identifiers.")
        if source not in node_map or target not in node_map:
            raise HTTPException(422, f"Edge {edge_id!r} references a missing graph node.")
        if scopes[source] != scopes[target]:
            raise HTTPException(422, f"Edge {edge_id!r} crosses an immediate graph scope.")
        if (target, target_handle) in occupied:
            raise HTTPException(422, f"Target handle {target!r}/{target_handle!r} is connected more than once.")
        occupied.add((target, target_handle))
        source_type = output_types[source].get(source_handle)
        if source_type is None:
            raise HTTPException(422, f"Edge {edge_id!r} uses unknown source handle {source_handle!r}.")
        if kinds[target] == "input":
            raise HTTPException(422, "Input nodes cannot have incoming edges.")
        if kinds[target] in {"output", "loss-output"}:
            if target_handle != "in":
                raise HTTPException(422, f"Terminal {target!r} accepts only the 'in' handle.")
            required = "output" if kinds[target] == "output" else "loss"
            if source_type != required:
                raise HTTPException(422, f"Terminal {target!r} requires a {required} typed source.")
        elif kinds[target] == "join":
            if not isinstance(target_handle, str) or not target_handle.startswith("in-") or not target_handle[3:].isdigit() or int(target_handle[3:]) < 1:
                raise HTTPException(422, f"Join {target!r} requires numeric in-N target handles.")
        elif target_handle != "in":
            raise HTTPException(422, f"Node {target!r} accepts only the 'in' target handle.")
        if kinds[source] in {"output", "loss-output"}:
            raise HTTPException(422, "Output terminals cannot have outgoing edges.")
        adjacency[source].append(target)
    for node_id in node_map:
        if _cycle_reachable(node_id, adjacency):
            raise HTTPException(422, "Graph contains a directed cycle.")
    _validate_operations(manifest.get("operations", []), node_map, kinds, scopes, output_types)
    return project


def _validate_operations(operations: Any, nodes, kinds, scopes, output_types) -> None:
    if not isinstance(operations, list):
        raise HTTPException(422, "manifest.operations must be an array.")
    names: set[str] = set()
    for operation in operations:
        if not isinstance(operation, dict):
            raise HTTPException(422, "Each manifest operation must be an object.")
        name = operation.get("name")
        if (not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name)
                or name.startswith("_") or name in {"infer", "inference", "run_operation"}):
            raise HTTPException(422, f"Invalid or reserved operation name {name!r}.")
        if name in names:
            raise HTTPException(422, f"Duplicate operation name {name!r}.")
        names.add(name)
        endpoints = {}
        for role in ("input", "output"):
            endpoint = operation.get(role)
            if not isinstance(endpoint, dict) or not all(isinstance(endpoint.get(key), str) and endpoint[key] for key in ("node", "handle")):
                raise HTTPException(422, f"Operation {name!r} has an invalid {role} endpoint.")
            if endpoint.get("codec") not in {"dataset", "tensor"}:
                raise HTTPException(422, f"Operation {name!r} has an invalid {role} codec.")
            node_id, handle = endpoint["node"], endpoint["handle"]
            if node_id not in nodes or scopes[node_id]:
                raise HTTPException(422, f"Operation {name!r} references a missing or non-root {role} node.")
            kind = kinds[node_id]
            if role == "input":
                if kind == "input":
                    valid = handle in output_types[node_id]
                elif kind == "join":
                    valid = handle.startswith("in-") and handle[3:].isdigit() and int(handle[3:]) > 0
                else:
                    valid = kind not in {"output", "loss-output"} and handle == "in"
            else:
                valid = kind not in {"input", "output", "loss-output"} and handle in output_types[node_id]
            if not valid:
                raise HTTPException(422, f"Operation {name!r} references undeclared {role} handle {node_id!r}.{handle}.")
            endpoints[role] = endpoint


def _validate_package_dependencies(packages: dict[tuple[str, str], dict[str, Any]]) -> None:
    graph: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for identity, package in packages.items():
        dependencies = package.get("dependencies", {})
        if not isinstance(dependencies, dict):
            raise HTTPException(422, f"Package {identity[0]} has invalid dependencies.")
        resolved = []
        for package_id, constraint in dependencies.items():
            matches = [candidate for candidate in packages if candidate[0] == package_id and _satisfies(candidate[1], str(constraint))]
            if len(matches) != 1:
                raise HTTPException(422, f"Package dependency {package_id!r} from {identity[0]!r} does not resolve uniquely.")
            resolved.append(matches[0])
        graph[identity] = resolved
    active: set[tuple[str, str]] = set()
    done: set[tuple[str, str]] = set()

    def visit(identity: tuple[str, str]) -> None:
        if identity in active:
            raise HTTPException(422, f"Package dependency cycle includes {identity[0]!r}.")
        if identity in done:
            return
        active.add(identity)
        for dependency in graph[identity]:
            visit(dependency)
        active.remove(identity)
        done.add(identity)

    for identity in graph:
        visit(identity)


def _satisfies(version: str, constraint: str) -> bool:
    if constraint in {"", "*"} or version == constraint:
        return True
    if not constraint.startswith(("^", "~")):
        return False
    try:
        wanted = tuple(int(part) for part in constraint[1:].split("."))
        actual = tuple(int(part) for part in version.split("."))
    except ValueError:
        return False
    if len(wanted) != 3 or len(actual) != 3 or actual < wanted:
        return False
    if constraint.startswith("~"):
        return actual[:2] == wanted[:2]
    return actual[0] == wanted[0] if wanted[0] else actual[1] == wanted[1] if wanted[1] else actual == wanted


def _resource_refs(refs: Any, label: str) -> list[dict[str, str]]:
    if not isinstance(refs, list):
        raise HTTPException(422, f"manifest.{label} must be an array.")
    result = []
    for ref in refs:
        if not isinstance(ref, dict) or not all(isinstance(ref.get(key), str) and ref[key] for key in ("id", "version", "path")):
            raise HTTPException(422, f"Each manifest.{label} entry requires id, version, and path.")
        path = safe_relative_path(ref["path"])
        result.append({"id": ref["id"], "version": ref["version"], "path": path.as_posix()})
    return result


def _unique_resources(refs: list[dict[str, str]], label: str) -> None:
    identities: set[tuple[str, str]] = set()
    paths: set[str] = set()
    for ref in refs:
        identity = (ref["id"], ref["version"])
        if identity in identities or ref["path"] in paths:
            raise HTTPException(422, f"Duplicate {label} identity or path: {identity[0]}@{identity[1]}.")
        identities.add(identity)
        paths.add(ref["path"])


def _read_json(value: bytes | Path | None, label: str) -> dict[str, Any]:
    if isinstance(value, Path):
        try:
            parsed = json.loads(value.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise HTTPException(422, f"{label} is missing or invalid JSON.") from None
    elif isinstance(value, bytes):
        try:
            parsed = json.loads(value.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise HTTPException(422, f"{label} is missing or invalid JSON.") from None
    else:
        raise HTTPException(422, f"{label} is missing or invalid JSON.")
    if not isinstance(parsed, dict):
        raise HTTPException(422, f"{label} must be a JSON object.")
    return parsed


def _definition_file(manifest: dict[str, Any], label: str) -> str:
    entrypoints = manifest.get("entrypoints")
    definition = entrypoints.get("definition") if isinstance(entrypoints, dict) else None
    return _safe_entrypoint(definition, f"{label} definition")


def _safe_entrypoint(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise HTTPException(422, f"{label} has an invalid relative path.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")) or ":" in path.parts[0]:
        raise HTTPException(422, f"{label} has an unsafe relative path.")
    return path.as_posix()


def _cycle_reachable(start: str, adjacency: dict[str, list[str]]) -> bool:
    active: set[str] = set()
    done: set[str] = set()

    def visit(node: str) -> bool:
        if node in active:
            return True
        if node in done:
            return False
        active.add(node)
        for target in adjacency[node]:
            if visit(target):
                return True
        active.remove(node)
        done.add(node)
        return False

    return visit(start)


def validate_training(value: Any) -> dict[str, Any]:
    required = {"epochs", "batch_size", "learning_rate", "seed"}
    if not isinstance(value, dict) or frozenset(value) not in {frozenset(required), frozenset(required | {"publish_every_steps"})}:
        raise HTTPException(422, "training requires epochs, batch_size, learning_rate, seed, and optional publish_every_steps.")
    epochs = value["epochs"]
    batch = value["batch_size"]
    seed = value["seed"]
    rate = value["learning_rate"]
    publish_every_steps = value.get("publish_every_steps", 10)
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 10000:
        raise HTTPException(422, "epochs must be an integer from 1 to 10000.")
    if isinstance(batch, bool) or not isinstance(batch, int) or not 1 <= batch <= 4096:
        raise HTTPException(422, "batch_size must be an integer from 1 to 4096.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not -(2**31) <= seed < 2**32:
        raise HTTPException(422, "seed must be a 32-bit integer.")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0 < float(rate) <= 1:
        raise HTTPException(422, "learning_rate must be finite and in (0, 1].")
    import math
    if not math.isfinite(float(rate)):
        raise HTTPException(422, "learning_rate must be finite and in (0, 1].")
    if isinstance(publish_every_steps, bool) or not isinstance(publish_every_steps, int) or not 1 <= publish_every_steps <= 100000:
        raise HTTPException(422, "publish_every_steps must be an integer from 1 to 100000.")
    return {"epochs": epochs, "batch_size": batch, "learning_rate": float(rate), "seed": seed,
            "publish_every_steps": publish_every_steps}


def resolve_core_packages(project: dict[str, Any], files: dict[str, bytes] | None = None) -> dict[str, Path]:
    """Resolve only node-referenced core packages and their declared core dependencies."""
    catalog: dict[tuple[str, str], Path] = {}
    for folder in config.CORE_ROOT.iterdir() if config.CORE_ROOT.is_dir() else ():
        if folder.is_symlink() or not folder.is_dir():
            continue
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            continue
        try:
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
            identity = (metadata["id"], metadata["version"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        catalog[identity] = folder
    needed: set[tuple[str, str]] = set()
    pending: list[tuple[str, str]] = []
    for node in project["nodes"]:
        if not isinstance(node, dict):
            continue
        data = node.get("data")
        package = data.get("package") if isinstance(data, dict) else None
        if isinstance(package, dict) and isinstance(package.get("id"), str) and package["id"].startswith("core."):
            pending.append((package["id"], str(package.get("version", ""))))
    files = files or {}
    manifest = project.get("manifest", {})
    for ref in manifest.get("customPackages", []) if isinstance(manifest, dict) else []:
        if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
            continue
        data = files.get(f"{ref['path'].rstrip('/')}/manifest.json")
        if not data:
            continue
        try:
            dependencies = json.loads(data.decode("utf-8")).get("dependencies", {})
        except (UnicodeDecodeError, ValueError, AttributeError):
            continue
        if isinstance(dependencies, dict):
            for package_id, constraint in dependencies.items():
                if isinstance(package_id, str) and package_id.startswith("core."):
                    matches = [key for key in catalog if key[0] == package_id and _satisfies(key[1], str(constraint))]
                    if len(matches) != 1:
                        raise HTTPException(422, f"Core dependency {package_id} does not resolve uniquely.")
                    pending.append(matches[0])
    while pending:
        identity = pending.pop()
        if identity in needed:
            continue
        folder = catalog.get(identity)
        if folder is None:
            raise HTTPException(422, f"Required core package {identity[0]}@{identity[1]} is not installed.")
        needed.add(identity)
        metadata = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        dependencies = metadata.get("dependencies", {})
        if not isinstance(dependencies, dict):
            raise HTTPException(422, f"Core package {identity[0]} has invalid dependencies.")
        for dep_id, constraint in dependencies.items():
            if dep_id.startswith("core."):
                matches = [key for key in catalog if key[0] == dep_id and _satisfies(key[1], str(constraint))]
                if len(matches) != 1:
                    raise HTTPException(422, f"Core dependency {dep_id} from {identity[0]} does not resolve uniquely.")
                pending.append(matches[0])
    result: dict[str, Path] = {}
    for package_id, version in sorted(needed):
        source = catalog[(package_id, version)]
        _reject_symlinks(source)
        result[source.name] = source
    return result


def _reject_symlinks(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise HTTPException(422, f"Core package contains a symlink: {path.name}.")


def copy_core_packages(packages: dict[str, Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name, source in packages.items():
        shutil.copytree(source, destination / name, symlinks=False)
