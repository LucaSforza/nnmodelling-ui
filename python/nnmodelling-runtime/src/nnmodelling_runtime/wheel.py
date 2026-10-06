from __future__ import annotations

import base64
import csv
import hashlib
import json
import re
import shutil
import tempfile
import tomllib
from urllib.parse import urlsplit
import zipfile
from pathlib import Path
from typing import Any

from .graph import GraphModule, _Catalog, _read_json, _safe_relative


def _copy_resource_tree(source: Path, destination: Path) -> None:
    for item in source.rglob("*"):
        rel = item.relative_to(source)
        if any(part in {"__pycache__", ".venv"} for part in rel.parts):
            continue
        if item.is_symlink():
            raise ValueError(f"resource contains symlink: {item}")
        if item.is_file():
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, target)


def _active_dataset(root: Path, model: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    active = model.get("manifest", {}).get("activeDataset")
    matches = [r for r in model.get("manifest", {}).get("customDatasets", []) if active and (r.get("id"), r.get("version")) == (active.get("id"), active.get("version"))]
    if len(matches) != 1:
        raise ValueError("wheel export requires one exact active dataset")
    candidate = root / _safe_relative(matches[0]["path"])
    current = root
    for part in _safe_relative(matches[0]["path"]).parts:
        current /= part
        if current.is_symlink():
            raise ValueError("dataset path traverses symlink")
    directory = candidate.resolve()
    if not directory.is_relative_to(root):
        raise ValueError("dataset path escapes project")
    return directory, _read_json(directory / "manifest.json")


def _wheel_record(payload: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode("ascii")
    return f"sha256={digest}"


def _distribution_name(model: dict[str, Any]) -> str:
    manifest = model.get("manifest")
    project_id = manifest.get("id") if isinstance(manifest, dict) else None
    if not isinstance(project_id, str):
        raise ValueError("wheel export requires a string project ID")
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", project_id).strip("_").lower()
    if not normalized:
        raise ValueError("wheel export requires a project ID with ASCII letters or digits")
    return f"nnm_{normalized}"


def _copy_runtime(destination: Path) -> None:
    source_root = Path(__file__).resolve().parents[1]
    for package in ("nnmodelling_runtime", "stereotype_runtime"):
        _copy_resource_tree(source_root / package, destination / "_vendor" / package)


def _declared_python_dependencies(paths: list[Path]) -> list[str]:
    dependencies: set[str] = {"torch>=2.2", "safetensors>=0.4", "numpy>=1.26"}
    for directory in paths:
        pyproject = directory / "pyproject.toml"
        if not pyproject.is_file():
            continue
        project = tomllib.loads(pyproject.read_text(encoding="utf-8")).get("project", {})
        for requirement in project.get("dependencies", []):
            requirement = requirement.strip()
            if requirement.startswith(("/", "\\", "./", "../", "file:", "~")) or re.match(r"^[A-Za-z]:[\\/]", requirement):
                raise ValueError(f"wheel cannot preserve local Python dependency {requirement!r}")
            direct_url = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*@\s*(\S+)$", requirement)
            if direct_url:
                name, url = direct_url.groups()
                parsed = urlsplit(url)
                if url.startswith(("/", "\\", "./", "../", "file:")) or parsed.scheme == "file" or parsed.username is not None or parsed.password is not None:
                    raise ValueError(f"wheel cannot preserve local or authenticated Python dependency {requirement!r}")
                dependency_name = name.lower().replace("_", "-")
            else:
                match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
                if not match:
                    raise ValueError(f"unsupported Python dependency declaration {requirement!r}")
                dependency_name = match.group(1).lower().replace("_", "-")
            if dependency_name != "nnmodelling-runtime":
                dependencies.add(requirement)
    return sorted(dependencies)


def build_wheel(
    project_dir: str | Path,
    core_dir: str | Path | None,
    weights_path: str | Path,
    output_dir: str | Path,
    job_id: str,
) -> Path:
    """Export an inference wheel with model resources and no dataset payloads."""
    root = Path(project_dir).resolve()
    core_source = Path(core_dir).resolve() if core_dir else None
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = _read_json(root / "model.json")
    # Fail early using the same graph/resource validation as execution.
    graph = GraphModule(root, core_source)
    del graph
    dataset_source, dataset_manifest = _active_dataset(root, model)
    python_entry = dataset_manifest.get("entrypoints", {}).get("python", {})
    if python_entry.get("language") != "python" or not python_entry.get("file"):
        raise ValueError("wheel export requires a Python dataset adapter")

    dist_name = _distribution_name(model)
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", job_id).strip("_").lower()
    if not normalized or not normalized[0].isalpha():
        normalized = "job_" + normalized
    package_name = f"nnmodel_{normalized}"
    version = "0.1.0"
    wheel_name = f"{dist_name}-{version}-py3-none-any.whl"
    with tempfile.TemporaryDirectory(prefix="nnmodel-wheel-") as temporary:
        stage = Path(temporary) / package_name
        stage.mkdir()
        (stage / "__init__.py").write_text("from .model import Model\n\n__all__ = ['Model']\n", encoding="utf-8")
        _copy_runtime(stage)
        project_snapshot = stage / "_snapshot" / "project"
        core_snapshot = stage / "_snapshot" / "core"
        project_snapshot.mkdir(parents=True)
        core_snapshot.mkdir(parents=True)
        shutil.copyfile(root / "model.json", project_snapshot / "model.json")

        custom_refs = model.get("manifest", {}).get("customPackages", [])
        for ref in custom_refs:
            candidate = root / _safe_relative(ref["path"])
            current = root
            for part in _safe_relative(ref["path"]).parts:
                current /= part
                if current.is_symlink():
                    raise ValueError("stereotype package path traverses symlink")
            source = candidate.resolve()
            if not source.is_relative_to(root):
                raise ValueError("stereotype package path escapes project")
            _copy_resource_tree(source, project_snapshot / _safe_relative(ref["path"]))

        if core_source is None:
            repo_core = Path(__file__).resolve().parents[4] / "stereotype-packages" / "core"
            core_source = root / "stereotype-packages" / "core" if (root / "stereotype-packages" / "core").is_dir() else repo_core
        catalog = _Catalog(root, core_source)
        # The catalog also activates parameter/default stereotype references,
        # such as a join constructed inside HorizontalRepeat.
        required = catalog.activated
        for package_id in required:
            if package_id not in catalog.records:
                continue
            directory, manifest, _ = catalog.records[package_id]
            # Package path in the project manifest is authoritative for custom packages.
            custom_ref = next((r for r in custom_refs if r.get("id") == package_id), None)
            if custom_ref:
                target_rel = _safe_relative(custom_ref["path"])
            else:
                target_rel = Path("core") / directory.name
            if custom_ref:
                _copy_resource_tree(directory, project_snapshot / target_rel)
            else:
                _copy_resource_tree(directory, core_snapshot / directory.name)

        dataset_ref = model["manifest"]["activeDataset"]
        dataset_rel = _safe_relative(next(r["path"] for r in model["manifest"]["customDatasets"] if r["id"] == dataset_ref["id"] and r["version"] == dataset_ref["version"]))
        allowed = {"manifest.json"}
        allowed.add(dataset_manifest.get("entrypoints", {}).get("definition", "dataset.json"))
        allowed.add(python_entry["file"])
        allowed.update(path.relative_to(dataset_source).as_posix() for path in dataset_source.rglob("*.py"))
        dataset_destination = project_snapshot / dataset_rel
        for name in allowed:
            candidate = dataset_source / _safe_relative(name)
            current = dataset_source
            for part in _safe_relative(name).parts:
                current /= part
                if current.is_symlink():
                    raise ValueError(f"dataset inference resource traverses symlink: {name!r}")
            source = candidate.resolve()
            if not source.is_relative_to(dataset_source) or not source.is_file():
                raise ValueError(f"invalid or missing dataset inference resource: {name!r}")
            target = dataset_destination / _safe_relative(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        for asset in dataset_manifest.get("inferenceAssets", []):
            asset_rel = _safe_relative(asset)
            candidate = root / asset_rel
            current = root
            for part in asset_rel.parts:
                current /= part
                if current.is_symlink():
                    raise ValueError(f"inference asset traverses symlink: {asset!r}")
            source = candidate.resolve()
            if not source.is_relative_to(root) or not source.is_file():
                raise ValueError(f"invalid or missing project inference asset: {asset!r}")
            target = project_snapshot / asset_rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        # Runtime expects core_dir to contain direct package subdirectories.
        (stage / "model.py").write_text(
            "from __future__ import annotations\n"
            "from pathlib import Path\n"
            "import torch\n"
            "from safetensors.torch import load_file\n"
            "from ._vendor.nnmodelling_runtime import GraphModule, load_dataset\n"
            "\n"
            "class Model:\n"
            "    def __init__(self, weights_path=None):\n"
            "        self._root = Path(__file__).parent\n"
            "        self._project = self._root / '_snapshot' / 'project'\n"
            "        self._adapter = load_dataset(self._project)\n"
            "        self._graph = GraphModule(self._project, self._root / '_snapshot' / 'core')\n"
            "        path = Path(weights_path) if weights_path is not None else self._root / 'weights.safetensors'\n"
            "        if not path.is_file(): raise FileNotFoundError(path)\n"
            "        self._graph.load_state_dict(load_file(str(path), device='cpu'))\n"
            "        self._graph.eval()\n"
            "    def infer(self, value):\n"
            "        tokenized = self._adapter.tokenize(value)\n"
            "        if isinstance(tokenized, torch.Tensor):\n"
            "            inputs = self._graph.single_input_mapping(tokenized)\n"
            "        else: inputs = tokenized\n"
            "        with torch.inference_mode(): prediction = self._graph(inputs)['prediction']\n"
            "        return self._adapter.untokenize(prediction)\n"
            "    inference = infer\n",
            encoding="utf-8",
        )
        shutil.copyfile(weights_path, stage / "weights.safetensors")
        dist_info = f"{dist_name}-{version}.dist-info"
        entries: dict[str, bytes] = {}
        for path in stage.rglob("*"):
            if path.is_file():
                entries[f"{package_name}/{path.relative_to(stage).as_posix()}"] = path.read_bytes()
        runtime_dependencies = _declared_python_dependencies([dataset_source] + [catalog.records[p][0] for p in required if p in catalog.records])
        metadata = (
            f"Metadata-Version: 2.1\nName: {dist_name}\nVersion: {version}\n"
            "Requires-Python: >=3.11\n"
            + "".join(f"Requires-Dist: {dependency}\n" for dependency in runtime_dependencies)
            + "\n"
        )
        entries[f"{dist_info}/METADATA"] = metadata.encode()
        entries[f"{dist_info}/WHEEL"] = b"Wheel-Version: 1.0\nGenerator: nnmodelling-runtime\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        record_path = f"{dist_info}/RECORD"
        rows = [(name, _wheel_record(payload), str(len(payload))) for name, payload in sorted(entries.items())]
        rows.append((record_path, "", ""))
        import io
        record = io.StringIO(newline="")
        csv.writer(record, lineterminator="\n").writerows(rows)
        entries[record_path] = record.getvalue().encode()
        destination = output / wheel_name
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in sorted(entries.items()):
                archive.writestr(name, payload)
    return destination
