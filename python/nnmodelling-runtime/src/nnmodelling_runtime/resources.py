from __future__ import annotations

from pathlib import Path
from typing import Any

from .dataset import DatasetAdapter
from ._imports import resource_module
from .graph import _project_path, _read_json, _safe_relative


def load_dataset(project_dir: str | Path) -> DatasetAdapter:
    """Load the exact active project's Python dataset adapter."""
    root = Path(project_dir).resolve()
    model = _read_json(root / "model.json")
    manifest = model.get("manifest", {})
    active = manifest.get("activeDataset")
    if not active:
        raise ValueError("project has no active dataset")
    matches = [ref for ref in manifest.get("customDatasets", []) if ref.get("id") == active.get("id") and ref.get("version") == active.get("version")]
    if len(matches) != 1:
        raise ValueError(f"active dataset {active!r} is missing or ambiguous")
    directory = _project_path(root, matches[0]["path"])
    dataset_manifest = _read_json(_project_path(directory, "manifest.json"))
    if (dataset_manifest.get("id"), dataset_manifest.get("version")) != (active.get("id"), active.get("version")):
        raise ValueError("active dataset identity does not match its manifest")
    entry = dataset_manifest.get("entrypoints", {}).get("python")
    if not isinstance(entry, dict) or entry.get("language") != "python":
        raise ValueError("active dataset has no Python adapter entrypoint")
    resource = _project_path(directory, entry.get("file", ""))
    module = resource_module(f"_nnmodel_dataset_{active['id'].replace('.', '_')}_{active['version'].replace('.', '_')}", directory, resource, __package__)
    adapter_type = getattr(module, "Dataset", None)
    # Wheel-local runtime copies must recognize adapters derived from the public
    # SDK package, which can have a distinct Python class identity.
    if not isinstance(adapter_type, type) or not issubclass(adapter_type, DatasetAdapter):
        raise TypeError("dataset entrypoint must export Dataset, a DatasetAdapter subclass")
    return adapter_type(directory)
