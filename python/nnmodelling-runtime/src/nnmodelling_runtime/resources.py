from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .dataset import Batch, DatasetAdapter
from ._imports import resource_module
from .graph import _project_path, _read_json, _safe_relative


class _BoundDatasetAdapter(DatasetAdapter[Any, Any]):
    """Normalize adapter input names at the dataset boundary."""

    def __init__(self, adapter: DatasetAdapter, input_names: tuple[str, ...]):
        super().__init__(adapter.directory)
        self._adapter = adapter
        self._input_names = input_names

    def __getattr__(self, name: str) -> Any:
        return getattr(self._adapter, name)

    @property
    def data_directory(self) -> Path:
        return self._adapter.data_directory

    @property
    def cache_directory(self) -> Path | None:
        return self._adapter.cache_directory

    def _normalize_inputs(self, inputs: Any, source: str) -> dict[str, torch.Tensor]:
        if not isinstance(inputs, Mapping):
            raise TypeError(f"dataset input error: {source} must return a named tensor map")
        if any(not isinstance(name, str) for name in inputs):
            raise TypeError(f"dataset input error: {source} input names must be strings")
        if len(self._input_names) == 1:
            if len(inputs) != 1:
                raise ValueError(
                    f"dataset input error: {source} returned {len(inputs)} inputs; "
                    "the dataset declares one input"
                )
            items = ((self._input_names[0], next(iter(inputs.values()))),)
        else:
            actual = set(inputs)
            expected = set(self._input_names)
            if actual != expected:
                raise ValueError(
                    f"dataset input error: {source} names must exactly match "
                    f"declared inputs {self._input_names!r}; got {tuple(inputs)!r}"
                )
            items = ((name, inputs[name]) for name in self._input_names)
        normalized = {}
        for name, tensor in items:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"dataset input error: {source} input {name!r} must be a Tensor")
            normalized[name] = tensor
        return normalized

    def tokenize(self, value: Any) -> torch.Tensor | dict[str, torch.Tensor]:
        tokenized = self._adapter.tokenize(value)
        if isinstance(tokenized, torch.Tensor):
            if len(self._input_names) != 1:
                raise ValueError("dataset input error: one Tensor requires exactly one declared input")
            return tokenized
        return self._normalize_inputs(tokenized, "tokenize")

    def untokenize(self, tensor: torch.Tensor) -> Any:
        return self._adapter.untokenize(tensor)

    def load(self, split: str, batch_size: int):
        for batch in self._adapter.load(split, batch_size):
            if not isinstance(batch, Batch):
                raise TypeError("dataset input error: load must yield Batch values")
            yield Batch(
                inputs=self._normalize_inputs(batch.inputs, f"load({split!r})"),
                targets=batch.targets,
            )


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
    definition_name = dataset_manifest.get("entrypoints", {}).get("definition", "dataset.json")
    definition = _read_json(_project_path(directory, definition_name))
    declared_inputs = definition.get("batch", {}).get("inputs")
    if not isinstance(declared_inputs, dict) or not declared_inputs or any(
        not isinstance(name, str) or not name for name in declared_inputs
    ):
        raise ValueError("dataset input error: definition must declare named batch.inputs")
    resource = _project_path(directory, entry.get("file", ""))
    module = resource_module(f"_nnmodel_dataset_{active['id'].replace('.', '_')}_{active['version'].replace('.', '_')}", directory, resource, __package__)
    adapter_type = getattr(module, "Dataset", None)
    # Wheel-local runtime copies must recognize adapters derived from the public
    # SDK package, which can have a distinct Python class identity.
    if not isinstance(adapter_type, type) or not issubclass(adapter_type, DatasetAdapter):
        raise TypeError("dataset entrypoint must export Dataset, a DatasetAdapter subclass")
    return _BoundDatasetAdapter(adapter_type(directory), tuple(declared_inputs))
