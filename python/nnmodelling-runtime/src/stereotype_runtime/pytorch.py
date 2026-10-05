"""Small stable Python ABI exposed to stereotype ``pytorch.py`` modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import torch

DType = str
BuildContext = Mapping[str, Any]
NoServices = Any


def torch_dtype(dtype: DType) -> torch.dtype:
    try:
        return {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "bfloat16": torch.bfloat16,
            "int8": torch.int8,
            "int16": torch.int16,
            "int32": torch.int32,
            "int64": torch.int64,
            "uint8": torch.uint8,
            "bool": torch.bool,
        }[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported tensor dtype: {dtype!r}") from exc


@dataclass(frozen=True)
class StereotypeReference:
    id: str
    version: str
    path: str
    manifest: Mapping[str, Any]
    definition: Mapping[str, Any]


class SubflowServices(Protocol):
    def build_subflow(self) -> torch.nn.Module: ...


class StereotypeServices(Protocol):
    def resolve(self, package_id: str, version: str | None = None) -> StereotypeReference: ...
    def build_subflow(self) -> torch.nn.Module: ...
    def build_stereotype(self, reference: StereotypeReference) -> torch.nn.Module: ...
