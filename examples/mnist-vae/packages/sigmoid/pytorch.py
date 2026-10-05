from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


class Sigmoid(nn.Module):
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) != 1:
            raise ValueError("Sigmoid requires exactly one input")
        value = inputs[0]
        if not value.is_floating_point():
            raise ValueError("Sigmoid requires a floating tensor")
        return torch.sigmoid(value)


def build(parameters: Mapping[str, Any], context: Mapping[str, Any], services: object) -> nn.Module:
    del parameters, context, services
    return Sigmoid()
