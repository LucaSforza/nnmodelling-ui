from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


class TotalLoss(nn.Module):
    """Combine summed 28x28 pixel error with mean per-sample KL."""

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) != 2:
            raise ValueError("VAE total loss requires reconstruction MSE then per-sample KL")
        reconstruction, kl = inputs
        if reconstruction.ndim != 0 or kl.ndim != 1 or kl.shape[0] < 1:
            raise ValueError("Expected scalar reconstruction MSE and nonempty per-sample KL [B]")
        if not reconstruction.is_floating_point() or reconstruction.dtype != kl.dtype or reconstruction.device != kl.device:
            raise ValueError("Reconstruction MSE and KL must have the same floating dtype and device")
        return reconstruction * (28 * 28) + kl.mean()


def build(parameters: Mapping[str, Any], context: Mapping[str, Any], services: object) -> nn.Module:
    del parameters, context, services
    return TotalLoss()
