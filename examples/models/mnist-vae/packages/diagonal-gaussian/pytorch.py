from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


class DiagonalGaussian(nn.Module):
    """Join ordered mean and log-variance heads as [B,2,L]."""

    def __init__(self, latent_features: int) -> None:
        super().__init__()
        if type(latent_features) is not int or latent_features < 1:
            raise ValueError("latent_features must be a positive integer")
        self.latent_features = latent_features

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) != 2:
            raise ValueError("Diagonal Gaussian requires ordered mean and log-variance inputs")
        mean, log_variance = inputs
        if mean.ndim != 2 or log_variance.ndim != 2:
            raise ValueError("Gaussian heads must have rank 2 [B,L]")
        if mean.shape[0] < 1 or mean.shape[1] != self.latent_features:
            raise ValueError("Mean must have shape [B,latent_features]")
        if mean.shape != log_variance.shape:
            raise ValueError("Mean and log-variance must have matching shapes")
        if mean.dtype != log_variance.dtype or mean.device != log_variance.device:
            raise ValueError("Mean and log-variance must have matching dtype and device")
        if not mean.is_floating_point():
            raise ValueError("Gaussian parameters must be floating tensors")
        return torch.stack((mean, log_variance), dim=1)


def build(parameters: Mapping[str, Any], context: Mapping[str, Any], services: object) -> nn.Module:
    del context, services
    return DiagonalGaussian(parameters.get("latent_features", 32))
