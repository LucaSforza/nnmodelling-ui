from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


class KLDivergence(nn.Module):
    """KL(q(z|x) || N(0,I)), reduced over latent features per sample."""

    def __init__(self, latent_features: int) -> None:
        super().__init__()
        if type(latent_features) is not int or latent_features < 1:
            raise ValueError("latent_features must be a positive integer")
        self.latent_features = latent_features

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) != 1:
            raise ValueError("KL divergence requires one packed Gaussian tensor")
        parameters = inputs[0]
        if parameters.ndim != 3 or parameters.shape[1] != 2 or parameters.shape[2] != self.latent_features:
            raise ValueError("Expected Gaussian parameters [B,2,latent_features]")
        if parameters.shape[0] < 1 or not parameters.is_floating_point():
            raise ValueError("Gaussian parameters must be a nonempty floating tensor")
        mean, log_variance = parameters.unbind(dim=1)
        return -0.5 * (1.0 + log_variance - mean.square() - log_variance.exp()).sum(dim=1)


def build(parameters: Mapping[str, Any], context: Mapping[str, Any], services: object) -> nn.Module:
    del context, services
    return KLDivergence(parameters.get("latent_features", 32))
