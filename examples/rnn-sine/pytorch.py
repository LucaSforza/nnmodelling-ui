"""Standalone reference composition of this design; not a client entrypoint.

No data generation, optimizer, training loop or backend integration is provided.
"""

import importlib.util
from pathlib import Path

import torch
from torch import nn


def _build_encoder() -> nn.Module:
    # Explicit project-owned path: no ambient package discovery or runtime API.
    path = (
        Path(__file__).resolve().parent
        / "packages"
        / "sine.rnn-last-state-1.0.0"
        / "pytorch.py"
    )
    spec = importlib.util.spec_from_file_location("sine_rnn_last_state", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the RNN resource: {path}")
    resource = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(resource)
    return resource.build(
        {
            "sequence_length": 32,
            "input_size": 1,
            "hidden_size": 32,
            "nonlinearity": "tanh",
            "initial_state": "zero",
            "dtype": "float32",
            "bias": True,
        },
        {},
        None,
    )


class SinePredictor(nn.Module):
    """One internal RNN followed by exactly one Linear 32->1 head."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = _build_encoder()
        self.head = nn.Linear(32, 1, bias=True, dtype=torch.float32)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(sequence))


def build_predictor() -> nn.Module:
    """Manual reference factory, not registered as a model/compiler entrypoint."""
    return SinePredictor()


def build_loss() -> nn.Module:
    """Future caller supplies predictions and float32 targets, both [B,1]."""
    return nn.MSELoss(reduction="mean")
