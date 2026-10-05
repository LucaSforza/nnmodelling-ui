"""Numerical resource for the internal operator, not a graph compiler.

Uses the core packages' build(parameters, context, services) convention without
depending on their external stereotype_runtime module. No prediction head here.
"""

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


class LastStateRNN(nn.Module):
    """[B,T,F] -> [B,H], with one shared-weight tanh RNN and fresh zero h0."""

    def __init__(
        self,
        sequence_length: int = 32,
        input_size: int = 1,
        hidden_size: int = 32,
        bias: bool = True,
    ) -> None:
        super().__init__()
        for name, value in (
            ("sequence_length", sequence_length),
            ("input_size", input_size),
            ("hidden_size", hidden_size),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(bias) is not bool:
            raise ValueError("bias must be boolean")
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            nonlinearity="tanh",
            bias=bias,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
            dtype=torch.float32,
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3 or tuple(sequence.shape[1:]) != (
            self.sequence_length,
            self.input_size,
        ):
            raise ValueError("Expected [B,sequence_length,input_size]")
        if sequence.shape[0] < 1 or sequence.dtype != torch.float32:
            raise ValueError("Expected a nonempty float32 batch")
        if self.rnn.weight_ih_l0.dtype != torch.float32:
            raise ValueError("RNN weights must remain float32")
        h0 = sequence.new_zeros((1, sequence.shape[0], self.hidden_size))
        _, h_n = self.rnn(sequence, h0)
        return h_n[0]


def build(
    parameters: Mapping[str, Any],
    context: Mapping[str, Any],
    services: object,
) -> nn.Module:
    """Package entrypoint; context/services are unused, no runtime API assumed."""
    if parameters.get("dtype", "float32") != "float32":
        raise ValueError("Only float32 is supported")
    if parameters.get("nonlinearity", "tanh") != "tanh":
        raise ValueError("Only tanh is supported")
    if parameters.get("initial_state", "zero") != "zero":
        raise ValueError("Only a zero initial state is supported")
    return LastStateRNN(
        sequence_length=parameters.get("sequence_length", 32),
        input_size=parameters.get("input_size", 1),
        hidden_size=parameters.get("hidden_size", 32),
        bias=parameters.get("bias", True),
    )
