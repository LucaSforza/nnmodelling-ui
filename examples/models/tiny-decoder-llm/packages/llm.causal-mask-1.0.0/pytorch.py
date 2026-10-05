import torch
from torch import nn


class CausalMask(nn.Module):
    def __init__(self, max_length):
        super().__init__()
        self.max_length = max_length

    def forward(self, scores):
        if scores.ndim != 3 or scores.shape[-1] != scores.shape[-2]:
            raise ValueError("causal mask expects [B,T,T] attention scores")
        length = scores.shape[-1]
        if length > self.max_length:
            raise ValueError("sequence length exceeds configured context")
        mask = torch.ones(
            (length, length), dtype=torch.bool, device=scores.device
        ).triu(1)
        return scores.masked_fill(mask, torch.finfo(scores.dtype).min)


def build(parameters, context, services):
    del context, services
    return CausalMask(parameters["max_length"])
