import torch
from torch import nn


class CausalMask(nn.Module):
    def __init__(self, max_length):
        super().__init__()
        self.max_length = max_length

    def forward(self, scores):
        if scores.ndim != 3 or scores.shape[-1] != scores.shape[-2]:
            raise ValueError("causal mask expects rank-3 square [B,T,T] attention scores")
        if not scores.is_floating_point():
            raise ValueError("causal mask expects floating attention scores")
        length = scores.shape[-1]
        if length > self.max_length:
            raise ValueError("sequence length exceeds configured context")
        mask = torch.ones(
            (length, length), dtype=torch.bool, device=scores.device
        ).triu(1)
        return scores.masked_fill(mask, torch.finfo(scores.dtype).min)


def build(parameters, context, services):
    del context, services
    max_length = parameters.get("max_length", 128)
    if type(max_length) is not int or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    return CausalMask(max_length)
