import torch
from torch import nn


class CausalAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout, bias):
        super().__init__()
        self.d_model = d_model
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, bias=bias, batch_first=True
        )

    def forward(self, value):
        if value.ndim != 3 or value.shape[-1] != self.d_model:
            raise ValueError("causal attention expects floating [B,T,d_model]")
        if not value.is_floating_point():
            raise ValueError("causal attention expects floating input")
        length = value.shape[1]
        mask = torch.ones(
            (length, length), dtype=torch.bool, device=value.device
        ).triu(1)
        output, _ = self.attention(
            value, value, value, attn_mask=mask, need_weights=False
        )
        return output


def build(parameters, context, services):
    del context, services
    d_model = parameters.get("d_model", 64)
    num_heads = parameters.get("num_heads", 4)
    dropout = parameters.get("dropout", 0.0)
    bias = parameters.get("bias", True)
    if type(d_model) is not int or d_model < 1:
        raise ValueError("d_model must be a positive integer")
    if type(num_heads) is not int or num_heads < 1:
        raise ValueError("num_heads must be a positive integer")
    if d_model % num_heads:
        raise ValueError("d_model must be divisible by num_heads")
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout <= 1:
        raise ValueError("dropout must be in [0,1]")
    if type(bias) is not bool:
        raise ValueError("bias must be boolean")
    return CausalAttention(d_model, num_heads, float(dropout), bias)
