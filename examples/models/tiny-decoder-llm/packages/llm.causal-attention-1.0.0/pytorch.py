import torch
from torch import nn


class CausalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(64, 4, batch_first=True)

    def forward(self, value):
        length = value.shape[1]
        mask = torch.ones(
            (length, length), dtype=torch.bool, device=value.device
        ).triu(1)
        output, _ = self.attention(
            value, value, value, attn_mask=mask, need_weights=False
        )
        return output


def build(parameters, context, services):
    del parameters, context, services
    return CausalAttention()
