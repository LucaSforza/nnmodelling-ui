import torch
from torch import nn


class TokenCrossEntropy(nn.Module):
    def forward(self, logits, target):
        if logits.ndim != 3 or target.shape != logits.shape[:2]:
            raise ValueError("token cross entropy expects logits [B,T,V] and targets [B,T]")
        flattened_logits = logits.reshape(-1, logits.shape[-1])
        flattened_target = target.reshape(-1).long()
        return torch.nn.functional.cross_entropy(flattened_logits, flattened_target)


def build(parameters, context, services):
    del parameters, context, services
    return TokenCrossEntropy()
