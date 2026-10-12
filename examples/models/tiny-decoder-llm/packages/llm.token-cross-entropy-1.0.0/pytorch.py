import torch
from torch import nn


class TokenCrossEntropy(nn.Module):
    def __init__(self, ignore_index, label_smoothing):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        if logits.ndim != 3 or target.shape != logits.shape[:2]:
            raise ValueError("token cross entropy expects logits [B,T,V] and targets [B,T]")
        flattened_logits = logits.reshape(-1, logits.shape[-1])
        flattened_target = target.reshape(-1).long()
        return torch.nn.functional.cross_entropy(
            flattened_logits,
            flattened_target,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )


def build(parameters, context, services):
    del context, services
    ignore_index = parameters.get("ignore_index", -100)
    label_smoothing = parameters.get("label_smoothing", 0.0)
    if type(ignore_index) is not int:
        raise ValueError("ignore_index must be an integer")
    if isinstance(label_smoothing, bool) or not isinstance(label_smoothing, (int, float)) or not 0 <= label_smoothing <= 1:
        raise ValueError("label_smoothing must be in [0,1]")
    return TokenCrossEntropy(ignore_index, float(label_smoothing))
