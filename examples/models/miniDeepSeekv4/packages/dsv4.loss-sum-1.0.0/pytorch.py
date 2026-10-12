import torch
from torch import nn
from torch.nn import functional as F

class Sum(nn.Module):
    def forward(self,*losses):
        return torch.stack(losses).sum()
def build(p,context,services):
    return Sum()
