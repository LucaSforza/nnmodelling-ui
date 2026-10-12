import torch
from torch import nn
from torch.nn import functional as F

class Fusion(nn.Module):
    def forward(self,embedded,state):
        return embedded.unsqueeze(-2)+state
def build(p,context,services):
    return Fusion()
