import torch
from torch import nn
from torch.nn import functional as F

class Expand(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.n=n
    def forward(self,x):
        return x.unsqueeze(-2).expand(-1,-1,self.n,-1)
def build(p,context,services):
    return Expand(p['streams'])
