import torch
from torch import nn
from torch.nn import functional as F

class Select(nn.Module):
    def __init__(self,n,d):
        super().__init__()
        self.start=n*d
        self.dim=d
    def forward(self,pack):
        return pack[...,self.start:self.start+self.dim]
def build(p,context,services):
    return Select(p['streams'],p['dim'])
