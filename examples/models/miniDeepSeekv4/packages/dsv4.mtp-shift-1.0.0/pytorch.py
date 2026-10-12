import torch
from torch import nn
from torch.nn import functional as F

class Shift(nn.Module):
    def __init__(self,mode):
        super().__init__()
        self.mode=mode
    def forward(self,x):
        return x[:,1:] if self.mode=='embedding' else x[:,:-1]
def build(p,context,services):
    return Shift(p['mode'])
