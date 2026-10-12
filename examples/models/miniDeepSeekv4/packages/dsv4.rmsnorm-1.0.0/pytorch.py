import torch
from torch import nn
from torch.nn import functional as F

class RMSNorm(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(d))
        self.weight._nnm_optimizer='adamw'
    def forward(self,x):
        y=x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)
        return (y*self.weight.float()).to(x.dtype)
def build(p,context,services):
    return RMSNorm(p['dim'])
