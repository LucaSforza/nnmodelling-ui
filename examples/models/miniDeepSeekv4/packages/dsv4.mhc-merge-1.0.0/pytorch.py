import torch
from torch import nn
from torch.nn import functional as F

class Merge(nn.Module):
    def __init__(self,n,d):
        super().__init__()
        self.n,self.d=n,d
    def forward(self,pack,hidden):
        n,d=self.n,self.d
        state=pack[...,:n*d].unflatten(-1,(n,d))
        post=pack[...,n*d+d:n*d+d+n]
        matrix=pack[...,n*d+d+n:].unflatten(-1,(n,n))
        residual=torch.einsum('...ij,...jd->...id',matrix.float(),state.float())
        return (residual+post.float().unsqueeze(-1)*hidden.float().unsqueeze(-2)).to(pack.dtype)
def build(p,context,services):
    return Merge(p['streams'],p['dim'])
