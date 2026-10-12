import torch
from torch import nn
from torch.nn import functional as F

class Head(nn.Module):
    def __init__(self,n,d):
        super().__init__()
        self.dynamic=nn.Parameter(torch.empty(n,n*d))
        self.base=nn.Parameter(torch.zeros(n))
        self.scale=nn.Parameter(torch.full((1,),.01))
        self.base._nnm_optimizer=self.scale._nnm_optimizer='adamw'
        nn.init.normal_(self.dynamic,std=(n*d)**-.5)
    def forward(self,state):
        flat=state.float().flatten(-2)
        flat=flat*torch.rsqrt(flat.square().mean(-1,keepdim=True)+1e-6)
        with torch.autocast(device_type=state.device.type,enabled=False):
            weights=torch.sigmoid(F.linear(flat,self.dynamic.float())*self.scale+self.base)
        return (state.float()*weights.unsqueeze(-1)).sum(-2).to(state.dtype)
def build(p,context,services):
    return Head(p['streams'],p['dim'])
