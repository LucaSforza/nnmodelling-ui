import torch
from torch import nn
from torch.nn import functional as F

class Pre(nn.Module):
    def __init__(self,hc):
        super().__init__()
        self.hc=hc
    def forward(self,state):
        hidden,post,matrix=self.hc.pre(state)
        # Ordinary graph tensor carries every value needed by the matching merge.
        return torch.cat((state.flatten(-2),hidden,post.to(state.dtype),matrix.flatten(-2).to(state.dtype)),dim=-1)
def build(p,context,services):
    return Pre(services.build_stereotype({'id':'dsv4.mhc','version':'1.0.0','parameters':p}))
