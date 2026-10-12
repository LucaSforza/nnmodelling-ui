import torch
from torch import nn
from torch.nn import functional as F

class Repeat(nn.Module):
    def __init__(self,bodies):
        super().__init__()
        self.bodies=nn.ModuleList(bodies)
    def forward(self,state):
        loss=state.new_zeros((),dtype=torch.float32)
        for body in self.bodies:
            result=body(state)
            if isinstance(result,dict):
                state=result['out']
                loss=loss+result['loss']
            else:
                state=result
        return {'out':state,'loss':loss}
def build(p,context,services):
    return Repeat([services.build_subflow() for _ in range(p['times'])])
