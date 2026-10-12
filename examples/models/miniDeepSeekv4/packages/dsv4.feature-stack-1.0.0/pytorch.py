import torch
from torch import nn
from torch.nn import functional as F

class Stack(nn.Module):
    def forward(self,main,mtp):
        return torch.stack((main,F.pad(mtp,(0,0,0,1))),dim=2)
def build(p,context,services):
    return Stack()
