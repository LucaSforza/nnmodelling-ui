import torch
from torch.nn import functional as F


class Objective(torch.nn.Module):
    def forward(self,logits,auxiliary,targets):
        main = F.cross_entropy(logits[:,:,0,:].float().flatten(0,1),targets.flatten())
        mtp = F.cross_entropy(logits[:,:-1,1,:].float().flatten(0,1),targets[:,1:].flatten()) if targets.shape[1]>1 else main.new_zeros(())
        return main+.3*mtp+auxiliary


def build(parameters,context,services):
    return Objective()
