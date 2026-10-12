import torch


class MainLogits(torch.nn.Module):
    def forward(self,logits):
        return logits[:,:,0,:]


def build(parameters,context,services):
    return MainLogits()
