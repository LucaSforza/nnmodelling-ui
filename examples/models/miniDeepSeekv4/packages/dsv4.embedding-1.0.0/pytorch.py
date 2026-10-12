import torch
from torch import nn
from torch.nn import functional as F

def build(p,context,services):
    module=nn.Embedding(p['vocab_size'],p['dim'])
    module.weight._nnm_optimizer='adamw'
    nn.init.normal_(module.weight,std=.02)
    return module
