import torch
from torch import nn
from torch.nn import functional as F

def build(p,context,services):
    return services.build_stereotype({'id':'dsv4.attention','version':'1.0.0','parameters':p})
