"""Manifold-constrained hyper-connections, paper equations (1)-(8)."""
import torch
from torch import nn
from torch.nn import functional as F


class HyperConnection(nn.Module):
    def __init__(self, dim, streams=4, iterations=20):
        super().__init__()
        self.streams = streams
        self.iterations = iterations
        size = streams * (streams + 2)
        self.dynamic = nn.Parameter(torch.empty(size, streams * dim))
        self.base = nn.Parameter(torch.zeros(size))
        self.scale = nn.Parameter(torch.full((3,), .01))
        self.base._nnm_optimizer = self.scale._nnm_optimizer = 'adamw'
        nn.init.normal_(self.dynamic, std=(streams * dim) ** -.5)
        with torch.no_grad():
            # Start near the identity while remaining inside the Birkhoff polytope.
            self.base[2 * streams:].view(streams, streams).copy_(torch.eye(streams) * 4)

    def pre(self, state):
        n = self.streams
        flat = state.float().flatten(-2)
        normalized = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
        with torch.autocast(device_type=state.device.type, enabled=False):
            raw = F.linear(normalized, self.dynamic.float())
            pre = torch.sigmoid(raw[..., :n] * self.scale[0] + self.base[:n])
            post = 2 * torch.sigmoid(raw[..., n:2*n] * self.scale[1] + self.base[n:2*n])
            matrix = (raw[..., 2*n:] * self.scale[2] + self.base[2*n:]).unflatten(-1, (n,n))
            matrix = torch.exp(matrix - matrix.amax(dim=(-2,-1), keepdim=True))
            for _ in range(self.iterations):
                matrix = matrix / matrix.sum(-2, keepdim=True).clamp_min(1e-6)
                matrix = matrix / matrix.sum(-1, keepdim=True).clamp_min(1e-6)
        hidden = (pre.unsqueeze(-1) * state.float()).sum(-2)
        return hidden.to(state.dtype), post, matrix

    @staticmethod
    def post(state, hidden, post, matrix):
        residual = torch.einsum('...ij,...jd->...id', matrix, state.float())
        return (residual + post.unsqueeze(-1) * hidden.float().unsqueeze(-2)).to(state.dtype)

    def forward(self, state):
        return self.pre(state)[0]


def build(parameters, context, services):
    return HyperConnection(parameters['dim'], parameters.get('streams',4), parameters.get('iterations',20))
