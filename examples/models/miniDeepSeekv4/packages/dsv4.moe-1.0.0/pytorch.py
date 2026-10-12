"""Fine-grained SwiGLU MoE with hash/learned sqrt-softplus routing (§2.1)."""
import torch
from torch import nn
from torch.nn import functional as F


class Expert(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoE(nn.Module):
    def __init__(self, dim, hidden, experts, topk, hash_routing, route_scale=1.):
        super().__init__()
        self.router = nn.Linear(dim, experts, bias=False)
        self.experts = nn.ModuleList(Expert(dim, hidden) for _ in range(experts))
        self.shared = Expert(dim, hidden)
        self.topk = topk
        self.hash_routing = hash_routing
        self.route_scale = route_scale
        # Selection bias is updated by load, never differentiated through routing.
        self.register_buffer('routing_bias', torch.zeros(experts))

    def forward(self, hidden, token_ids):
        b,t,d = hidden.shape
        if t == 0:
            return {'out':hidden,'loss':hidden.new_zeros((),dtype=torch.float32)}
        x = hidden.reshape(-1,d)
        with torch.autocast(device_type=x.device.type, enabled=False):
            scores = F.softplus(F.linear(x.float(),self.router.weight.float())).sqrt()
        if self.hash_routing:
            # Stable deterministic token-ID hash with distinct experts per top-k slot.
            base = (token_ids.reshape(-1).long() * 2654435761 + 1013904223) % len(self.experts)
            chosen = (base[:,None] + torch.arange(self.topk,device=x.device)) % len(self.experts)
        else:
            chosen = (scores + self.routing_bias).topk(self.topk,dim=-1).indices
        weights = scores.gather(-1,chosen)
        weights = weights / weights.sum(-1,keepdim=True).clamp_min(1e-12) * self.route_scale
        output = self.shared(x).float()
        for i,expert in enumerate(self.experts):
            rows,slots = torch.where(chosen == i)
            if rows.numel():
                values = expert(x.index_select(0,rows)).float() * weights[rows,slots,None]
                output = output.index_add(0,rows,values)
        normalized = scores / scores.sum(-1,keepdim=True).clamp_min(1e-12)
        frequency = F.one_hot(chosen,len(self.experts)).float().view(b,t,self.topk,-1).mean((1,2))
        probability = normalized.view(b,t,-1).mean(1)
        balance = (frequency * probability).sum(-1).mean() * len(self.experts) * .0001
        if self.training and not self.hash_routing:
            with torch.no_grad():
                counts = torch.bincount(chosen.flatten(),minlength=len(self.experts)).float()
                self.routing_bias.add_(.001 * (counts.mean()-counts).sign())
                self.routing_bias.sub_(self.routing_bias.mean())
        return {'out':output.to(hidden.dtype).view(b,t,d),'loss':balance}


def build(parameters, context, services):
    return MoE(parameters['dim'],parameters['hidden'],parameters['experts'],parameters['topk'],parameters.get('hash_routing',False),parameters.get('route_scale',1.))
