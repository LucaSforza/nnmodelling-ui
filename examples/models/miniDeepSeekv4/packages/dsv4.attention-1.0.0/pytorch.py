"""Differentiable CSA/HCA shared-KV attention (§2.3), without quantized kernels."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.weight._nnm_optimizer = 'adamw'

    def forward(self,x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)
        return (y*self.weight.float()).to(x.dtype)


def rotary(x,positions,dim=64,inverse=False,base=40000.):
    dim = min(dim, x.shape[-1])
    if dim % 2:
        raise ValueError("Rotary dimension must be even")
    frequencies = base ** (-torch.arange(0,dim,2,device=x.device,dtype=torch.float32)/dim)
    phase = positions.float()[:,None] * frequencies[None,:] * (-1 if inverse else 1)
    shape = [1,len(positions)] + [1]*(x.ndim-3) + [dim//2]
    cosine,sine = phase.cos().view(shape),phase.sin().view(shape)
    pairs = x[..., -dim:].float().unflatten(-1,(-1,2))
    a,b = pairs.unbind(-1)
    rotated = torch.stack((a*cosine-b*sine,a*sine+b*cosine),-1).flatten(-2)
    return torch.cat((x[..., :-dim],rotated.to(x.dtype)),-1)


class Compressor(nn.Module):
    def __init__(self,dim,head_dim,ratio):
        super().__init__()
        self.ratio = ratio
        self.overlap = ratio == 4
        channels = head_dim*(2 if self.overlap else 1)
        self.values = nn.Linear(dim,channels,bias=False)
        self.gates = nn.Linear(dim,channels,bias=False)
        self.position = nn.Parameter(torch.zeros(ratio,channels))
        self.norm = RMSNorm(head_dim)

    def forward(self,x):
        b,t,_ = x.shape
        r = self.ratio
        count = t//r
        if not count:
            return x.new_empty(b,0,self.norm.weight.numel())
        with torch.autocast(device_type=x.device.type,enabled=False):
            values = F.linear(x[:,:count*r].float(),self.values.weight.float()).unflatten(1,(count,r))
            gates = F.linear(x[:,:count*r].float(),self.gates.weight.float()).unflatten(1,(count,r))+self.position
            if self.overlap:
                # Current normal channels plus previous overlapping channels (eq.11-12).
                previous_values = F.pad(values[:,:-1,:,:values.shape[-1]//2],(0,0,0,0,1,0))
                previous_gates = F.pad(gates[:,:-1,:,:gates.shape[-1]//2],(0,0,0,0,1,0),value=-float('inf'))
                values = torch.cat((values[...,values.shape[-1]//2:],previous_values),dim=2)
                gates = torch.cat((gates[...,gates.shape[-1]//2:],previous_gates),dim=2)
            pooled = (values*gates.softmax(dim=2)).sum(2)
        normalized = self.norm(pooled.to(x.dtype))
        return rotary(normalized,torch.arange(count,device=x.device)*r)


class Attention(nn.Module):
    def __init__(self,dim,heads,head_dim,q_rank,o_groups,o_rank,ratio,index_heads=4,index_dim=64,topk=64,window=128):
        super().__init__()
        self.heads,self.head_dim = heads,head_dim
        self.groups,self.o_rank = o_groups,o_rank
        self.ratio,self.topk,self.window = ratio,topk,window
        self.q_down = nn.Linear(dim,q_rank,bias=False)
        self.q_norm = RMSNorm(q_rank)
        self.q_up = nn.Linear(q_rank,heads*head_dim,bias=False)
        self.kv = nn.Linear(dim,head_dim,bias=False)
        self.kv_norm = RMSNorm(head_dim)
        self.group_out = nn.Parameter(torch.empty(o_groups,o_rank,heads*head_dim//o_groups))
        nn.init.normal_(self.group_out,std=(heads*head_dim//o_groups)**-.5)
        self.out = nn.Linear(o_groups*o_rank,dim,bias=False)
        self.sinks = nn.Parameter(torch.zeros(heads))
        self.index_heads,self.index_dim = index_heads,index_dim
        if ratio:
            self.compressor = Compressor(dim,head_dim,ratio)
        if ratio == 4:
            self.index_queries = nn.Linear(q_rank,index_heads*index_dim,bias=False)
            self.index_weights = nn.Linear(dim,index_heads,bias=False)
            self.index_compressor = Compressor(dim,index_dim,ratio)

    def forward(self,x):
        b,t,_ = x.shape
        if t == 0:
            return {'out':x,'loss':x.new_zeros((),dtype=torch.float32)}
        positions = torch.arange(t,device=x.device)
        qr = self.q_norm(self.q_down(x))
        q = self.q_up(qr).view(b,t,self.heads,self.head_dim)
        q = (q.float()*torch.rsqrt(q.float().square().mean(-1,keepdim=True)+1e-6)).to(x.dtype)
        base = 40000. if self.ratio else 10000.
        q = rotary(q,positions,base=base)
        window_kv = rotary(self.kv_norm(self.kv(x)),positions,base=base)
        allowed = (positions[None,:] <= positions[:,None]) & (positions[None,:] > positions[:,None]-self.window)
        allowed = allowed[None,:,:].expand(b,-1,-1)
        kv = window_kv
        index_loss = x.new_zeros((),dtype=torch.float32)
        if self.ratio:
            compressed = self.compressor(x)
            count = compressed.shape[1]
            kv = torch.cat((window_kv,compressed),1)
            # A block is usable only after its final token; future perturbations cannot leak.
            complete = (torch.arange(count,device=x.device)+1)*self.ratio <= positions[:,None]
            compressed_allowed = complete[None,:,:].expand(b,-1,-1)
            if self.ratio == 4 and count:
                index_keys = self.index_compressor(x)
                index_q = rotary(self.index_queries(qr).view(b,t,self.index_heads,self.index_dim),positions)
                weights = self.index_weights(x).float() / math.sqrt(self.index_dim*self.index_heads)
                score = (torch.einsum('bthd,bnd->bthn',index_q.float(),index_keys.float()).relu()*weights.unsqueeze(-1)).sum(2)
                masked = score.masked_fill(~compressed_allowed,-float('inf'))
                selected = masked.topk(min(count,self.topk),dim=-1).indices
                sparse = torch.zeros_like(compressed_allowed).scatter_(-1,selected,True)
                if self.training:
                    with torch.no_grad():
                        teacher = torch.einsum('bthd,bnd->bhtn',q.float(),compressed.float()) / math.sqrt(self.head_dim)
                        teacher = teacher.masked_fill(~complete[None,None,:,:],-1e30).softmax(-1).mean(1)
                    valid = complete.any(-1)
                    if valid.any():
                        student = masked[:,valid,:].log_softmax(-1)
                        target = teacher[:,valid,:]
                        # Masking zero-mass terms avoids 0*(-inf) in KL.
                        safe_student = student.masked_fill(~complete[valid][None,:,:],0)
                        index_loss = F.kl_div(safe_student,target,reduction='none').sum(-1).mean()
                compressed_allowed = compressed_allowed & sparse
            allowed = torch.cat((allowed,compressed_allowed),-1)
        logits = torch.einsum('bthd,bnd->bhtn',q.float(),kv.float()) / math.sqrt(self.head_dim)
        logits = logits.masked_fill(~allowed[:,None,:,:],-float('inf'))
        sink = self.sinks.float()[None,:,None,None].expand(b,-1,t,1)
        probabilities = torch.cat((logits,sink),-1).softmax(-1)[...,:-1]
        output = torch.einsum('bhtn,bnd->bthd',probabilities,kv.float()).to(x.dtype)
        output = rotary(output,positions,inverse=True,base=base).view(b,t,self.groups,-1)
        projected = torch.einsum('btgd,grd->btgr',output,self.group_out).flatten(-2)
        return {'out':self.out(projected),'loss':index_loss}


def build(parameters, context, services):
    return Attention(**{key:parameters[key] for key in ['dim','heads','head_dim','q_rank','o_groups','o_rank','ratio','index_heads','index_dim','topk','window']})
