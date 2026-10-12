return function(context,parameters,services)
  local p=parameters
  local total=p.dim*p.q_rank+p.q_rank+p.q_rank*p.heads*p.head_dim+p.dim*p.head_dim+p.head_dim+p.heads*p.head_dim*p.o_rank+p.o_groups*p.o_rank*p.dim+p.heads
  if p.ratio > 0 then
    local overlap=p.ratio==4 and 2 or 1
    total=total+2*p.dim*overlap*p.head_dim+p.ratio*overlap*p.head_dim+p.head_dim
    if p.ratio==4 then total=total+p.q_rank*p.index_heads*p.index_dim+p.dim*p.index_heads+4*p.dim*p.index_dim+2*p.ratio*p.index_dim+p.index_dim end
  end
  return {status="success",total=total,trainable=total}
end
