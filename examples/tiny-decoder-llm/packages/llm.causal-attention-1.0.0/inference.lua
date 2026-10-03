return function(context)
  local x = context.inputs[1]
  local rank, err = tensor.rank(x)
  if err then return {status='error', message=err} end
  local width = tensor.dimension(x, -1)
  local dtype = tensor.dtype(x)
  if rank ~= 3 or width ~= 512 or dtype ~= 'float32' then
    return {status='error', message='Causal attention requires float32 [B,T,512], with 8 heads'}
  end
  return {status='success', output=x}
end