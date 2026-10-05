return function(context, parameters, services)
  local input = context.inputs[1]
  local rank, err = tensor.rank(input)
  if err then return {status = "error", message = err} end
  if rank ~= 3 then return {status = "error", message = "Causal attention expects [B,T,64]"} end
  local width, width_error = tensor.dimension(input, -1)
  if width_error then return {status = "error", message = width_error} end
  if width ~= 64 then return {status = "error", message = "Causal attention model width must be 64"} end
  return {status = "success", output = input}
end
