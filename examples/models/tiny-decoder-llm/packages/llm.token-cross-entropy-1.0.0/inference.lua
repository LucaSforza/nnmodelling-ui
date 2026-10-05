return function(context, parameters, services)
  local input = context.inputs[1]
  local rank, err = tensor.rank(input)
  if err then return {status = "error", message = err} end
  if rank ~= 3 then return {status = "error", message = "Token Cross Entropy expects logits [B,T,V]"} end
  local dtype, dtype_error = tensor.dtype(input)
  if dtype_error then return {status = "error", message = dtype_error} end
  if dtype ~= "float16" and dtype ~= "bfloat16" and dtype ~= "float32" and dtype ~= "float64" then
    return {status = "error", message = "Token Cross Entropy expects floating logits"}
  end
  local loss, loss_error = tensor.create({}, dtype)
  if loss_error then return {status = "error", message = loss_error} end
  return {status = "success", output = loss}
end
