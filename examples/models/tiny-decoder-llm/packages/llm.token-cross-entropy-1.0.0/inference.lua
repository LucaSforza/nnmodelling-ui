return function(context, parameters, services)
  local ignore_index = parameters.ignore_index
  if ignore_index == nil then ignore_index = -100 end
  local label_smoothing = parameters.label_smoothing
  if label_smoothing == nil then label_smoothing = 0 end
  if type(ignore_index) ~= "number" or ignore_index % 1 ~= 0 then
    return {status = "error", message = "Token Cross Entropy ignore_index must be an integer"}
  end
  if type(label_smoothing) ~= "number" or label_smoothing ~= label_smoothing or
     label_smoothing < 0 or label_smoothing > 1 then
    return {status = "error", message = "Token Cross Entropy label_smoothing must be in [0,1]"}
  end
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
