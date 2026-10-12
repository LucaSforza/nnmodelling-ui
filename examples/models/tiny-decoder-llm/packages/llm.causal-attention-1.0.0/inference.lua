return function(context, parameters, services)
  local input = context.inputs[1]
  local d_model = parameters.d_model or 64
  local num_heads = parameters.num_heads or 4
  local dropout = parameters.dropout
  if dropout == nil then dropout = 0 end
  local bias = parameters.bias
  if bias == nil then bias = true end
  if type(d_model) ~= "number" or d_model < 1 or d_model % 1 ~= 0 then
    return {status = "error", message = "Causal attention requires positive integer d_model"}
  end
  if type(num_heads) ~= "number" or num_heads < 1 or num_heads % 1 ~= 0 then
    return {status = "error", message = "Causal attention requires positive integer num_heads"}
  end
  if d_model % num_heads ~= 0 then
    return {status = "error", message = "Causal attention d_model must be divisible by num_heads"}
  end
  if type(dropout) ~= "number" or dropout ~= dropout or dropout < 0 or dropout > 1 then
    return {status = "error", message = "Causal attention dropout must be in [0,1]"}
  end
  if type(bias) ~= "boolean" then
    return {status = "error", message = "Causal attention bias must be boolean"}
  end
  local rank, err = tensor.rank(input)
  if err then return {status = "error", message = err} end
  if rank ~= 3 then return {status = "error", message = "Causal attention expects floating [B,T,d_model]"} end
  local dtype, dtype_error = tensor.dtype(input)
  if dtype_error then return {status = "error", message = dtype_error} end
  if dtype ~= "float16" and dtype ~= "bfloat16" and dtype ~= "float32" and dtype ~= "float64" then
    return {status = "error", message = "Causal attention expects floating input"}
  end
  local width, width_error = tensor.dimension(input, -1)
  if width_error then return {status = "error", message = width_error} end
  if width ~= d_model then return {status = "error", message = "Causal attention model width must match d_model"} end
  return {status = "success", output = input}
end
