return function(context, parameters, services)
  local input = context.inputs[1]
  local rank, err = tensor.rank(input)
  if err then return {status = 'error', message = err} end
  if rank ~= 3 then return {status = 'error', message = 'RNN expects rank 3 [B,T,F]'} end
  local dtype, dtype_error = tensor.dtype(input)
  if dtype_error then return {status = 'error', message = dtype_error} end
  if dtype ~= 'float32' or dtype ~= parameters.dtype then return {status = 'error', message = 'RNN expects float32 input and weights'} end
  for _, key in ipairs({'sequence_length', 'input_size', 'hidden_size'}) do
    local value = parameters[key]
    if type(value) ~= 'number' or value < 1 or value % 1 ~= 0 then return {status = 'error', message = 'RNN requires positive integer ' .. key} end
  end
  if parameters.nonlinearity ~= 'tanh' or parameters.initial_state ~= 'zero' then return {status = 'error', message = 'RNN requires tanh and zero initial state'} end
  if type(parameters.bias) ~= 'boolean' then return {status = 'error', message = 'RNN bias must be boolean'} end
  local batch, batch_error = tensor.dimension(input, 1)
  if batch_error then return {status = 'error', message = batch_error} end
  if batch ~= 'B' and (type(batch) ~= 'number' or batch < 1 or batch % 1 ~= 0) then return {status = 'error', message = 'RNN requires B or a positive batch dimension'} end
  local steps, steps_error = tensor.dimension(input, 2)
  if steps_error then return {status = 'error', message = steps_error} end
  if steps ~= parameters.sequence_length then return {status = 'error', message = 'RNN sequence length does not match sequence_length'} end
  local features, features_error = tensor.dimension(input, 3)
  if features_error then return {status = 'error', message = features_error} end
  if features ~= parameters.input_size then return {status = 'error', message = 'RNN feature dimension does not match input_size'} end
  local output, output_error = tensor.create({batch, parameters.hidden_size}, dtype)
  if output_error then return {status = 'error', message = output_error} end
  return {status = 'success', output = output}
end
