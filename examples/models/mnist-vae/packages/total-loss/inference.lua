return function(context, parameters, services)
  local reconstruction, kl = context.inputs[1], context.inputs[2]
  if #context.inputs ~= 2 or not reconstruction or not kl then
    return { status = "error", message = "VAE total loss requires reconstruction MSE then per-sample KL" }
  end
  local dtype = reconstruction.dtype
  if dtype ~= "float16" and dtype ~= "bfloat16" and dtype ~= "float32" and dtype ~= "float64" then
    return { status = "error", message = "VAE total loss requires floating tensors" }
  end
  if dtype ~= kl.dtype or #reconstruction.shape ~= 0 or #kl.shape ~= 1 then
    return { status = "error", message = "Expected scalar reconstruction MSE and compatible per-sample KL [B]" }
  end
  local output, err = tensor.create({}, dtype)
  if err then return { status = "error", message = err } end
  return { status = "success", output = output }
end
