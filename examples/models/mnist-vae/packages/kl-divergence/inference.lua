return function(context, parameters, services)
  local input = context.inputs[1]
  if not input then return { status = "unresolved", message = "Latent distribution missing" } end
  local dtype = input.dtype
  if dtype ~= "float16" and dtype ~= "bfloat16" and dtype ~= "float32" and dtype ~= "float64" then
    return { status = "error", message = "KL requires floating Gaussian parameters" }
  end
  if #input.shape ~= 3 or input.shape[2] ~= 2 or input.shape[3] ~= parameters.latent_features then
    return { status = "error", message = "Expected Gaussian parameters [B,2,latent_features]" }
  end
  local output, err = tensor.create({input.shape[1]}, dtype)
  if err then return { status = "error", message = err } end
  return { status = "success", output = output }
end
