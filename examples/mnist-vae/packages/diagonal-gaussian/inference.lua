return function(context, parameters, services)
  if #context.inputs ~= 2 then
    return { status = "error", message = "Gaussian requires ordered mean and log_variance inputs" }
  end
  local mean, variance = context.inputs[1], context.inputs[2]
  if #mean.shape ~= 2 or #variance.shape ~= 2 then
    return { status = "error", message = "Gaussian heads must have rank 2 [B,L]" }
  end
  local dtype = mean.dtype
  if dtype ~= "float16" and dtype ~= "bfloat16" and dtype ~= "float32" and dtype ~= "float64" then
    return { status = "error", message = "Gaussian requires floating tensors" }
  end
  if variance.dtype ~= dtype or variance.shape[1] ~= mean.shape[1]
      or variance.shape[2] ~= mean.shape[2] or mean.shape[2] ~= parameters.latent_features then
    return { status = "error", message = "Gaussian head shapes/dtypes must match latent_features" }
  end
  local output, err = tensor.create({mean.shape[1], 2, mean.shape[2]}, dtype)
  if err then return { status = "error", message = err } end
  return { status = "success", output = output }
end
