return function(context, parameters, services)
  local input = context.inputs[1]
  if not input then return { status = "unresolved", message = "Decoder input missing" } end
  local dtype = input.dtype
  if dtype ~= "float16" and dtype ~= "bfloat16" and dtype ~= "float32" and dtype ~= "float64" then
    return { status = "error", message = "Sigmoid requires a floating tensor" }
  end
  return { status = "success", output = input }
end
