return function(context, parameters, services)
  local input = context.inputs[1]
  if not input then return {status = "unresolved", message = "Input is missing"} end
  return {status = "success", output = input}
end
