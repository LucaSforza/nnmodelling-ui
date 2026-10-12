return function(context, parameters, services)
  local input = context.inputs[1]
  if not input then return {status="unresolved",message="Input is not connected"} end
  local shape = input.shape
  if #shape ~= 4 or shape[3] ~= 2 or not context.inputs[2] then return {status="error",message="Objective requires decoder logits and auxiliary loss"} end
  return {status="success",outputs={loss={dtype="float32",shape={}}}}
end
