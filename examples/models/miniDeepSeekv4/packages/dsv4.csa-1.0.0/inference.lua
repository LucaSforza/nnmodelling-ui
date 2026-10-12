return function(context, parameters, services)
  local input = context.inputs[1]
  if not input then return {status="unresolved",message="Input is not connected"} end
  local shape = input.shape
  if #shape ~= 3 or shape[3] ~= parameters.dim then return {status="error",message="Expected hidden [B,T,dim]"} end
  return {status="success",outputs={out=input,loss={dtype="float32",shape={}}}}
end
