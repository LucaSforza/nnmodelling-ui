return function(context, parameters, services)
  local input = context.inputs[1]
  if not input then return {status="unresolved",message="Input is not connected"} end
  local shape = input.shape
  if #shape ~= 4 or shape[3] ~= 2 then return {status="error",message="Expected [B,T,2,V] logits"} end
  return {status="success",output={dtype=input.dtype,shape={shape[1],shape[2],shape[4]}}}
end
