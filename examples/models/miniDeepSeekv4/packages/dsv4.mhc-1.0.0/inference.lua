return function(context, parameters, services)
  local input = context.inputs[1]
  if not input then return {status="unresolved",message="Input is not connected"} end
  local shape = input.shape
  if #shape ~= 4 or shape[3] ~= parameters.streams or shape[4] ~= parameters.dim then return {status="error",message="mHC requires [B,T,streams,dim]"} end
  return {status="success",output={dtype=input.dtype,shape={shape[1],shape[2],parameters.dim}}}
end
