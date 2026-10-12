return function(context,p,services)
  local x=context.inputs[1]
  if not x then return {status="unresolved",message="Input is not connected"} end
  local s=x.shape
  local y=context.inputs[2]
  if not y then return {status="unresolved",message="Connect projected embeddings and residual streams"} end
  if #s~=3 or s[3]~=p.dim or #y.shape~=4 or y.shape[1]~=s[1] or y.shape[2]~=s[2] or y.shape[3]~=p.streams or y.shape[4]~=p.dim then return {status="error",message="Invalid MTP fusion"} end
  return {status="success",output=y}
end
