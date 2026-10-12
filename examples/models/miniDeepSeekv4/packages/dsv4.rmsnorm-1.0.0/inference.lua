return function(context,p,services)
  local x=context.inputs[1]
  if not x then return {status="unresolved",message="Input is not connected"} end
  local s=x.shape
  if #s<1 or s[#s]~=p.dim then return {status="error",message="RMSNorm last dimension must equal dim"} end
  return {status="success",output=x}
end
