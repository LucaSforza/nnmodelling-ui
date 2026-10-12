return function(context,p,services)
  local x=context.inputs[1]
  if not x then return {status="unresolved",message="Input is not connected"} end
  local s=x.shape
  if #s~=3 or s[3]~=p.dim then return {status="error",message="Expected [B,T,dim]"} end
  return {status="success",output={dtype=x.dtype,shape={s[1],s[2],p.streams,p.dim}}}
end
