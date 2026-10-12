return function(context,p,services)
  local x=context.inputs[1]
  if not x then return {status="unresolved",message="Input is not connected"} end
  local s=x.shape
  local y=context.inputs[2]
  if not y then return {status="unresolved",message="Connect main and MTP features"} end
  if #s~=3 or s[3]~=p.dim or #y.shape~=3 or y.shape[1]~=s[1] or y.shape[2]~=s[2]-1 or y.shape[3]~=s[3] then return {status="error",message="Invalid main/MTP feature stack"} end
  return {status="success",output={dtype=x.dtype,shape={s[1],s[2],2,p.dim}}}
end
