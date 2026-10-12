return function(context,p,services)
  local x=context.inputs[1]
  if not x then return {status="unresolved",message="Input is not connected"} end
  local s=x.shape
  local y=context.inputs[2]
  if not y then return {status="unresolved",message="Connect mHC pack and sublayer output"} end
  local n,d=p.streams,p.dim
  if #s~=3 or s[3]~=n*d+d+n+n*n or #y.shape~=3 or y.shape[1]~=s[1] or y.shape[2]~=s[2] or y.shape[3]~=d then return {status="error",message="Incompatible mHC merge inputs"} end
  return {status="success",output={dtype=x.dtype,shape={s[1],s[2],n,d}}}
end
