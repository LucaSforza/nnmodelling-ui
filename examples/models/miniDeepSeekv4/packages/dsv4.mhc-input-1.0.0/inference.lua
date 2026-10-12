return function(context,p,services)
  local x=context.inputs[1]
  if not x then return {status="unresolved",message="Input is not connected"} end
  local s=x.shape
  local n,d=p.streams,p.dim
  if #s~=3 or s[3]~=n*d+d+n+n*n then return {status="error",message="Invalid mHC pack"} end
  return {status="success",output={dtype=x.dtype,shape={s[1],s[2],d}}}
end
