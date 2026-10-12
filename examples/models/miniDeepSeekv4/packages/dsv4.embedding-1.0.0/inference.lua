return function(context,p,services)
  local x=context.inputs[1]
  if not x then return {status="unresolved",message="Input is not connected"} end
  local s=x.shape
  if #s~=2 or x.dtype~="int64" then return {status="error",message="Expected int64 [B,T] tokens"} end
  return {status="success",output={dtype="float32",shape={s[1],s[2],p.dim}}}
end
