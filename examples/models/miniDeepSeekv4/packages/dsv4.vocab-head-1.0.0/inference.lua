return function(context,p,services)
  local x=context.inputs[1]
  if not x then return {status="unresolved",message="Input is not connected"} end
  local s=x.shape
  if #s~=4 or s[3]~=2 or s[4]~=p.dim then return {status="error",message="Expected [B,T,2,dim]"} end
  return {status="success",output={dtype=x.dtype,shape={s[1],s[2],2,p.vocab_size}}}
end
