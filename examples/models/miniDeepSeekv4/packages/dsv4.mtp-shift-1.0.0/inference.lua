return function(context,p,services)
  local x=context.inputs[1]
  if not x then return {status="unresolved",message="Input is not connected"} end
  local s=x.shape
  if #s<3 or type(s[2])~="number" or s[2]<2 then return {status="error",message="MTP requires sequence length >=2"} end
  local shape={}
  for i,v in ipairs(s) do shape[i]=v end
  shape[2]=s[2]-1
  return {status="success",output={dtype=x.dtype,shape=shape}}
end
