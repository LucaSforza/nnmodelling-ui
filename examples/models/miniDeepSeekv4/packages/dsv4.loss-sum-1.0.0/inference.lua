return function(context,p,services)
  if #context.inputs<1 then return {status="unresolved",message="Connect auxiliary losses"} end
  for _,x in ipairs(context.inputs) do
    if #x.shape~=0 or x.dtype~="float32" then return {status="error",message="Auxiliary losses must be FP32 scalars"} end
  end
  return {status="success",output={dtype="float32",shape={}}}
end
