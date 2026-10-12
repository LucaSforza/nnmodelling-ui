return function(context,p,services)
  local x=context.inputs[1]
  if not x then return {status="unresolved",message="Input is not connected"} end
  local s=x.shape
  local loss={dtype="float32",shape={}}
  for i=1,p.times do
    local result=services.infer_subflow(x)
    if result.status~="success" then return result end
    x=result.outputs.out
    loss=result.outputs.loss
  end
  return {status="success",outputs={out=x,loss=loss}}
end
