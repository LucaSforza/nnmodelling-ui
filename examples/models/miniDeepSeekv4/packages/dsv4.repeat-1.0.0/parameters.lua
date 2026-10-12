return function(context,p,services)
  local body=services.count_subflow()
  if body.total>math.maxinteger//p.times or body.trainable>math.maxinteger//p.times then return {status="error",message="Parameter count overflow"} end
  return {status="success",total=body.total*p.times,trainable=body.trainable*p.times,complete=body.complete,message=body.message}
end
