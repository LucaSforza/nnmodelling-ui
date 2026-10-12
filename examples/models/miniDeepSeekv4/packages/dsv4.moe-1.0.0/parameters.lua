return function(context,parameters,services)
  local total=3*parameters.dim*parameters.hidden*(parameters.experts+1)+parameters.dim*parameters.experts
  return {status="success",total=total,trainable=total}
end
