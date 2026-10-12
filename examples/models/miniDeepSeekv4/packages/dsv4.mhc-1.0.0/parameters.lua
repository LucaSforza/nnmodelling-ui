return function(context,parameters,services)
  local n,d=parameters.streams,parameters.dim
  local total=(n*n+2*n)*(n*d+1)+3
  return {status="success",total=total,trainable=total}
end
