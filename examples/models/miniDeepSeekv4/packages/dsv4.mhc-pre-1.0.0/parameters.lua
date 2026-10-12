return function(context,p,services)
  local total=(p.streams*p.streams+2*p.streams)*(p.streams*p.dim+1)+3
  return {status="success",total=total,trainable=total}
end
