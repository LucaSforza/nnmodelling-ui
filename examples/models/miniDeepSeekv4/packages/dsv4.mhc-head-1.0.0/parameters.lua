return function(context,p,services)
  local total=p.streams*p.streams*p.dim+p.streams+1
  return {status="success",total=total,trainable=total}
end
