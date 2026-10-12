return function(context,p,services)
  local total=p.dim
  return {status="success",total=total,trainable=total}
end
