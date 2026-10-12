return function(context,p,services)
  local total=p.vocab_size*p.dim
  return {status="success",total=total,trainable=total}
end
