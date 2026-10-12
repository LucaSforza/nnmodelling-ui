return function(context, p, services)
    local total = p.affine and (2 * p.num_features) or 0
    return {status = "success", total = total, trainable = total}
end
