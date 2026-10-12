return function(context, p, services)
    local total = p.elementwise_affine and (2 * p.normalized_shape) or 0
    return {status = "success", total = total, trainable = total}
end
