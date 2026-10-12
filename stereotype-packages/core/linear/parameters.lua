return function(context, p, services)
    if p.in_features > math.maxinteger // p.out_features then
        return {status = "error", message = "parameter count overflow"}
    end
    local total = p.in_features * p.out_features
    if p.bias then
        if total > math.maxinteger - p.out_features then
            return {status = "error", message = "parameter count overflow"}
        end
        total = total + p.out_features
    end
    return {status = "success", total = total, trainable = total}
end
