return function(context, p, services)
    if p.in_channels % p.groups ~= 0 then
        return {status = "error", message = "in_channels must be divisible by groups"}
    end
    local per_output = p.in_channels // p.groups
    if per_output > math.maxinteger // p.kernel_size then
        return {status = "error", message = "parameter count overflow"}
    end
    per_output = per_output * p.kernel_size
    if per_output > math.maxinteger // p.kernel_size then
        return {status = "error", message = "parameter count overflow"}
    end
    per_output = per_output * p.kernel_size
    if p.out_channels > math.maxinteger // per_output then
        return {status = "error", message = "parameter count overflow"}
    end
    local total = p.out_channels * per_output
    if p.bias then
        if total > math.maxinteger - p.out_channels then
            return {status = "error", message = "parameter count overflow"}
        end
        total = total + p.out_channels
    end
    return {status = "success", total = total, trainable = total}
end
