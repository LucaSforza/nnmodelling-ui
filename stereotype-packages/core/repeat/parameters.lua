return function(context, p, services)
    local body = services.count_subflow()
    if body.total > math.maxinteger // p.times or
       body.trainable > math.maxinteger // p.times then
        return {status = "error", message = "parameter count overflow"}
    end
    local total = body.total * p.times
    local trainable = body.trainable * p.times
    return {status = "success", total = total, trainable = trainable,
            complete = body.complete, message = body.message}
end
