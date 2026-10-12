return function(context, p, services)
    local body = services.count_subflow()
    local join = services.count_stereotype(p.join)
    if body.total > math.maxinteger // p.times or
       body.trainable > math.maxinteger // p.times then
        return {status = "error", message = "parameter count overflow"}
    end
    local repeated_total = body.total * p.times
    local repeated_trainable = body.trainable * p.times
    if repeated_total > math.maxinteger - join.total or
       repeated_trainable > math.maxinteger - join.trainable then
        return {status = "error", message = "parameter count overflow"}
    end
    local total = repeated_total + join.total
    local trainable = repeated_trainable + join.trainable
    return {status = "success", total = total, trainable = trainable,
            complete = body.complete and join.complete}
end
