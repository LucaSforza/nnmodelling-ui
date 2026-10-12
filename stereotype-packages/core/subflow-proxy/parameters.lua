return function(context, p, services)
    local body = services.count_subflow()
    return {status = "success", total = body.total, trainable = body.trainable,
            complete = body.complete, message = body.message}
end
