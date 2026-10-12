return function(context, p, services)
    if p.num_embeddings > math.maxinteger // p.embedding_dim then
        return {status = "error", message = "parameter count overflow"}
    end
    local total = p.num_embeddings * p.embedding_dim
    return {status = "success", total = total, trainable = total}
end
