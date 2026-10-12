return function(context, parameters, services)
  local input = context.inputs[1]
  if not input then return {status = "unresolved", message = "Input is missing"} end
  local max_length = parameters.max_length or 128
  if type(max_length) ~= "number" or max_length < 1 or max_length % 1 ~= 0 then
    return {status = "error", message = "Causal mask max_length must be a positive integer"}
  end
  local rank, rank_error = tensor.rank(input)
  if rank_error then return {status = "error", message = rank_error} end
  if rank ~= 3 then return {status = "error", message = "Causal mask expects floating square [B,T,T] scores"} end
  local dtype, dtype_error = tensor.dtype(input)
  if dtype_error then return {status = "error", message = dtype_error} end
  if dtype ~= "float16" and dtype ~= "bfloat16" and dtype ~= "float32" and dtype ~= "float64" then
    return {status = "error", message = "Causal mask expects floating scores"}
  end
  local rows, rows_error = tensor.dimension(input, 2)
  if rows_error then return {status = "error", message = rows_error} end
  local columns, columns_error = tensor.dimension(input, 3)
  if columns_error then return {status = "error", message = columns_error} end
  if rows ~= columns then return {status = "error", message = "Causal mask expects square [B,T,T] scores"} end
  if type(rows) == "number" and rows > max_length then
    return {status = "error", message = "Causal mask sequence length exceeds max_length"}
  end
  return {status = "success", output = input}
end
