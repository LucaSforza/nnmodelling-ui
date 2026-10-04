return function(parameters)
  local count = parameters.times or 1
  if count < 1 or count > 4096 or count ~= math.floor(count) then
    error("repeat count is outside the visualization limit")
  end
  local nodes, edges = {}, {}
  for i = 1, count do
    local id = "body-" .. i
    nodes[#nodes + 1] = {id = id, body = true, label = "Body " .. i .. "/" .. count}
    if i == 1 then
      edges[#edges + 1] = {source = "$input", sourceHandle = "out", target = id, targetHandle = "in"}
    else
      edges[#edges + 1] = {source = "body-" .. (i - 1), sourceHandle = "out", target = id, targetHandle = "in"}
    end
  end
  return {
    nodes = nodes,
    edges = edges,
    outputs = {out = {node = "body-" .. count, handle = "out"}}
  }
end
