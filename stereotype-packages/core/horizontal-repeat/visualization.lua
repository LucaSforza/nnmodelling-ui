return function(parameters)
  local count = parameters.times or 2
  local join = parameters.join
  if count < 2 or count > 4096 or count ~= math.floor(count) then
    error("horizontal repeat count is outside the visualization limit")
  end
  if type(join) ~= "table" or type(join.id) ~= "string" or type(join.version) ~= "string" then
    error("horizontal repeat requires a valid join reference")
  end
  local nodes, edges = {}, {}
  for i = 1, count do
    local id = "body-" .. i
    nodes[#nodes + 1] = {id = id, body = true, label = "Branch " .. i .. "/" .. count}
    edges[#edges + 1] = {source = "$input", sourceHandle = "out", target = id, targetHandle = "in"}
  end
  nodes[#nodes + 1] = {
    id = "joined",
    package = {id = join.id, version = join.version},
    parameters = join.parameters or {},
    label = "Join"
  }
  for i = 1, count do
    edges[#edges + 1] = {
      source = "body-" .. i,
      sourceHandle = "out",
      target = "joined",
      targetHandle = "in-" .. i
    }
  end
  return {
    nodes = nodes,
    edges = edges,
    outputs = {out = {node = "joined", handle = "out"}}
  }
end
