# Expanded 3D network explorer

Accepted 2026-10-04 from explicit user requirements. This is a read-only client
explorer, independent of numerical execution. All scopes and scripted occurrences
are expanded; no collapse or scope isolation in 3D. Filters between occurrences,
LoRA placement and backend execution are future use cases, not this milestone.

## Ownership and presentation

`src/visualization/` is C11 in nnmodelling_core. It owns disposable expanded
scene, validation, deterministic layout, grouping, occurrence identity, camera,
perspective projection, depth ordering and picking. Qt only translates widget
input to C calls and paints projected polygons/lines/text in a Network 3D tab.
Use software perspective projection and Qt QPainter: no GL context, external
renderer, duplicate model, or Qt dependency in core. C snapshots are copied into
owned scene strings before return. Build and exploration never mutate NNModel,
positions, history, analysis or dirty state. Scene and camera are never persisted.
Rebuild on activation after graph edits; failed build clears stale scene and
shows meaningful English error. 2D editor remains available in its own tab.

Operators are colored extruded cards with node label and the same definition-
positioned parameter rows as 2D. Geometry is schematic, not tensor-size scaling;
symbolic batch is never rendered. Directed connections preserve output/loss
classification and handle identity/order. Frame/wire outlines and labels show
nested group membership without hiding interior operators. Each repeated body
has its own labeled group. Selection shows full occurrence path, source node ID
and package; synthetic recipe operators identify their subflow owner. No shape
is copied from the last inference invocation and falsely labeled per occurrence.
Distance-dependent C label visibility avoids overlapping text in overview while
all operator geometry remains visible. Whole-network Fit and Home/start restore navigability. Mouse look/flight and
keyboard movement explore 3D; controls and current selection remain discoverable.

## Generic Lua composition

Optional schema-1 manifest entrypoint:
`entrypoints.visualization = {language: "lua", file: "visualization.lua"}`.
Allowed only for kind=subflow; paths use catalog confinement, regular-file and
size checks. It is distinct from inference and never called by type analysis.
No package-ID switches choose expansion. Missing entrypoint means one body
instance, connected to owner input and mapped outputs. Explicit scripts return
a function `function(parameters)` returning the following declarative plan:

```lua
return function(parameters)
  return {
    nodes = {
      {id = "first", body = true, label = "Block 1"},
      {id = "second", body = true, label = "Block 2"}
    },
    edges = {
      {source = "$input", sourceHandle = "out", target = "first", targetHandle = "in"},
      {source = "first", sourceHandle = "out", target = "second", targetHandle = "in"}
    },
    outputs = {out = {node = "second", handle = "out"}}
  }
end
```

A body instance recursively expands the owner's real immediate child graph.
Its `in` maps to its unique Input, and its outputs map to typed terminals by
boundaryHandle. `$input` is a presentation boundary anchor for owner's incoming
tensor. Outputs map every declared owner handle exactly once. Recipe nodes may
instead specify `package={id=...,version=...}, parameters={...}, label=...` for
synthetic non-subflow operators (e.g. a join). Resolve version constraints against
the active catalog; require exactly one provider, apply/validate typed defaults
and declared parameter values and input/output handles. Synthetic subflows are
rejected because they have no model-owned body. No ambient resources or Lua
filesystem/process/network/module loading. Script cannot mutate graph, infer
shapes or call backend. Body instances are structural occurrences, not claims
about trained weights or parameter sharing.

Validate unique safe local IDs (reserved $input), endpoint and handle existence,
input occupancy, output/loss terminal compatibility, complete output mapping and
acyclic recipe/expanded graph. Fail explicitly on invalid plan, containment,
missing/duplicate nested boundaries, unknown package, script faults or limits;
no partial successful scene or silently truncated expansion. Incomplete ordinary
connections remain viewable. Isolated nodes remain visible. Bound Lua source
1 MiB, memory 8 MiB, instructions 1000000; maximum depth 32, 4096 body instances,
20000 scene nodes and 60000 edges. Malformed/orphan containment fails visibly.
Occurrence path combines source IDs and local instance IDs, independent of
layout/camera. Same source ID may have many scene occurrences. Groups retain
source owner ID and parent group; nested membership is inspectable.

Core Proxy uses default single body. Core Repeat's script describes serial
instances. Horizontal Repeat's script describes parallel bodies plus its
configured join operator. These are package data, not C expansion cases.

## Horizontal Repeat prerequisite

Supersedes the former object-valued stereotype limitation in model.md and
resource-authoring.md. NNValue adds owned objects of unique nonempty string keys
and typed values; copy/dispose/equality, JSON persistence/snapshots, undo and Lua
parameter conversion preserve them with existing depth/item bounds. `json`
parameters remain arrays. `stereotype` values are `{id,version,parameters}`:
resolve one active package matching exact/caret version, require declared kind
when present, validate its parameters/defaults and do not allow recursive
unbounded references. No custom string encoding. Catalog keeps optional
parameter kind. Existing Horizontal Repeat default is now usable and inspector
can edit reference JSON through normal C validation.

`services.infer_stereotype(reference, tensors)` is a generic bounded Lua host
service resolving active catalog reference and invoking its rule with validated
parameters and ordered tensors, without model mutation. Existing depth/invocation
limits apply, faults/root causes propagate. Synthetic subflow references are
rejected. This enables preserved Horizontal Repeat Lua, including Concat.

## LLM fixture and acceptance

Replace examples/tiny-decoder-llm's opaque causal-attention node with
Horizontal Repeat of eight explicit attention heads. Each body projects Q/K/V
512->64, transposes K, joins Q/K with MatMul, scales by 1/sqrt(64), applies causal
mask then Softmax, joins probabilities/V with MatMul; concatenate eight heads
on last axis then project 512->512 outside repeat. Retain six serial decoder
blocks, residuals, FFN, logits and scalar loss. Small project-owned shape rules
may implement operations missing from core, never opaque whole attention.
README documents metadata-only behavior and expected shapes. The unused original
attention resource may remain declared because resource deletion is deferred;
no graph node uses the opaque attention package. Model authoring and final
inspection use the actual Qt application through computer use.

Verify core C-only build, full existing tests, sanitizer checks, new generic
composition/object/reference/camera tests and git diff --check. Fixture analysis
must complete with logits float32[B,128,32000], loss scalar; 3D must expose all
6*8 head occurrences and distinct source paths. Fresh real Qt/noVNC QA must
inspect Fit, Home, flight/look, picking, visible grouped heads, 2D/3D switching,
no mutation and error reporting. Record actual evidence and limitations in KB.

C catalog shares `nn_catalog_resolve` (unique exact/caret active reference),
`nn_catalog_parameters` (owned defaulted validated NNParameter list) and
`nn_catalog_parameters_free` between inference, application and visualization.
Catalog may use public NNValue types; no module includes another module's
private header. JSON object keys are owned NNParameter entries.
