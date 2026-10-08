# Typed model and package contract

## State

`NNModel` owns a graph of stable node IDs and edge IDs, a project manifest,
and persistent presentation metadata. A node stores exact package `{id,
version}`, primitive parameter values, parent scope, and position. It does not
store a display name as package identity. An edge stores source/target node and
handle IDs, plus optional scope-local route points. Handles are resolved from
the activated package definition and node topology, not stored as independent
graph nodes. `Subflow` owns a nested immediate graph. `Input`, `Layer`, `Join`,
`Subflow`, and other kinds are package definition data, not a closed C enum of
package IDs. `StereotypeApplication` is a node's exact package reference and
parameter instance set. Source/target handles have stable local IDs.
Package definitions optionally enumerate typed outputs; normalized defaults
and overrides follow [typed outputs](typed-outputs.md). Native input
topology supplies `in` on single-input nodes and
ordered `in-1`, `in-2`, ... on join nodes. Join order follows numeric suffix;
malformed join handle IDs are rejected. This convention uses package kind,
never concrete package IDs. Output/loss-output terminals have no outgoing ports.
Nodes additionally own optional data.boundaryHandle for nested terminal mapping.

Persistent model state is distinct from editor selection/camera/drag state and
from per-frame draw commands. A typed model is authoritative; JSON parser
objects are temporary. Current release implements project persistence with
vendored yyjson and the legacy package-native `model.json` schema v2. Unknown
node/edge presentation fields may be ignored on read only when semantic
behavior survives; saving must preserve supported manifest/resource references
and graph semantics. Unsupported custom resource formats fail visibly before
switching projects.

## Application operations

Accepted 2026-09-30 additions and exact C ABI: see
[resource authoring](resource-authoring.md). Parameters additionally carry
optional top/bottom presentation position; nested scopes now have bounded
typed per-handle inference per typed-outputs.md, without cross-scope edges or
compiler changes. Subflow creation atomically spawns one inherited Input and
its declared terminals per typed-outputs.md.

| Operation | Parameters and result | Mutation/failure |
| --- | --- | --- |
| `project_open(path) -> Result` | Validated project directory and model snapshot | Stage full package closure and graph; commit atomically or preserve previous active project. |
| `project_save(path) -> Result` | Active typed model | Atomic file replacement; errors leave prior file intact. |
| `model_add_node(package_id, version, scope, position, parameters) -> Result<NodeId>` | Exact active package and validated primitive values | Create one node or no mutation. Defaults come from definition. |
| `model_delete_node(id) -> Result` | Existing node ID | Remove incident edges in same transaction; report missing ID. |
| `model_connect(source, source_handle, target, target_handle) -> Result<EdgeId>` | IDs in active graph | Add exactly one valid edge or no mutation. |
| `model_disconnect(edge_id) -> Result` | Existing edge ID | Remove exactly one edge or no mutation. |
| `node_set_parameter(id, key, value) -> Result` | Declared key and valid typed value | Change one value; invalidate dependent type analysis; no change on error. |
| `graph_validate() -> Diagnostics` | Borrowed active model | No mutation; distinguish structural, semantic, unresolved, runtime faults. |

## Exported model operations (accepted 2026-10-06)

The optional `manifest.operations` array declares named, single-input,
single-output inference entrypoints for the standalone Python wheel. Missing
array means no named operations and remains compatible with schema-v2 projects.
Each record is data, not executable source:

```json
{
  "name": "decode",
  "input": { "node": "decoder", "handle": "in", "codec": "tensor" },
  "output": { "node": "decoder", "handle": "out", "codec": "dataset" }
}
```

Operation names are unique safe Python identifiers; `infer`, `inference`,
`run_operation`, Python keywords and private names are reserved. Endpoints name
stable root-scope node and handle IDs. An input endpoint on a kind=input node
identifies its output handle and replaces that node's dataset binding for this
call. On another node it identifies one declared input handle; the operation
injects its value at that handle and cuts only that incoming edge from the
operation's dependency closure. The saved graph edge and normal model path are
unchanged. An output endpoint identifies a declared output handle on a
computational node. Terminals, nested-scope nodes, undeclared handles and stale
references are invalid.

`codec` is `dataset` or `tensor`. Dataset input calls the active adapter's
`tokenize` and requires one Tensor; tensor input requires a batched
`torch.Tensor`. Dataset output calls the adapter's `untokenize`; tensor output
returns the selected batched Tensor. Dataset codecs own image/string or other
application-value conversion. Operation execution evaluates only the backward
dependency closure of the selected output, stopping at the declared input
boundary, under inference mode. It uses same graph modules and weights as full
inference. The first contract supports one input and one output per operation;
multi-input/output operations are deferred.

The C project owner parses, validates, copies and persists operation metadata
inside `manifest.operations`; it never runs adapter or PyTorch code. Qt exposes
a project-level Operations manager; it uses stable node/handle selectors and
current C shape-analysis results to display signatures. Changes mark project
dirty and save with ordinary project persistence. Graph edits that leave stale
operation endpoints stay visible and block training/export until corrected or
removed. UI/runtime do not infer operations from package IDs.

The wheel exposes each declared name directly (`model.encode(value)`) and also
provides `model.run_operation(name, value)`. `model.infer` and `model.inference`
remain unchanged. For MNIST VAE, `encode` uses dataset tokenization and returns
the reparameterization node output; in eval mode this package deterministically
returns posterior mean. `decode` accepts latent Tensor `[B,32]`, bypasses the
existing edge into the decoder subflow, and applies dataset untokenization to
its reconstructed pixels. Prior sampling is ordinary caller code, for example
`model.decode(torch.randn(1, 32))`; the wheel does not add a second sampling
engine.

The application C ABI is specified below. UI and local automation call it
without reimplementing validation.

The current C graph API is `src/model/model.h`: `nn_model_new/free`,
`nn_model_add_node/remove_node/move_node/rename_node`,
`nn_model_connect/disconnect`, `nn_model_set_parameter`, and borrowed
`nn_model_node_at/edge_at/find_node` accessors. All mutations return `bool`
and write caller-owned error buffer on failure. Add/connect accept exact stable
string IDs from persistence or generated IDs from UI. The model owns copied
IDs, labels, package references, scope and deep typed parameters. Package and
handle existence are checked by the application catalog before mutation;
model enforces unique IDs, same scope, occupied target and acyclicity.
`NNValue` is a tagged boolean/integer/real/string/array tree with deep copy and
dispose operations. Borrowed pointers expire on next mutation or free.

For this release, the model API exposes owned model lifecycle and mutations,
plus borrowed read-only node/edge snapshots valid until next mutation. Node
IDs, edge IDs and package identities are stable strings. Parameters are
typed values (boolean, integer, real, string, array) with deep ownership;
JSON DOM pointers cannot be stored in the model. All successful mutations set
project dirty state through application owner.

## Invariants

Formal: `∀e∈Edges(g): source(e),target(e)∈Nodes(g) ∧ scope(source(e))=scope(target(e))=scope(e)`.
Natural: every edge connects nodes in one immediate graph scope.

Formal: `∀e1≠e2: (target(e1), targetHandle(e1)) ≠ (target(e2), targetHandle(e2))`.
Natural: one target handle accepts at most one edge.

Formal: `∀g: directedCycle(g)=false` for committed graph mutations.
Natural: connections cannot create cycles; reconvergent branches are valid.

Formal: `joinInputs(j)=sortByTargetHandle(incoming(j))`.
Natural: non-commutative joins follow handle order, never traversal order.

Formal: `activePackages = corePackages ∪ manifest.customPackages`, exact IDs and
versions; `node.package∈activePackages`.
Natural: undeclared packages cannot be loaded by name or ambient discovery.

Formal: `openFailed ⇒ (model', catalog')=(model, catalog)`.
Natural: failed project switch preserves old graph and package scope.

Formal: `collapsed(s) ⇒ semanticChildren(s)=children(s)`.
Natural: hiding subflow children never removes them from analysis.

Whole-graph completion requires valid boundary Input and exactly one root
Output and Loss Output per typed-outputs.md. Incomplete graphs remain editable.
Typed terminal restrictions are client semantics, not backend training rules.

## Qt migration application ABI (accepted)

`src/application/application.h` exposes opaque `NNApplication`. `nn_app_new(core_root)`
returns owned state, released with `nn_app_free`. `nn_app_open/create/save/close`
return bool and write caller-owned error text; close takes explicit discard.
`nn_app_project/model` return const borrowed snapshots, never mutable owners.
All command functions return bool with caller-owned error buffer and capacity.
`nn_app_add_node(app,id,package_id,version,scope,x,y,error,capacity)` resolves
exact package, applies existing editor defaults atomically and marks dirty.
`nn_app_remove_node`, `nn_app_move_node`, `nn_app_rename_node`,
`nn_app_connect`, `nn_app_disconnect`, `nn_app_set_parameter` delegate storage
to C model. Unknown keys, invalid types/ranges/choices and invalid handle
names/directions fail before mutation. Parameter text parsing is in C via
`nn_app_set_parameter_text`; arrays use JSON array text and stay NNValue trees.
`nn_app_parameter_text` returns owned malloc text, released with `nn_app_free_text`.
`nn_app_port_count(app,node_id,output)` and `nn_app_port_id(app,node_id,output,
index,buffer,capacity)` expose C-derived handle IDs. Join query enumerates
existing handles in numeric order plus first free slot, with in-1/in-2 for an
empty join; no allocation proportional to a potentially large suffix.
`nn_app_node_is_subflow` reads package kind. All string arguments are borrowed
for the call; model copies committed data. IDs stay stable strings.

Accepted 2026-10-06: `nn_app_operations_json(app,error,capacity)` returns owned
JSON array text released with `nn_app_free_text`. `nn_app_set_operations_json`
accepts borrowed JSON text, validates list shape, safe unique method names and
codec values, copies it into project-owned manifest state, and marks project
dirty only on success. Syntactically valid but stale node/handle references are
preserved for UI repair; operation status is derived from current graph/catalog
and analysis. Project open never loses a graph just because an old operation
reference is stale.

Accepted 2026-10-08: a changed operation list creates an unsaved graph-history
barrier, like dataset selection, because graph snapshots do not contain manifest
metadata. Failed and semantically unchanged setters preserve dirty state,
undo/redo and saved revisions. `nn_project_set_operations_json` accepts an
optional `bool *changed` out parameter, initialized false and set true only on
committed replacement; the application ABI remains unchanged.

Scope display follows editor contract. New non-root scopes require a subflow
owner; existing imported scope strings are preserved. Nonempty subflow deletion
is rejected rather than silently orphaning children. Coordinate rules below
apply. Fix borrowed-ID deletion safety in model without changing graph
semantics. Schema v2 now persists optional data.boundaryHandle; no legacy
migration is required. Existing core assets remain intact; core.loss-output is
added. Typed boundary setter and output type query follow typed-outputs.md.

Object-valued `stereotype` parameters (currently Horizontal Repeat's `join`)
remain a pre-existing native limitation: NNValue has no object variant. Preserve
the old editor's omission of that unsupported default when creating the node;
keep the package selectable and report unsupported editing explicitly. Do not
invent a string encoding or extend model semantics during this GUI migration.
The inspector presents this field read-only. Existing supported primitive/array
values and schema-v2 persistence remain unchanged.

## Integer grid positions (accepted 2026-10-03)

Node position snapshots store signed 32-bit integer x/y, aligned to a fixed
20-unit scene grid (`NN_MODEL_GRID_SPACING = 20`). Model add/move operations
retain double arguments at the input boundary to support Qt gestures and JSON
numbers, but normalize once in C before storing: round each coordinate/20 to
the nearest integer, ties away from zero, then multiply by 20. Reject non-finite
or out-of-range normalized coordinates before mutation; no overflow or clipping.
All producers, including imported project positions and spawned subflow
terminals, use this same normalization. Existing fractional project coordinates
normalize on open; saving writes JSON integer positions. No schema version
change or parallel coordinate representation. Snapshot responses also serialize
integer positions. Invalid positions preserve graph, dirty state and analysis.
Node creation, drag previews and layout use the same grid in Qt; view scale and
card dimensions remain floating presentation geometry. Position changes do not
invalidate tensor analysis. Saving and reopening preserve integer coordinates.

## Graph edit history (accepted 2026-10-03)

Application owns undo/redo snapshots of the existing NNModel, never a second
inference system. Model supports owned deep `nn_model_copy`, structural
`nn_model_equal` and allocation-free `nn_model_swap` for restoring history while
preserving the project owner. Copy preserves IDs, order, scope, typed parameters,
integer coordinates, boundary mappings and edges. Equality compares all fields.
Project exposes `nn_project_set_dirty` for restoring saved-revision state.
`nn_app_can_undo/can_redo`, `nn_app_undo/redo` use the normal caller-owned error
buffer convention. `nn_app_begin_edit` begins one nonnested graph transaction;
`nn_app_end_edit(app,commit,error,cap)` commits one changed edit or restores its
starting graph. History allocation succeeds before mutating active graph; failed
commands/groups preserve graph and history. Lifecycle/save/resource operations
reject an active group. Maximum 100 completed edits; oldest snapshots are released.
See editor.md for UI behavior, save revisions and resource barriers.

## Stereotype objects (accepted 2026-10-04)

[3D contract](visualization-3d.md) supersedes the earlier unsupported object
parameter limitation: owned NNValue objects preserve stereotype references,
parameters, history and schema-v2 JSON. C validates active package identity,
version constraint, expected kind and referenced parameters.

Accepted 2026-10-05: [backend contract](backend.md) and [backend UML](../uml/backend.md) supersede earlier backend/training deferrals. Native C11 graph authority and Lua shape analysis remain unchanged.
