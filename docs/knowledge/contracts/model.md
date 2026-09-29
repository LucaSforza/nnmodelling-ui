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
Current package definitions state `kind` but do not enumerate handles. Native
topology supplies `out` on non-output nodes, `in` on single-input nodes and
ordered `in-1`, `in-2`, ... on join nodes. Join order follows numeric suffix;
malformed join handle IDs are rejected. This convention uses package kind,
never concrete package IDs, and matches preserved model edges.

Persistent model state is distinct from editor selection/camera/drag state and
from per-frame draw commands. A typed model is authoritative; JSON parser
objects are temporary. Current release implements project persistence with
vendored yyjson and the legacy package-native `model.json` schema v2. Unknown
node/edge presentation fields may be ignored on read only when semantic
behavior survives; saving must preserve supported manifest/resource references
and graph semantics. Unsupported custom resource formats fail visibly before
switching projects.

## Application operations

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

The application C ABI is specified below. UI and future automation call it
without reimplementing validation.

The current C graph API is `src/model.h`: `nn_model_new/free`,
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

Whole-graph completion requires a valid boundary Input and accepted terminal
topology; incomplete editor graphs may still show local inference and unresolved
diagnostics. Training-specific terminal rules are deferred to backend contract.

## Qt migration application ABI (accepted)

`src/application.h` exposes opaque `NNApplication`. `nn_app_new(core_root)`
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

Scope display follows editor contract. New non-root scopes require a subflow
owner; existing imported scope strings are preserved. Nonempty subflow deletion
is rejected rather than silently orphaning children. Finite coordinates are
required. Fix borrowed-ID deletion safety in model without changing graph
semantics. No file format change; core packages remain byte-for-byte intact.

Object-valued `stereotype` parameters (currently Horizontal Repeat's `join`)
remain a pre-existing native limitation: NNValue has no object variant. Preserve
the old editor's omission of that unsupported default when creating the node;
keep the package selectable and report unsupported editing explicitly. Do not
invent a string encoding or extend model semantics during this GUI migration.
The inspector presents this field read-only. Existing supported primitive/array
values and schema-v2 persistence remain unchanged.
