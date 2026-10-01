# Typed outputs and terminal boundaries

Accepted 2026-10-01 from the user's eight review decisions. This supersedes
single-output topology, single-output subflow analysis and the earlier
multi-output deferral. UI/client shape analysis only; execution/training remain
deferred. Existing projects/examples are rewritten, not migrated or supported
through legacy-model compatibility. Existing historical core assets remain
preserved; add the new core.loss-output package rather than repurpose a loss
calculator. Presentation overrides boundary colors by kind.

## Definition syntax and logic

Optional stereotype field (definition schema remains 1):

```json
"outputs": [
  { "id": "prediction", "type": "output" },
  { "id": "objective", "type": "loss" }
]
```

An explicit list replaces defaults. Computational kinds have one or two outputs;
at most one per type, IDs unique/nonempty safe ASCII identifiers (letters,
digits, dot, underscore, hyphen), types exactly output/loss. Reject wrong shapes,
unknown types, duplicate IDs/types and empty computational lists. Definition
order is port/display order. Omitting the field derives {id:out,type:output},
except kind=loss derives {id:loss,type:loss}. A loss calculator otherwise behaves
exactly like a layer and may explicitly declare a normal output instead.

Kinds output and loss-output are terminal collectors, not calculators: one
input in, no outgoing ports, and absent outputs or explicit [] only. Input has
no incoming port. Other existing input rules (in, numeric join in-N) remain.
Join/Scale/etc. do not propagate an incoming edge's classification: their
declared/default outputs determine the outgoing classification.

Each edge takes its classification from its source handle. Normal intermediate
targets accept either classification as an ordinary tensor. Only terminals
restrict it: output accepts output, loss-output accepts loss. Direction, same
scope, occupancy and DAG rules remain. A collector accepts one edge; designers
must insert an appropriate join for multiple contributions. No implicit sum,
auto-wiring, loss bubbling or numerical computation is added.

Formal: |outputs(n)| <= 2 and for each t in {output,loss},
|{h in outputs(n): h.type=t}| <= 1.
Formal: terminal(t) implies |incoming(t)| <= 1 and outputs(t)=empty.
Formal: edgeType(e)=type(resolve(source(e),sourceHandle(e))).
Formal: kind(target(e))=output implies edgeType(e)=output;
kind(target(e))=loss-output implies edgeType(e)=loss.

## Root and subflows

A complete root contains exactly one immediate output terminal and one immediate
loss-output terminal, even when initially disconnected. New blank projects seed
these two terminals; templates contain them. Missing boundaries or disconnected
terminals are Incomplete; duplicate/wrong boundaries are semantic errors.
Incomplete graphs remain editable/saveable: do not force completion at open,
node removal or every mutation. Invalid handle/type edges do fail before commit
and on project load, including resource candidate validation.

Adding a subflow atomically adds its owner and one immediate terminal per resolved
output, using core.output/core.loss-output by type. It does not create internal
Input or internal edges: designers build those. Spawned IDs are generated stable
IDs, labels use output IDs, positions are scope-local. Rollback leaves no owner,
children, dirty change or analysis invalidation on failure. No automatic repair
on open or analysis. Designers can inspect/edit terminal mappings.

Child terminals persist data.boundaryHandle as a string, exposed as
NNNode.boundary_handle_id. It names the owning subflow's declared output ID;
root terminals have no mapping. Every declared output matches exactly one
immediate terminal with the same ID and type, and no extra terminal exists.
Exactly one immediate Input remains required for inherited input analysis.
Missing Input/mappings/boundaries/upstream tensors are Incomplete; duplicate,
unknown or wrong-type mappings and malformed containment are semantic errors.
Mapping edits use the application API; duplicate mappings remain diagnosable
while editing, unknown/wrong-type edits are rejected. Root/nonterminal mapping
edits are rejected. Default subflow has only out/output; internal losses are
not exported unless declared, joined and connected by the designer.

## C ABI and ownership

NNOutputDef {const char *id; const char *type;} and NNPackage.outputs/output_count
are catalog-owned normalized resolved definitions, including implicit defaults.
Use the same resolved metadata for load, commands, rendering and inference.
nn_app_output_type(app,node_id,handle_id) returns borrowed output/loss text or
NULL for unknown/non-output handles. Existing port_count/port_id enumerate the
normalized outputs. nn_app_set_boundary_handle(app,node_id,handle_id,error,cap)
copies the mapping into model state, marks dirty and invalidates analysis only
on successful semantic change. Model mapping setter owns/free/copies the string;
project persistence reads/writes data.boundaryHandle. No Qt types in core.

## Lua and analysis

Single-output rules may return {status="success",output=tensor}; this is the
single-tensor shorthand for their only declared output, independent of its ID
or type. Multi-output rules return {status="success",outputs={
prediction=tensor1,objective=tensor2}} keyed by declared ID. Require exactly the
declared keys and valid tensors; missing/extra outputs or using shorthand for
two outputs is a semantic error. Never duplicate one tensor implicitly across
handles. Terminal rules use output=tensor to validate/inspect the consumed
tensor without creating an outgoing handle.

services.infer_subflow(tensor) returns output shorthand for one output, or the
same keyed outputs map for two outputs. Child terminal mappings select each
result; edge inference must resolve sourceHandle, not just source node. Existing
Repeat is single-output only; a rule's explicit combination of repeated outputs
remains its responsibility. Depth/invocation bounds and fault/cause propagation
remain. Successful outputs are separately owned report tensors keyed by ID/type;
terminal outcomes retain their consumed tensor for inspection.

NNInferenceTensor has handle_id, type, dtype, dimensions and dimension_count.
NNInferenceResult adds outputs/output_count. Existing dtype/dimensions fields
are a primary tensor view (first output, or consumed tensor for terminals), not
an alternative source for edge inference. Per-node statuses/cause grouping
remain; failures publish no partial successful outputs. CLI successful tensors
include handle/type (null for terminals), one entry per handle. Inspector lists
all typed outputs and terminal consumed tensor, not merely the first.

## Qt and authoring

Input is a single filled black circle, Output a filled brown circle, Loss Output
a filled red circle; label outside, no card. They retain drag/select/connect,
inspector and visible diagnostics with generous hit areas. Computational loss
nodes remain normal stereotype-colored cards. Output handles and all fanout
edges/arrows are black/output or red/loss; drafts follow source classification.
Selection/hover must not erase classification (use a separate halo/outline).
Other nodes retain card/parameter/subflow design.

Stereotype form includes loss and loss-output kinds and an optional outputs
override editor with ID/type rows (max one each). Defaults remain visibly
distinguished from explicit choices. Terminal forms allow no outputs only.
Subflow terminal boundaryHandle is editable in inspector through C validation.
CLI stereotype.create accepts the same definition; node.boundary {id,handle}
sets a mapping. project.snapshot exposes normalized outputs on packages and
boundaryHandle on nodes. No automatic terminal connection or aggregation.

## Examples and verification

MLP exposes predictions through Output and its scalar Cross Entropy through
Loss Output. VAE exposes reconstruction through Output; KL is kind=loss.
An explicit project-owned join models reconstruction MSE + mean per-sample KL
and declares a loss output to the sole Loss Output. The join's Lua validates
scalar/[B] floating inputs and returns scalar shape metadata only. Encoder and
decoder keep default single out boundaries with persisted matching IDs.

Test defaults/overrides, all schema rejections, handle-sensitive independent
tensors, terminal-only restrictions, single-input collectors, transactional
spawn and mapping persistence, exact root/nested boundaries, CLI parity,
circle/port/edge graphics and retained forms. Run just test, just test-ui,
sanitizers and git diff --check; inspect actual screenshots.
