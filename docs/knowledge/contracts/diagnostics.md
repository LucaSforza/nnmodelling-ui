# Errors, analysis diagnostics and dynamic strings

Accepted 2026-09-30 from user review decisions. This contract supersedes the
success-row diagnostics presentation in ui-release.md. No model compiler/IR,
backend or training is introduced: compilation errors mean Lua syntax errors.

## C ownership and failures

Operation APIs use bool/NULL and optional caller-owned bounded UTF-8 error
buffers. Success clears the buffer; failure supplies a meaningful English
message when capacity permits. NULL/zero-capacity buffers are valid. Error
formatting and copying live in utils.c/h, without allocations for errors.
Preserve the original failure through cleanup. Partial allocations are never
published as successful results; allocation failure is not unresolved state.
Cleanup must tolerate partially initialized owners. No global last-error,
abort-on-OOM policy, compatibility wrappers or macro-based exception system.

User revision: do not introduce SDS for a single path-joining helper. Remove
the vendored dependency and build target. nn_path_join uses one checked malloc
and copies, with explicit size/overflow bounds. Borrowed strings remain ordinary
const char*; owned ABI text uses malloc/free. Error buffers stay allocation-free.
Remove the custom JSON TextBuffer: escaping/serialization uses existing yyjson.
No new string framework or third-party dependency is introduced.

## Analysis report

NNApplication owns a lazy read-only NNInferenceReport, invalidated on semantic
graph/resource/dataset changes and project replacement/close, not position,
label or save. Queries never mutate project or dirty state. Borrowed report
references expire on invalidation/application free. Whole-analysis failure
returns NULL plus error, never an empty successful panel.

Per-node outcomes retain tensors for success. Problems distinguish Lua
compilation error (error), model semantic error (error), incomplete (warning),
and internal/runtime fault (internal). Each problem exposes stable code,
node ID, optional root cause node ID, optional source file and Lua line.
Scope, label and exact package identity resolve from the current model/catalog.
Lua syntax errors have a separate status from rule execution faults. Rules and
containment failures retain their existing semantic/unresolved classification.
Blocked descendants inherit the originating cause ID, without parsing messages.
Subflow propagation preserves the child root cause. Order is deterministic.
Children not invoked because their owner is blocked inherit that owner's root
cause; intentionally uninvoked children of a successful owner remain independent
Incomplete outcomes. No duplicate root warnings for every blocked hidden child.
No success outcomes appear in the problem list; tensors remain inspectable.

Exact additions: NNInferenceStatus adds NN_INFERENCE_COMPILATION_ERROR.
NNInferenceResult adds borrowed code, cause_node_id, source_file and size_t
source_line (zero if unavailable). Codes are lua.compile, model.semantic,
model.incomplete, model.blocked and analysis.internal; a root has no cause ID.
nn_inference_category(status) returns lua-compilation/model/incomplete/internal/
success; nn_inference_severity(status) returns error/warning/internal/info.
nn_inference_error_line(message) extracts a Lua line if present, else zero.
nn_app_analysis(app,error,cap) returns const borrowed lazy NNInferenceReport*
or NULL with explicit error. Existing standalone inference remains available.
Allocation/report-construction failure cannot publish partial successful tensors.

## Qt presentation

Keep the existing right-hand panel, labelled Model problems. Show whole project
with optional current-scope filter, readable node names and scope paths. Root
causes appear first with blocked descendants as collapsed children.
The scope filter keeps outside-scope root causes needed by visible blocked
nodes as labelled context, rather than hiding the actual cause/severity.
Distinct icons, English category labels and colors distinguish Lua compilation errors,
model errors, Incomplete and internal unavailability (not color alone).
Use full-width wrapping problem rows within this narrow panel: category and
node name, scope path, then reason. Avoid two cramped columns that hide the
node/category or reason behind ellipses; technical details remain expandable.
Internal raw messages appear only in expandable technical details, with a
concise Analysis unavailable summary; never claim no problems when analysis
failed or incomplete. Success tensors live in inspector/port tooltip instead.
Problems are clickable: switch scope, select and center the node, using copied
stable IDs and queued synchronization. Cards have a discrete problem marker.

Rejected canvas commands, especially connections, open a nonblocking English
error dialog stating the operation and reason (no mutation, no silent status
bar-only error). Critical file operations retain visible dialogs. Resource
authoring stays open on validation/write errors and preserves every field;
Lua errors appear beside the source with line information when available.

## CLI parity

analysis.diagnostics {} is a read-only C command returning available, complete,
problems (code/category/severity/node/scope/package/file/line/message/causeNode),
and successful tensors. A whole-analysis failure returns ok:false and error.
The result is {available:true,complete:bool,problems:[],tensors:[]}; complete
means every reported node resolved successfully, not backend/training readiness.
Package is {id,version}; scope is the stable scope ID; file/causeNode are nullable
and line is zero if unavailable. Accepted 2026-10-01: tensors are
{node,handle,type,dtype,shape:[dimension text]}, one entry per typed output;
handle/type are null for consumed terminal tensors. Boundary completion and
mapped subflow diagnostics follow typed-outputs.md; graph edits stay available.
Root boundary-completion problems additionally have node/package null and are
not navigable. Root metadata APIs and complete conjunction follow typed-outputs.md;
even an empty graph reports Incomplete without synthetic node IDs.
Internal details are labelled as such; the UI renders the same C outcomes.
ui.reveal {id} uses the same navigation as clicking a problem; unknown IDs fail
without changing scope/selection. Rejected commands return the same C reason as
the UI dialog; do not open modal UI during CLI dispatch. Existing response
envelope and exit codes remain. No separate CLI validation or graph authority.

## Verification

Cover optional/tiny error buffers, bounded path joining and ownership, malformed
Lua with source line, semantic/incomplete/runtime classes, propagated causes,
failed allocations and partial cleanup, report cache lifetime, invalidation,
CLI/UI equivalent diagnostics, filters/navigation/markers and retained Lua forms.
Run just test, just test-ui (includes CLI), sanitizer checks and diff --check.
