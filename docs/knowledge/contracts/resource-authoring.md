# Resource authoring, presentation and VAE

## Dataset deletion

The manager offers Delete for the selected dataset, with confirmation identifying
the dataset, permanent removal of its project files, and saving current edits.
Deletion of the active dataset clears the active selection; existing graph bindings
remain and diagnostics apply until another dataset is activated. Project C checks
that the dataset directory is confined under datasets and does not overlap any
other declared dataset or package. It stages the directory by rename, removes the
reference and saves the project atomically; save failure restores directory and
in-memory state. Success invalidates analysis and creates a saved history barrier.
If staged files cannot be fully removed after save, report their recovery path.

## Dataset import and explicit activation (accepted 2026-11-01)

The dataset manager imports a user-selected local dataset directory into the
project. The source is never modified. Existing schema-1 manifest, definition,
adapter and data are retained. When metadata is absent or incomplete, the UI
asks for missing identity, name and input/target tensor slot information using
the existing dataset form; it preserves valid supplied metadata. A missing
Python adapter is explicitly identified and a nonfunctional scaffold is created;
the UI must never imply arbitrary raw data became trainable. Malformed metadata,
unsafe paths, symlinks, duplicate identities and destination collisions fail
visibly. Only C project code writes project files. Import stages a confined copy,
validates the candidate, atomically saves the project and publishes it; failure
rolls back created files and preserves prior model, resources and dirty state.
Successful import saves current edits and creates a history barrier. First import
is activated; later imports retain the active dataset. Legacy projects with no
active dataset remain loadable, but UI offers no deactivation action.

Resource-list clicks only select. A clearly labelled Activate action selects an
exact dataset; the active identity is prominent in both resources and manager.
If the selected dataset has exactly one input slot and the root graph has exactly
one Input node, activation updates its binding to that slot. With more than one
root Input, activation fails visibly without changing dataset or bindings. With
zero Input nodes activation succeeds. Multiple-slot datasets keep authored
bindings for manual editing. Nested subflow Inputs are inherited, not bound.
Dataset activation and binding update are one atomic operation and failed
activation preserves project state. New root Input with an active single-slot
dataset receives that binding when unambiguous; otherwise diagnostics guide the
user. No selection operation silently deactivates the only active dataset.

Accepted refinement 2026-11-01: with one active dataset input and exactly one
root Input, binding is automatic and displayed read-only with an explanation.
Synchronize the persisted binding on import/creation/activation, active dataset
metadata edits and opening a project with a stale binding (mark changed graph
dirty). Reject manual writes of a different binding in this unambiguous case.
Nested Inputs and multi-slot datasets retain their existing semantics. Opening
an incomplete graph with multiple root Inputs remains allowed; activating a
single-slot dataset against that graph still fails. Dataset metadata edits and
their binding change share the existing save/rollback transaction. The UI must
not bypass binding synchronization merely because the dataset is already active.
Runtime adapts a single returned input to the single declared slot as specified
in backend.md; this does not rewrite adapter Python source.
Deleting a root Input that leaves exactly one root Input restores the automatic
binding of that survivor within the same graph history transaction. Undo/redo
restore both topology and binding together.

MNIST MLP and VAE examples offer both the existing small sample and a complete
MNIST resource. Full data includes all 60,000 official training and 10,000 test
images, with deterministic non-overlapping 55,000/5,000 train/validation split;
adapters produce each model's existing tensor and target semantics. Data is
available offline in the repository. The `examples/Test` example contains the MLP graph
without a declared dataset, for importing a bundled dataset through the UI.

Accepted 2026-09-30 from user request. Supersedes migration-only limits for
resource creation, positioned parameters and single-tensor subflow analysis.
Accepted 2026-10-01: typed-outputs.md supersedes single-output limits, boundary
topology and example terminal semantics.
Accepted 2026-10-05: backend.md defines numerical execution, training and Python
resource scaffolds; C retains native resource and graph ownership.

## Persistence and presentation

All newly written project/resource JSON uses two-space indentation and a final
newline. Reading accepts compact JSON; schema versions and atomic save remain.
Core semantics/assets remain preserved. Accepted 2026-10-03 user-requested
presentation-only exception: core Repeat declares `times` (repeat count) position=top; no rule,
parameter validation/default or package identity changes. `NNParameterDef.position` is optional borrowed text,
`top` or `bottom`; absent means inspector-only, invalid values fail loading.
Cards show positioned parameters in definition order in key/value rows above
or below a central title, with badges and content-sized geometry. Ports follow
the actual top/bottom. Native controls use explicit contrasting text, base,
placeholder, selected, header and disabled colors under light or dark desktops.

## C application operations

All operations return bool with caller-owned error buffer; UTF-8 is borrowed:

* `nn_app_create_stereotype(app,id,version,definition_json,inference_lua,
  dependencies_json,error,cap)` creates a schema-1 project package.
* `nn_app_create_dataset(app,id,version,definition_json,select,error,cap)` creates
  schema-1 dataset metadata; select persists it as active dataset.
* `nn_app_select_dataset(app,id,version,error,cap)` selects an exact declared
  dataset, marks dirty; invalid identity makes no change.
  Binding mismatches remain unresolved diagnostics, not selection failures;
  incomplete graphs must permit choosing datasets before editing bindings.
* `nn_app_create(app,parent,id,name,version,mnist,error,cap)` accepts and
  persists the project version. `nn_app_create_vae(app,parent,id,name,version,
  error,cap)` copies
  examples/models/mnist-vae into a new editable project, including its packages
  and datasets.
* `nn_app_create_llm(app,parent,id,name,version,error,cap)` copies
  examples/models/tiny-decoder-llm into a new editable project through the same
  staged template transaction as VAE; both template operations persist the
  supplied version. Qt uses LLM creation for the mini LLM shortcut. The local
  CLI continues to create template projects with version `0.1.0`.

Qt builds boundary definition JSON from visual forms: stereotype name,
description, kind, color, parameter rows (key, type, default, minimum, choices,
position), dependency rows (ID/version constraint), optional Lua editor; dataset
name/description and input/target slot rows (name, dtype, shape). No raw JSON
editing is required. Default Lua is a visibly labelled pass-through shape rule,
not an implementation. Authored kinds: input/layer/join/loss/output/loss-output/
subflow. Optional typed output ID/type rows follow typed-outputs.md. Parameter
types: boolean/integer/number/string/dtype/json arrays. Defaults, minima, choices
must be typed and usable by the existing application validator. Dataset requires
at least one input, unique nonempty slot names, rank 1..64 and dimensions positive
integers or B; dtypes float16/bfloat16/float32/float64/int8/int16/int32/int64/uint8/
bool. Reject duplicate/empty parameter keys, invalid type/position, nonfinite
numbers, invalid schema and malformed Lua. Lua source is <=1 MiB and must compile
and return a function in a protected bounded runtime before activation.
Validation failures keep the Qt form open, preserve fields and display English
errors beside the form/Lua source, with Lua line when available (diagnostics.md).

Only the project module writes project/resource files. Generated paths: packages/<id>-<version> or
datasets/<id>-<version>; safe IDs and semantic versions required. Reject core
replacement, duplicate identity, existing destination, missing dependency,
symlink parents, traversal and existing catalog/resource limits. Creation stages
files, candidate references/catalog/datasets against current graph, validates,
atomically saves candidate model.json, then publishes. Failure keeps graph,
resources, dirty flag and previous model.json, removing only transaction-created
files. Success saves all current edits and clears dirty. UI labels Create as
saving the project. Dataset metadata, deletion and stereotype Lua editing are accepted below; other stereotype
editing and stereotype deletion remain deferred.

## Subflow inference

Successful subflow analysis requires exactly one immediate package-kind input
and mapped terminal boundaries for all declared outputs per typed-outputs.md.
Children retain scope=owner.id and local positions;
cross-scope edges stay forbidden. Root Input resolves dataset binding; nested
Input receives the tensor passed to its owner instead. Lua service
`services.infer_subflow(tensor)` recursively evaluates calling subflow's scope
and returns its mapped tensor(s). Available only for kind=subflow, selected by kind
not package ID; Lua chooses Proxy/Repeat delegation. Hidden children analyze.
Report includes all children once per node (last invocation for Repeat).
Empty/incomplete/missing upstream boundaries: unresolved. Duplicate boundaries
or malformed containment: semantic error. Lua faults propagate as runtime faults.
Limits and successful scope memoization follow the Lua contract: depth 32,
bounded actual work and cumulative instructions replace the old shared
256-call ceiling; breaches are explicit runtime faults.
Orphan imported scopes stay viewable but unresolved. Horizontal Repeat's object
join is still unsupported; compiler remains deferred. Typed multi-output
subflows are accepted, not deferred.

## Bundled MNIST VAE design

examples/models/mnist-vae is the editable VAE project. Root: image -> Flatten ->
encoder (core.subflow-proxy) -> Reparameterize -> decoder (core.subflow-proxy)
-> reconstruction. Encoder shared hidden layer branches into mean/log_variance
Linear heads. Project-owned vae.diagonal-gaussian join packs equal floating
[B,L] tensors as [B,2,L], mean then log variance. vae.reparameterize returns
[B,L], documenting z=mu+exp(0.5*log_variance)*epsilon, epsilon~N(0,I).
vae.kl-divergence branches from encoder to per-sample [B], documenting
KL(q(z|x)||N(0,I)). Decoder maps 32 -> 128 -> 784 -> project-owned Sigmoid.
Accepted 2026-10-01: KL is kind=loss; a project-owned total-loss join models
reconstruction MSE plus mean per-sample KL, with a declared loss output.
Root has one reconstruction Output and one Loss Output. No numerical training
claim is made. Project owns five custom stereotypes and reconstruction metadata
(image [B,1,28,28], target [B,784]). Lua validates ordered joins, ranks, dtypes
and dimensions; no VAE package-ID switches in C.

Accepted 2026-10-08: the MNIST VAE training objective corrects reduction scales.
Core MSELoss averages over batch and all 784 pixels; the VAE KL operation sums
32 latent dimensions per sample. The scalar reconstruction MSE is multiplied
by 784 before adding mean per-sample KL, yielding batch-mean
`sum_pixels_squared_error + KL`. This is tied to the example's fixed 28x28
image shape; changing image resolution requires changing the pixel factor too.
Existing checkpoints and wheels retain old loss semantics and must be retrained
to obtain outputs from this objective.

Accepted: above behavior and CLI integration per automation contract. Deferred:
other stereotype editing/deletion, object values and training. Typed output/subflow
authoring follows typed-outputs.md.

Accepted 2026-10-04: Horizontal Repeat uses validated object-valued stereotype
references and the generic `services.infer_stereotype` host service. No
presentation-only visualization entrypoint or expanded network explorer is
part of the native client contract.

Accepted 2026-10-05: [backend contract](backend.md) and [backend UML](../uml/backend.md) supersede earlier backend/training deferrals. Native C11 graph authority and Lua shape analysis remain unchanged.

## Dataset management (accepted 2026-10-05)

The Project resources panel exposes `Dataset` alongside the existing `New dataset`
shortcut. `Dataset` opens a project-local manager listing exact identities and
active selection, with New dataset, Edit and Select actions. Edit/Select require
an existing selection; no-project actions report the existing open-project hint.
The manager refreshes after successful operations; closing it changes nothing.

The existing dataset form serves create and edit. Edit preloads name, description,
input/target slots and selected identity; ID/version are read-only. Save edits
name, description, dtype and shape/slot definitions. It preserves dataset adapter
code, uv files, samples and all unrelated JSON keys, including batch metadata and
extra properties of retained slots. Creation remains available through both entry
points. Validation failure keeps fields and shows inline errors; Cancel writes nothing.

`nn_app_dataset_definition(app,id,version,error,cap)` returns owned JSON for one
exact declared dataset, released with `nn_app_free_text`. It reads the confined
manifest definition entrypoint through C, never through Qt filesystem code.
`nn_app_update_dataset(app,id,version,definition_json,error,cap)` patches only
name, description and batch input/target maps of that dataset. Identity and active
selection remain unchanged. Schema validation uses the create rules. Root, batch
and retained-slot fields unrelated to the form survive; removed slots are removed.
Missing graph bindings become normal unresolved diagnostics, not edit rejection.

Project C stages merged JSON and owned dataset metadata, validates before writing,
then replaces the regular definition file using a flushed sibling temporary and
retained original backup. It saves current model edits using existing atomic
model persistence before publishing refreshed metadata. Ordinary failure restores
original definition bytes, active metadata, dirty state, graph and application
history/report. Rollback failure is reported explicitly and retains recovery data.
Symlink/traversal, invalid identity/schema and malformed definition entrypoints
are rejected. Successful update saves current graph edits, clears dirty state,
invalidates analysis and creates the same history barrier as resource creation.
The two-file operation has runtime rollback; no new crash-recovery/file schema.
Native client edits metadata only and never executes or edits Python/sample files.


Accepted 2026-10-12: bundled examples remain portable and bounded. Large V4
example, dedicated consumer and probes are removed from the shipped tree.
Generic runtime and parameter-count contracts remain unchanged.

## Project stereotype source editing (accepted 2026-10-28)

Creation form exposes an Edit initial source button. It opens a separate editor dialog
with Lua and Python tabs and an editable Python stub. Applying copies both drafts
back to the form without creating files. Only Create and save project submits
the drafts through the same resource transaction as metadata. A selected node whose exact
package identity is declared as a project-owned resource exposes a source
editor in the inspector, with Lua and Python views and simple syntax coloring. Core packages cannot be edited, even through C APIs.
The editor identifies the package, explains that all its nodes share the rule,
and applies explicitly; cancel or validation failure preserves the draft and
original. Successful apply immediately invalidates analysis and refreshes results.

`nn_app_stereotype_source` returns owned source (free with `nn_app_free_text`);
`nn_app_update_stereotype_source` accepts exact ID/version, source kind and text.
The kind is restricted to inference Lua or PyTorch Python. `nn_app_create_stereotype_with_python`
accepts both sources; the existing create API retains its default scaffold. Both reject
non-project resources, unsafe paths, symlinks and nonregular files. Only declared inference and PyTorch entrypoints may be edited; parameter-count
files are unchanged. Both sources are bounded to 1 MiB. Lua is validated with
the existing bounded validator before writing. Python is saved as text and never
executed or syntax-checked by the native client; training uses it in the existing
worker and existing jobs retain their snapshots. C stages a flushed sibling and backup, atomically replaces the source,
then saves current project edits; on failure restore original bytes and dirty
state, reporting rollback failure and recovery path. Successful apply is a saved
history barrier, and affects every node using that exact package.
