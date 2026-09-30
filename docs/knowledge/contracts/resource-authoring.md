# Resource authoring, presentation and VAE

Accepted 2026-09-30 from user request. Supersedes migration-only limits for
resource creation, positioned parameters and single-tensor subflow analysis.
Backend, numerical execution and training remain deferred.

## Persistence and presentation

All newly written project/resource JSON uses two-space indentation and a final
newline. Reading accepts compact JSON; schema versions and atomic save remain.
Core assets are immutable. `NNParameterDef.position` is optional borrowed text,
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
* `nn_app_create_vae(app,parent,id,name,error,cap)` copies examples/mnist-vae
  into a new editable project, including its packages and datasets.

Qt builds boundary definition JSON from visual forms: stereotype name,
description, kind, color, parameter rows (key, type, default, minimum, choices,
position), dependency rows (ID/version constraint), optional Lua editor; dataset
name/description and input/target slot rows (name, dtype, shape). No raw JSON
editing is required. Default Lua is a visibly labelled pass-through shape rule,
not an implementation. Authored kinds: input/layer/join/output/subflow. Parameter
types: boolean/integer/number/string/dtype/json arrays. Defaults, minima, choices
must be typed and usable by the existing application validator. Dataset requires
at least one input, unique nonempty slot names, rank 1..64 and dimensions positive
integers or B; dtypes float16/bfloat16/float32/float64/int8/int16/int32/int64/uint8/
bool. Reject duplicate/empty parameter keys, invalid type/position, nonfinite
numbers, invalid schema and malformed Lua. Lua source is <=1 MiB and must compile
and return a function in a protected bounded runtime before activation.
Validation failures keep the Qt form open, preserve fields and display English
errors beside the form/Lua source, with Lua line when available (diagnostics.md).

Only project.c writes files. Generated paths: packages/<id>-<version> or
datasets/<id>-<version>; safe IDs and semantic versions required. Reject core
replacement, duplicate identity, existing destination, missing dependency,
symlink parents, traversal and existing catalog/resource limits. Creation stages
files, candidate references/catalog/datasets against current graph, validates,
atomically saves candidate model.json, then publishes. Failure keeps graph,
resources, dirty flag and previous model.json, removing only transaction-created
files. Success saves all current edits and clears dirty. UI labels Create as
saving the project. Resource editing/deletion are not introduced.

## Single-tensor subflow inference

Successful subflow analysis requires exactly one immediate package-kind input
and one output boundary. Children retain scope=owner.id and local positions;
cross-scope edges stay forbidden. Root Input resolves dataset binding; nested
Input receives the tensor passed to its owner instead. Lua service
`services.infer_subflow(tensor)` recursively evaluates calling subflow's scope
and returns its Output tensor. Available only for kind=subflow, selected by kind
not package ID; Lua chooses Proxy/Repeat delegation. Hidden children analyze.
Report includes all children once per node (last invocation for Repeat).
Empty/incomplete/missing upstream boundaries: unresolved. Duplicate boundaries
or malformed containment: semantic error. Lua faults propagate as runtime faults.
Limits: depth 32, 256 subflow invocations per report; breaches are runtime faults.
Orphan imported scopes stay viewable but unresolved. Horizontal Repeat's object
join is still unsupported; multi-output subflows and compiler remain deferred.

## Bundled MNIST VAE design

examples/mnist-vae is untrained editable metadata. Root: image -> Flatten ->
encoder (core.subflow-proxy) -> Reparameterize -> decoder (core.subflow-proxy)
-> reconstruction. Encoder shared hidden layer branches into mean/log_variance
Linear heads. Project-owned vae.diagonal-gaussian join packs equal floating
[B,L] tensors as [B,2,L], mean then log variance. vae.reparameterize returns
[B,L], documenting z=mu+exp(0.5*log_variance)*epsilon, epsilon~N(0,I).
vae.kl-divergence branches from encoder to per-sample [B], documenting
KL(q(z|x)||N(0,I)). Decoder maps 32 -> 128 -> 784 -> project-owned Sigmoid.
Reconstruction and KL are separate outputs, not a claimed training objective.
Project owns four custom stereotypes and reconstruction dataset metadata
(image [B,1,28,28], target [B,784]). Lua validates ordered joins, ranks, dtypes
and dimensions; no VAE package-ID switches in C.

Accepted: above behavior and CLI integration per automation contract. Deferred:
resource editing/deletion, object values, multi-output subflows and training.
