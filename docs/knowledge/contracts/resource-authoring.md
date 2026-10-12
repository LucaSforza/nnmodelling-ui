# Resource authoring, presentation and VAE

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
* `nn_app_create_llm(app,parent,id,name,error,cap)` copies
  examples/models/tiny-decoder-llm into a new editable project through the same
  staged template transaction as VAE; Qt uses it for the mini LLM shortcut.
* `nn_app_create_vae(app,parent,id,name,error,cap)` copies examples/models/mnist-vae
  into a new editable project, including its packages and datasets.

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
saving the project. Dataset metadata editing is accepted below; stereotype editing and resource
deletion remain deferred.

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
Limits: depth 32, 256 subflow invocations per report; breaches are runtime faults.
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
stereotype editing/resource deletion, object values and training. Typed output/subflow
authoring follows typed-outputs.md.

Accepted 2026-10-04: [3D contract](visualization-3d.md) supersedes Horizontal
Repeat object/join limitations. Its generic inference service and separate
subflow-only visualization Lua entrypoint are accepted; numerical execution
and training remain deferred.

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

## miniDeepSeekv4 1B checkpoint (accepted 2026-10-08)

The previous miniDeepSeekv4 directory was a tiny-decoder copy, not V4. Replace
its example resources with an explicitly scaled V4 architecture following
supplied arXiv 2606.19348v1 sections 2 and 4 plus official inference reference.
Principal authors these model-specific resources; GPT-6 Luna owns software fixes.
GUI composes editable subflows and individual embedding, normalization, mHC,
attention, MoE, head and MTP nodes. A single opaque DeepSeek decoder stereotype
is rejected by the user. Generic graph services execute this topology; there is
no DeepSeek package-ID switch in C or Python graph runtime.

Keep four residual streams and 20 Sinkhorn iterations, sigmoid pre/2*sigmoid
post, RMS-normalized dynamic mHC mappings, shared-KV MQA, low-rank queries and
grouped output projection, sliding window 128, partial RoPE and inverse output
RoPE, learnable sinks, CSA overlap compression ratio4 plus lightning indexer,
HCA non-overlap ratio128, and early token-ID hash MoE followed by learned
sqrt(softplus) routing. MoE has one shared and fine-grained routed SwiGLU experts,
bias-only auxiliary-loss-free balancing update .001 and sequence balance loss
weight .0001. MTP depth1 shares embedding/head with main path, weight .3.

Scale width/layers/expert counts to approximately 1 billion registered model
parameters; publish exact total/trainable count from Lua and actual PyTorch
without padding or unused trainable tensors. Original 128K-family tokenizer is
an immutable inference asset. FP32/BF16 reference math replaces FP8/FP4 kernels
because available Quadro hardware lacks these formats. No claim of 1M context,
production kernel efficiency, original 32T-token data mixture or original-model
quality. Training context is selected explicitly for available GPU memory.
CSA indexer receives a detached attention-distribution distillation objective
so top-k selection does not leave it untrained. Causality masks allow only fully
completed prior compression blocks and current/recent window tokens. Tests
must perturb future tokens and verify unchanged earlier logits.

The shared vocabulary-head node returns [B,T,2,V] (main and one-step MTP
logits). Explicit joins sum balance/indexer auxiliary losses from visible
subflows. Next-token projection selects main logits;
objective joins head logits and aggregated auxiliary loss plus declared target tensor,
computing main CE + .3*MTP CE on valid shifted positions + auxiliary loss.
Prediction omits objective targets. Only one owner registers shared embedding
and head parameters; MTP reuses them without duplicate counts.

Dataset: public HuggingFaceFW/fineweb-edu sample/10BT, immutable revision
87f09149ef4734204d70ed1d046ddc9ca3f2b8f9, size/hash-pinned Parquet files acquired
on Slurm. Adapter reads row batches lazily and creates nonoverlapping token
windows within individual documents, so unrelated documents never attend across
boundaries. Deterministic SHA256(document id) partitions train/validation/test;
held-out evaluation uses explicit bounded document budgets. Training budget
chosen by user is 100 million processed training tokens, first checkpoint;
report actual tokens and updates, never call it complete 10BT pretraining.
Inference code belongs under user-requested examples/implmentation/miniDeepSeekv4.

The first 100M-token miniDeepSeekv4 checkpoint acquires the first two pinned
Parquet shards of FineWeb-Edu sample/10BT (about 4 GiB), sufficient source
material beyond its budget. The manifest records this explicit subset;
additional corpus shards can be declared without an aggregate download cap.

## Editable V4 subflows (user refinement 2026-10-08)

Root embedding is computed once and fanned out to main/MTP paths. Stream-expand
produces [B,T,4,D]. Main depth is: two SWA/hash blocks, one CSA/hash block, one
HCA/learned block, and five editable pairs of CSA/learned + HCA/learned blocks.
Each repeated owner is a project-owned two-output subflow: normal residual
state and explicitly summed scalar auxiliary loss. Its body is a real graph
with Input, mHC pre/select/merge, RMSNorm, attention, MoE, loss join, Output and
Loss Output nodes. Repeat owns independent built subflow copies; Lua counts
body parameters multiplied by times. Users can enter/expand all bodies.

mHC-pre owns dynamic/base/scale parameters and emits one ordinary float tensor
packing original residual streams, mixed sublayer input, post weights and
residual matrix. These are tensor values, not Python hidden state. Stateless
mHC-select exposes mixed input; mHC-merge joins that pack and the sublayer
output and implements residual mixing. The format is local to these resources
and validated by dimension/stream parameters. Norms remain separate visible
nodes. Token IDs for hash routing are declared batch.inputs.tokens bindings,
retaining integer dtype; tokens are never encoded in floating residual state.

Main and MTP each have visible mHC-head collapse and final RMSNorm. MTP uses
shifted original embeddings and previous residual streams, separate visible
RMSNorm and linear projections, a stateless fusion join, and one editable
CSA/learned block. A feature-stack join pads the MTP tail with zeros and stacks
main/MTP hidden vectors; one vocabulary projection applies shared weights to
both, without duplicate ownership. Embedding/head use AdamW; block projection
matrices use the accepted Muon policy. Existing obsolete decoder resources are
not activated or used as an alternative execution path.
