# Local backend and Python execution

Accepted 2026-10-05 by explicit user request. Supersedes backend/training
deferrals in earlier native-client documents. OpenAPI, not OpenAI integration,
is requested. C11 client remains graph authority; Python executes immutable
snapshots. First deployment is this computer, loopback by default.

## Ownership and layout

`backend/` owns FastAPI HTTP service, persistent job storage, container launcher
and training worker. `python/nnmodelling-runtime/` owns reusable dataset SDK,
stereotype_runtime compatibility ABI, generic PyTorch graph and wheel export.
uv workspace at repository root manages both packages and lockfile. Qt Network
in `src/gui/qt/` owns asynchronous HTTP presentation only. C project module
continues owning local resource creation. Python never runs inside native UI.

## HTTP protocol v1

All JSON errors are visible. FastAPI publishes `/openapi.json` and `/docs`.
Optional server bearer token comes from environment; UI keeps token in memory,
never project files. Local unauthenticated deployment has one local owner;
when configured, token defines that owner. No remote multi-user claim.

* `GET /health` reports service and container availability without secrets.
* `POST /v1/jobs` accepts `{project,files,training}`. `project` is exact saved
  schema-v2 model JSON; `files` maps project-relative names to base64 file
  bytes (resources/data, excluding model.json, symlinks and environments).
  `training` contains epochs, batch_size, learning_rate, seed and optional
  publish_every_steps (integer 1..100000, default 10). Response is Job.
* `GET /v1/jobs` returns `{jobs:[Job]}` for current owner.
* `GET /v1/jobs/{id}` returns Job: id, status, created_at, error, metrics.
  Status is queued/running/completed/failed/cancelled. Metrics is
  `{epochs:[{epoch,training_loss,validation_loss}],test_loss:null}`; test_loss
  becomes finite numeric value after final evaluation.
  New metrics additionally contain `steps:[{step,epoch,training_loss,validation_loss}]`;
  old records without steps remain readable and show epoch curves.
* `GET /v1/jobs/{id}/snapshot` returns original `{project,files}`.
* `GET /v1/jobs/{id}/weights` downloads safetensors after successful training.
* `GET /v1/jobs/{id}/wheel` downloads generated Python wheel after success.
* `POST /v1/jobs/{id}/cancel` stops running/queued container, returns Job.

Reject absolute/traversal/backslash paths, symlinks, duplicate identities,
invalid base64, oversize bundles, invalid graph/config before execution.
Store immutable model/resources, selected dataset/data and exact resolved core
package bytes per job; new UI edits cannot change old jobs. No host paths in
API. Recover interrupted jobs visibly after restart; never claim completion
without final metrics, weights and wheel. Worker failures publish errors.

## Container and training

Production path always launches Docker-compatible containers, no permanent
local-process alternative. Build CPU worker image via just recipe. Mount only
job snapshot read-only and job output read-write; no network during training,
no host credentials/socket, resource limits and unprivileged worker. One local
job at a time suffices; persistent queue, cancellation and shutdown are owned
by service. Image/runtime missing yields actionable health/job errors.
Diagnostic tests may execute worker directly in temporary directories.
Standard CPU image limits OpenMP, MKL and OpenBLAS pools to two threads, matching
the local launcher's two-CPU quota instead of oversubscribing host CPU counts.

Dataset manifest adds `entrypoints.python={language:"python",file:"dataset.py"}`.
Module exports `Dataset`, subclass of `DatasetAdapter[InputT,OutputT]` instantiated
with dataset directory. Abstract `tokenize(value:InputT)` returns Tensor or named
tensor map for multiple inputs; `untokenize(tensor:Tensor)->OutputT` decodes root
prediction. Abstract `load(split,batch_size)` yields `Batch(inputs,targets)` for
train/validation/test, inputs/targets named Tensor maps, already batched.
No implicit synthetic data or target fabrication. Missing adapter/data fails.
Tokenize returning one Tensor binds the sole declared input slot; multiple
input slots require a named tensor map. Worker image includes resource Python
dependencies before launch; training cannot install dependencies over network.
Dataset manifest may declare `inferenceAssets`, a list of safe project-relative
file paths needed for tokenization/decoding (for example vocabulary). Wheel
copies those exact assets plus Python/metadata only; training files are excluded
unless explicitly declared inference assets. Exported resources preserve external
Python dependencies in wheel METADATA; SDK itself is privately vendored.
Adapter construction/tokenize/untokenize must work using inference assets alone;
training split payloads are loaded lazily by load(), not in adapter constructor.
Train uses Adam, graph's scalar Loss Output, validation eval/no_grad each epoch,
test once at end; means weighted by batch size, finite losses required.
The scalar Loss Output is a per-batch mean; aggregate reporting weights that
mean by sample count. Resource authors must preserve this reduction convention.

Bundled image examples declare Pillow in their own uv project dependencies.
Root uv workspace has an examples dependency group, locked with the workspace;
the standard local worker image installs this group before launch. Keep image
decoding out of the base SDK dependency set and avoid a second unlocked pip
requirements path. Local example tests install the same group.

## Step publication and learning curves (accepted 2026-10-05)

User reference image defines a native Training dashboard: job history at left,
large central learning-curve plot, controls above, readable metric/event log below.
History shows creation time, short job identity and state without horizontal
scrolling; full identity/time remain available in tooltip. Connection status uses
readable readiness/error text rather than serialized health JSON.
Training and validation losses have separate labeled colors and toggles, one
shared loss axis, global optimizer step on x axis, linear/log loss scale, and
visible final test loss. No fabricated accuracy/error percentages or dual axes
for quantities with the same loss units. Use Qt painting, no browser chart or
new heavyweight chart dependency. Existing connection, submit, cancel, snapshot
and artifact actions remain accessible. The dashboard opens from Training.
Curve paths are strokes only; filled dots identify samples. Reset brush state
before drawing each path so a preceding series' sample fill cannot close and
fill the next open curve. Toggling validation must never create a filled polygon
or alter the training curve's styling.

One step is one successful optimizer update across the whole job; it is not a
dataset sequence position. Every publish_every_steps updates, run the full
validation split under eval/no_grad, then atomically publish a paired curve point.
Training point is the sample-weighted mean since the previous publication within
the current epoch. End of every epoch publishes any remaining window, exactly
once at its final step; reuse validation if that step was already published.
Epoch summary retains the full-epoch weighted training mean. Test runs once at
completion. Validation preserves optimizer state and returns model to train mode.
Keep RNG streams unchanged across reporting-only validation, including Python,
NumPy and torch, so changing publication frequency cannot perturb training.
Worker owns cadence; Qt polls asynchronously and plots only published values.
UI field describes both metric publication and validation cadence. Configuration
is frozen in the job snapshot. Invalid values and nonfinite points fail visibly.
Step points are strictly increasing and retain their epoch. Reader accepts old
epoch-only metrics. Bound serialized metrics to 64 MiB; reject exceeding limits
explicitly instead of silently claiming successful training with missing curves.

## Executable bundled examples (accepted 2026-10-05)

All five current examples (local-training, mnist-mlp, mnist-vae, rnn-sine,
tiny-decoder-llm) require executable dataset adapters and every project-owned
PyTorch resource. Keep numerical behavior in resources and generic graph runtime;
no example-name or package-ID switches in backend. Core resources stay preserved.
Any numerical mismatch in legacy metadata is repaired explicitly in the example
project, with native loading and shape analysis checked again.

MNIST adapters use real MNIST samples with explicit provenance and bounded train,
validation and test subsets; classifier targets are digit IDs and VAE targets
are normalized flattened images. PNG/path/array inputs tokenize without a caller
creating a tensor; integer image pixels use the 0..255 scale, including images
whose brightest pixel is 1. Normalized floating arrays retain their 0..1 scale.
Decoding returns a useful label or reconstructed image array and rejects nonfinite
predictions, including token logits.
RNN uses explicitly generated sine-series data, split before extracting windows
to avoid overlap leakage. Token adapter uses a bounded Tiny Shakespeare excerpt
from Karpathy char-rnn, contiguous disjoint splits before next-token windows,
character vocabulary as declared inference asset, raw text input and text output.
Training payloads remain excluded from exported wheels; preparation occurs before
submission, never inside network-isolated workers. Data provenance, checksums,
split counts and small-run limits are visible and reproducible.

The editable tiny decoder retains explicit Repeat/HorizontalRepeat scopes and
Q/K/V graph structure; repair attention to Q K-transpose / sqrt(head width),
causal mask, softmax, then weighted V. A project-owned token cross-entropy
resource handles logits [B,T,V] and labels [B,T] with mean reduction, preserving
core Cross Entropy bytes. CPU tiny decoder uses character vocabulary 65,
context length 128, model width 64, two decoder repeats, four parallel attention
heads of width 16, and feed-forward width 256. Vocabulary comes from the complete
published corpus; only bounded disjoint text subsets enter training. The bundled
VAE samples its posterior during training and uses posterior mean for evaluation
and raw-input reconstruction. Document changed dimensions and update existing
native example assertions. Every example receives graph/gradient/adapter/wheel
tests and a real container training run; LLM receives a small multi-epoch run.

After the first implementation commit, the accepted full-corpus follow-up trains
only this unchanged LLM for 20 epochs on all 1,115,394 Tiny Shakespeare characters.
Split contiguous text 80/10/10 before window extraction; enumerate every complete
129-character next-token window with stride 128 in each split. No window crosses
a split boundary. Retain final short tails in provenance but omit incomplete
windows. Counts are 6971 train, 871 validation, 871 test windows of 128 tokens.
Use Adam learning rate 0.001, batch 64, seed 0, publication/validation cadence 100;
109 updates per epoch, 2180 total updates, plus unique epoch-tail publications.
Preparation verifies the published corpus SHA256 and creates a separate new
project directory; checked-in small fixtures remain unchanged. A reproducible
preparation command and frozen training configuration accompany the second
commit, while full corpus, weights and wheel stay in local job artifacts. The
generic adapter/runtime/model semantics do not change for this follow-up.

Accepted follow-up: editable projects live under `examples/models/`; all five
existing directories move there with their current files and graph layout
preserved. Update template loaders, preparation commands, tests and current
documentation together. Historical verification records keep their original
paths as evidence of the run, with a relocation note where needed.
`examples/implementation/llm/` is a standalone uv consumer of the completed
20-epoch job's downloaded wheel, independent of the repository SDK and backend
after download. Keep downloaded wheel and its bundled weights local and ignored;
commit the consumer code, uv metadata/lock and reproducible download instructions.
Download the exact job artifact through HTTP and verify its recorded SHA256
before installation. The uv project depends on that local wheel and locked CPU
PyTorch dependencies, never the editable runtime workspace. Demonstrate
`Model.inference(raw_text)` and `Model.infer(raw_text)`, default bundled weights
and optional compatible safetensors path. A greedy continuation may repeatedly
take the final decoded character and truncate context to 128 using these public
methods; do not reach into private graph/adapter attributes or add a model API.
Show that inference runs without a service call or source project access.
The local CPU demonstration uses two PyTorch threads. Prove absence of the
public SDK in the isolated verification environment; do not reject otherwise
valid inference merely because a user also installed that SDK.

## Generic graph execution and export

Resolve exact package id/version and declared dependencies from snapshot.
Load manifest pytorch entrypoint `build(parameters,context,services)`; existing
`stereotype_runtime.pytorch` ABI supplies DType/torch_dtype, BuildContext,
StereotypeReference and subflow/stereotype services. Defaults come from definition.
BuildContext.inputs is the count of incoming graph edges plus declared objective
external inputs for this instance. It describes actual positional forward inputs,
not a guessed package-specific arity. Preserve existing input/output metadata keys.
Referenced stereotype construction without a graph instance has no known input
arity; do not fabricate one. MatMul and other joins receive graph-derived counts.
Generic DAG schedule keyed by scope and source handle; numeric join handle
ordering, multi-output dicts, nested boundary mapping and independent repeat
instances. Each requested repeated body is freshly constructed through its
resource builders, including nested bodies; do not initialize every instance
by copying one already initialized template. Distinct random initialization
follows the frozen seed and each builder's normal behavior. ModuleDict/ModuleList
register all trainable parameters. Runtime-owned subflow runners inherit the
owning GraphModule's train/eval mode when evaluated. This also covers structural
meta runners used by preserved vectorized repeat modules, whose copies are not
registered for normal PyTorch mode propagation. Synchronize this boundary when
its mode differs; do not walk arbitrary resource attributes or change core code.
Root Input
reads binding and must reference a declared dataset input slot. Scoped Input
inherits the subflow value and does not require a dataset binding, in both server
metadata validation and numerical execution. Loss external inputs follow definition objective.externalInputs
batch.targets paths. Terminals collect values, never package-ID switches.
Reject cycles, orphan scopes, invalid handles/boundaries, unsupported runtime
entrypoints; do not silently reduce model to Sequential. Inference evaluates
prediction dependency closure, avoiding loss targets. Training includes objective.
This pruning applies recursively to mapped subflow terminals; batch targets and
training/output context flow through subflow services, never replaced with an
empty target map. In numerical inference a keyed result may omit declared loss
outputs that are not consumed; require declared normal outputs and reject unknown
keys/non-tensors. Training requires all outputs consumed by prediction/objective.
Opaque resource modules used on prediction path must support data-only forward;
an objective requiring an unavailable target fails clearly, never fabricates one.
This does not relax Lua shape-analysis result completeness.

SDK exposes `GraphModule(project_dir,core_dir=None)` and
`forward(inputs,targets=None,include_loss=False)->dict[str,Tensor]` with
`prediction` and optional `loss`. `load_dataset(project_dir)` constructs adapter.
`build_wheel(project_dir,core_dir,weights_path,output_dir,job_id)->Path` exports it.
Server validation reads metadata only; uploaded Python executes only in worker.

TODO(performance): replace interpreted generic DAG traversal with cached/lowered
execution or torch.compile/export after profiling; preserve handle, join,
subflow and adapter semantics. Initial generic executor is accepted release path.

Export valid wheel with job-specific import package `nnmodel_<job-id-normalized>`
exposing `Model(weights_path=None)`.

Accepted 2026-10-06: distribution identity is separate from this job-specific import package. New exports use
`nnm_<normalized-project-id>-0.1.0-py3-none-any.whl`, with the same distribution
name in METADATA and `<distribution>-0.1.0.dist-info`. Normalize frozen
`manifest.id` by replacing runs outside ASCII letters/digits with `_`, trimming
underscores and lowercasing; reject a missing/non-string ID or an empty result.
The `nnm_` prefix makes numeric project IDs valid distribution names. Version
remains the export format's existing `0.1.0`, not the model manifest version.
No example-specific naming or new API field is introduced. For project ID `llm`,
the wheel is `nnm_llm-0.1.0-py3-none-any.whl`; `nnm_llm.whl` is invalid.
Preserve job-specific imports, Model API, HTTP routes and snapshots. Existing
stored wheels remain downloadable unchanged; do not rename their bytes or
metadata. Qt uses the actual HTTP artifact filename. Rebuild the worker image
before training to activate this exporter change. Exports sharing a normalized
project ID share distribution identity and version; install different runs in
separate environments to avoid replacement/conflicting installations, and keep
artifacts in separate job directories. Normalization may also collapse distinct
IDs such as `a-b` and `a_b`; it is a readable label, not a unique job identity.

Default loads bundled weights.safetensors; explicit path loads alternate
compatible weights. `Model.infer(value)` and alias
`Model.inference(value)` call dataset tokenize, prediction under eval/inference
mode, then untokenize. Bundle exact graph, required Python resources, adapter,
private runtime and weights; wheel needs torch/safetensors plus declared resource
dependencies, not service or original repository. Include valid dist-info/RECORD.
Export the same activated package closure as GraphModule, including stereotype
references stored in node parameters and definition defaults, not only package
dependencies and visible node packages. Referenced join operators remain present
when an exported HorizontalRepeat constructs its modules.
Keep copied project files in a private snapshot directory separate from wheel's
Model/runtime/weights/core folders, so valid project-relative resource/asset paths
cannot overwrite generated package code or bundled core resources.
Never package training data, tokens, absolute paths or pickle checkpoints.

## UI and resource authoring

Training dashboard: endpoint/token, connection status, job history, submit controls,
refresh/poll, published-step learning curves and legacy epoch fallback, final test loss, downloads and
restore snapshot into chosen NEW project directory. Submit saves current edits
first. A visible Training toolbar button and Model > Training backend menu action
open the same dialog, including when no project is open. Submission requires a
current project; connection/job browsing remains available without one.
UI communicates saving before submission. Nonblocking network, bounded request/response,
visible errors. Polling preserves the user's metric scroll position and does not
replace unchanged metric text. Snapshot reload uses normal C application open and existing
unsaved-change confirmation; failure preserves active graph. Restore must never
overwrite source or existing destination; reject unsafe response paths.
Qt bounds: model JSON 8 MiB, resource bundle 256 MiB, each resource 128 MiB,
10,000 files, JSON responses 512 MiB and streamed artifacts 1 GiB. Backend
configuration may impose smaller limits; failures remain visible in UI.

Every newly authored dataset/stereotype includes standalone uv-compatible
pyproject.toml declaring nnmodelling-runtime dependency. Dataset scaffolds abstract
adapter methods with explicit NotImplementedError; stereotype pytorch build stub
also fails clearly until authored. This is initialization of uv project metadata,
not dependency installation from inside C. Resource transaction includes files
and manifest entrypoints atomically; existing metadata-only resources still open.

## Verification

Tests under tests: API auth/errors and bundle confinement, immutable restore,
job lifecycle/cancel/restart, actual worker training metrics, nested/branch/join
multi-output execution, standalone wheel install/inference and alternate weights,
generic adapter, resource rollback and Qt HTTP behavior. Run just test,
just test-ui, Python/backend recipes and git diff --check. Exercise actual
container job when runtime available; report environment blockers separately.
Repository skill `.agents/skills/nnmodelling-backend/SKILL.md` documents actual
uv/just startup, image, jobs, logs, shutdown and local lifecycle.
