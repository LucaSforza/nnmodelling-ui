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
in `src/gui/qt/` owns asynchronous HTTP presentation. The Training backend
dialog may explicitly launch and supervise the local FastAPI service as a
separate child process; Python never runs inside the native UI process. C project
module continues owning local resource creation.

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

Production training always runs in containers: Docker-compatible containers locally
or Singularity containers submitted through SSH to Slurm (accepted 2026-10-08).
There is no permanent local-process training alternative. Build CPU worker image via just recipe. Mount only
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

## SSH / Slurm execution (accepted 2026-10-08)

The FastAPI service, HTTP protocol, owner checks, persistent queue, snapshots and
artifact downloads remain on this computer. Deployment configuration selects
`NNMODELLING_EXECUTOR=docker` (default) or `slurm`; both execute the same worker,
graph, training configuration and wheel exporter. This is an execution location,
not a model semantic variant. No backend service is launched on the cluster.

Slurm configuration is service-private: SSH alias (`NNMODELLING_SLURM_HOST`),
absolute remote root (`NNMODELLING_SLURM_ROOT`), absolute prebuilt SIF image
(`NNMODELLING_SLURM_IMAGE`), partition (`NNMODELLING_SLURM_PARTITION`, explicitly configured), CPUs (default 2), memory (default 4G) and time (default 00:30:00).
Use existing SSH authentication and host verification; never copy credentials
into jobs or disable host-key checks. Fail visibly on missing configuration,
SSH, image, dependencies, scheduler failures and timeouts; never fall back to
local training. Health retains its `container` object and reports selected executor.

Local immutable snapshot is transferred into a new UUID-specific remote directory.
Slurm launches Singularity with clean environment, containment, an isolated
loopback-only network (`--net --network none`), read-only snapshot and worker
source, writable output, and bounded CPU/thread resources. The SIF
contains Python, torch, NumPy, safetensors and declared resource dependencies;
worker/runtime source is frozen per submitted job. Dependency preparation occurs
before training, never in the worker. Remote home, host secrets and sockets are
not mounted into the worker. Slurm enforces job resources/time. CPU training is
sufficient for bundled examples; GPU execution is outside this change.

Persist remote scheduler identity and execution configuration before observation;
status remains running while the launcher waits for the scheduler. Poll scheduler
with bounded SSH commands, publish complete valid metric files atomically to local
output during training, and retrieve logs, weights and wheel after termination.
Never extract remote links or traversal paths. Completion requires successful
scheduler exit and existing local result validation. Cancellation uses scancel;
service shutdown/restart cancels only its own recorded jobs and makes interrupted
work visibly failed. Do not cancel unrelated cluster jobs. Transport failure must
attempt cancellation and retain scheduler identity/error for diagnosis.

### Slurm walltime checkpoint and continuation (accepted 2026-10-08)

`SLURM_TIME` is a per-allocation limit, not a total training deadline. Each batch
allocation requests an early `SIGUSR1` warning. The batch wrapper turns that
signal into an attempt-scoped request file in the persistent remote job output;
the wrapper exports its absolute path as `NNMODELLING_CHECKPOINT_REQUEST` and
its allocation ID as `NNMODELLING_SCHEDULER_ID`, and
the worker notices `checkpoint-request.<scheduler-id>` at the next
optimizer-step boundary, publishes current metrics, and atomically writes
`checkpoint.safetensors` plus `checkpoint.json` capped at 64 MiB before exiting with
reserved handoff exit code 75. Signal reserve is at most 60 seconds and reduced
for shorter allocations. Checkpoints use safetensors plus JSON metadata, never
pickle. They include local job and scheduler IDs, model and Adam state,
Python/NumPy/torch RNG states, global step, epoch/batch cursor, and in-progress
weighted metric windows. The manifest has `version`, `job_id`, `scheduler_id`,
`global_step`, `tensor_size` and `tensor_sha256`; it is committed last and hashes
the tensor payload. Incomplete or mismatched pairs are rejected.

The next Slurm allocation reuses the same UUID-specific remote job directory and
loads that checkpoint. It recreates the current training iterator from the saved
epoch-start RNG state, skips already-consumed batches, then restores checkpoint
RNG state before continuing. Dataset `load("train", batch_size)` must therefore
be repeatable from the seeded Python, NumPy and torch RNG streams within an epoch;
custom adapters with external or non-replayable iterator state are not resumable.
Successful continuation preserves the original local job ID and accumulated
metrics. Each scheduler allocation gets its own recorded Slurm ID; the UI shows
one logical job. Weights and wheel are still produced only after final test
evaluation and successful completion. Checkpoint files are internal and never
served as completed artifacts.

Automatic continuation occurs only after a complete checkpoint is validated and
the current worker exits with Slurm state `FAILED` and exit code `75:0`. A walltime kill before
that safe boundary, manual cancellation, backend shutdown/restart, missing or
invalid checkpoint, or non-replayable dataset failure remains visible as failed
or cancelled and is never silently reported as resumed. The local backend must
remain running between allocations; application restart recovery remains the
existing interruption behavior. A checkpoint is saved on the warning signal,
not periodically, and may lose at most the current in-flight batch before that
boundary.

## Qt local backend configuration (accepted 2026-10-08)

The Training dashboard and local-service configuration are separate windows.
The dashboard keeps its endpoint, connection status, training controls, job
history, curves and artifact actions visible without service-launch fields
occupying its workspace. A dedicated modeless configuration window is opened
from the dashboard and can be moved independently while the dashboard remains
usable. Its executor choices are local Docker-compatible runtime and
SSH/Slurm; selecting Slurm configures only the local service to submit workers
remotely. The configuration window never launches a backend on the cluster and
always binds a UI-launched service to loopback. Existing arbitrary endpoint/token
connection remains supported from the dashboard.

On managed startup, the service manager reserves an available loopback port and
sets `Endpoint` to that address automatically. Users do not need to guess or
enter the managed service port; manually launched services keep the configured
backend default unless changed explicitly.

Persist non-secret launcher settings per user in the shared JSON profile below: executor,
local job root, Docker runtime/image, or Slurm SSH alias, absolute remote job
root, absolute prebuilt SIF image and Slurm resource limits. The UI uses existing
SSH agent/configuration and host-key checks; it never collects SSH passwords or
stores credentials. Bearer token remains session-only. `uv` and the repository
workspace are required to launch the service; worker images are prepared
separately. Missing prerequisites and child startup/health errors are visible.
Configuration is passed to the child process environment, not written into
project files or sent through training HTTP routes.

The local service process survives closing the Training dialog, but is owned by
the main application. Application shutdown while that process is active warns
that running Slurm jobs will be interrupted and requires explicit confirmation.
Restarting a managed service is allowed only when its job list has no queued or
running jobs. Manually launched services and arbitrary remote endpoints are never
stopped or reconfigured by the UI. This is lifecycle control for the supported
repository workspace, not a general Python/backend installer or remote daemon.

The standalone `examples/implementation/llm/` consumer may be repinned to a new
verified cluster-produced wheel; recorded full-corpus historical proof remains.
Add `examples/implementation/vae/` as an isolated uv consumer with a verified
HTTP-downloaded wheel, bundled/default and explicit safetensors, reconstruction,
`encode` and `decode` (including a seeded prior sample). It uses public Model APIs,
validates finite results and writes PNGs. Both consumers run without backend or
source checkout after artifact download. Keep wheels, environments and images
ignored; freeze distribution/import identity and SHA256 in consumer metadata.

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
exposing `Model(weights_path=None)`. Existing import normalization replaces non-ASCII
identifier characters with underscores, trims and lowercases; when its first
character is not a letter, it inserts `job_` before the normalized job identity.
Read the actual wheel package when pinning consumers (numeric-leading UUIDs
therefore import as `nnmodel_job_<uuid-with-underscores>`).

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
10,000 files, JSON responses 512 MiB and streamed artifacts default 64 GiB (configurable independently of corpus size). Backend
configuration may impose smaller limits; failures remain visible in UI.

Every newly authored dataset/stereotype includes standalone uv-compatible
pyproject.toml declaring nnmodelling-runtime dependency. Dataset scaffolds abstract
adapter methods with explicit NotImplementedError; stereotype pytorch build stub
also fails clearly until authored. This is initialization of uv project metadata,
not dependency installation from inside C. Resource transaction includes files
and manifest entrypoints atomically; existing metadata-only resources still open.

## Named project operations in exported wheels (accepted 2026-10-06)

`model.md` defines optional `manifest.operations` records and endpoint/codec
validation. `GraphModule.run_operation(name, tensor)` binds one batched Tensor
at a root Input output or a root node input handle, cuts that boundary's
incoming edge only for this call, then evaluates the requested output's
backward dependency closure. Exported `Model.run_operation` applies the
operation's dataset/tensor codec before and after this graph call. Normal
`forward()` remains unchanged.
Calls run with the same registered modules and loaded `state_dict`; loss targets
are not fabricated or evaluated.
Each operation call starts a fresh subflow invocation budget, as `forward()`
does; the 256-invocation bound applies within one execution, not across calls.
Codec metadata must be a string equal to `dataset` or `tensor`; other JSON
types fail validation with HTTP 422 on submission and ValueError in the runtime.

Wheel `Model` exposes declared safe operation names as methods and retains
`run_operation(name, value)` for generic callers. The operation table is frozen
in the saved project snapshot and validated before job submission/export.
`dataset` codecs reuse active `tokenize`/`untokenize`; `tensor` codecs accept or
return batched PyTorch tensors. Only one input and one output are supported per
operation in this release. Operation records contain no code, filesystem paths,
or secrets and do not activate packages beyond the graph's existing package
closure. Wheels remain independent of the backend and source project.

The MNIST VAE example declares `encode` (dataset image -> `sample.out` Tensor)
and `decode` (`decoder.in` Tensor -> `decoder.out` dataset image). Evaluation
mode makes its reparameterization module return posterior mean, so `encode` is
deterministic. Generating a prior sample uses an ordinary tensor supplied to
`decode`, such as `torch.randn(1, 32)`. No VAE package-ID branch or extra random
sampling API is added to the runtime.

## Verification

Tests under tests: API auth/errors and bundle confinement, immutable restore,
job lifecycle/cancel/restart, actual worker training metrics, nested/branch/join
multi-output execution, standalone wheel install/inference and alternate weights,
generic adapter, named operation slicing and wheel methods, operation metadata
round-trip/validation and Qt manager flows, resource rollback and Qt HTTP behavior. Run just test,
just test-ui, Python/backend recipes and git diff --check. Exercise actual
container job when runtime available; report environment blockers separately.
Repository skill `.agents/skills/nnmodelling-backend/SKILL.md` documents actual
uv/just startup, image, jobs, logs, shutdown and local lifecycle.

## GPU and remote corpus acquisition (accepted 2026-10-08)

GPU training is optional and explicitly selected. Slurm configuration
adds `NNMODELLING_SLURM_GPUS` (integer 0..8, default 0); nonzero requests
`--gres=gpu:N`, enables Singularity `--nv` and sets worker default device to CUDA.
Qt Training backend configuration exposes GPU count and persists it with the
existing non-secret Slurm profile. CUDA requests fail if CUDA is unavailable;
there is no silent CPU fallback. Existing CPU jobs retain their semantics.

Dataset manifests may declare optional `remoteFiles` with public Hugging Face
dataset repository `repo`, immutable 40-hex `revision`, and `files`, an ordered
list of `{path,destination,size,sha256}` records. Paths are confined relative
POSIX paths, destinations unique, sizes positive exact integers and hashes
64-hex. Arbitrary URLs, credentials, branch names, symlinks and traversal are
rejected. `destination` is relative to the acquired data root. The submission
contains only resource code, metadata and tokenizer inference assets; corpus
bytes are excluded from the native upload. Bundle limits remain for code and
metadata, not remotely acquired corpus size. No corpus aggregate size cap is
imposed. Disk space and download errors remain visible.

Only Slurm supports this acquisition initially; Docker rejects it explicitly.
Before network-isolated training, a separately recorded Slurm preparation
allocation downloads files directly on the cluster via HTTPS Hugging Face
resolve endpoints, streaming to sibling temporary files, validating exact size
and SHA256, then atomically publishing. Existing verified files are reusable;
incomplete/changed files are never consumed. Preparation never runs model or
dataset code and never receives local credentials. Its scheduler identity is
owned by the same backend cancellation/lifecycle machinery. Training remains
network-isolated and reads the verified data root via a read-only bind.
The runtime DatasetAdapter exposes `data_directory`, defaulting to the resource
directory; a worker-provided `NNMODELLING_DATASET_DATA_ROOT` selects the confined
remote data root. Constructor and inference still require only inference assets.

Slurm artifacts and checkpoints exceed 1 GiB for a 1B model. Retrieve final
weights/wheel with bounded-memory streaming and atomic publication. Checkpoint
handoff validates identity/progress/size/hash remotely and retains tensors on
cluster across allocations; do not round-trip optimizer tensors per allocation.
Artifact safety bounds are configurable independently of corpus acquisition,
with a default 64 GiB maximum file/aggregate, safe filenames and no links.
Completion still requires verified metrics, safetensors and standalone wheel.

Training config accepts optional `device` (cpu/cuda), `precision`
(float32/bfloat16), `optimizer` (adam/deepseek-muon), `weight_decay`,
`gradient_clip`, `warmup_steps`, `max_steps` (0 means all epochs) and
`gradient_accumulation_steps`. Existing four/five-field requests retain defaults.
Qt submission exposes these controls; GPU/backend configuration is visible.
`max_steps` is an explicit optimizer-step budget, never labelled full-corpus
completion. Metrics record processed tokens and optimizer updates. Validation
and test remain independent held-out adapter splits.

DeepSeek Muon follows paper Algorithm 1: momentum 0.95, Nesterov, Frobenius
normalization, eight Newton-Schulz steps with (3.4445,-4.7750,2.0315), then two
with (2,-1.5,0.5), update scaling sqrt(max(n,m))*0.18 and decoupled weight decay.
Each logically independent matrix is orthogonalized separately, including each
expert or grouped matrix slice. Embedding/head, normalization, mHC static
biases/gates and all non-matrix parameters use AdamW. Resource modules declare
AdamW exclusions via parameter attributes; the generic optimizer never
branches on model/package IDs. AdamW uses betas (0.9,0.95) and epsilon 1e-20 as section 4.2.2. State dict supports checkpoint round-trips.
CUDA RNG and accumulation cursor are retained across allocation handoffs;
checkpoint only after an optimizer update. Data and targets move to selected
device; BF16 uses autocast with FP32 master weights, norm, loss and optimizer
operations as required. CPU existing tests remain supported.

Accepted 2026-10-08: worker sets `NNMODELLING_DATASET_CACHE_ROOT` to a dedicated
job-output `dataset-cache` directory. DatasetAdapter.cache_directory exposes
this optional writable scratch root; it is derived tokenization state, never
an inference asset or downloaded corpus. Adapters may atomically publish
revision/config-keyed token caches there. Cache reads validate shape/dtype and
source/config identity. Cache resumes avoid replaying expensive Parquet
conversion across 30-minute allocations. No cache bytes are served as final
weights/wheel, and raw acquired files remain read-only.

## Exported model device (accepted 2026-10-08)

`Model(weights_path=None, device="cpu")` preserves CPU defaults and accepts a
PyTorch CPU or CUDA device. CUDA selection must fail explicitly when unavailable.
The constructor loads safetensors, moves the registered graph to the chosen
device and sets eval mode. Dataset-tokenized tensors and tensor operation inputs
move to that device before execution; dataset decoding receives CPU tensors.
Tensor-codec operation outputs remain on the selected device. No private graph
access is required by inference examples. No dtype/compile policy is added.

GPU Slurm allocations request their handoff signal up to 300 seconds before
walltime, bounded to at most one quarter of the allocation. CPU allocations
retain the existing up-to-60-second reserve. GPU reserve covers a full gradient
accumulation group, validation publication and multi-GiB checkpoint persistence;
workers still stop only after an optimizer update.

## Declared external batch inputs (accepted 2026-10-08)

Existing definition.objective.externalInputs may bind either batch.targets.NAME
or batch.inputs.NAME, appending tensors after connected graph inputs in declared
order. Submission validates the chosen map against selected dataset metadata.
Numerical execution preserves root input bindings through nested subflows,
separately from their immediate tensor boundary. Missing named tensors fail
explicitly. Target-only behavior and flatten_batch remain unchanged.
Named operations whose input endpoint is a bound root Input seed that named
batch input for the whole call. An operation cutting an intermediate node has
no implicit root batch binding; an external-input consumer fails clearly when
its required binding is unavailable. Context state resets after each call and
on error; no cross-call or cross-model tensor leakage. This generic mechanism
supports visible hash-MoE nodes without a model-ID branch or floating token IDs.

Accepted 2026-10-09: a Slurm worker may not have published `metrics.json` during
startup or while queued. The remote publication command uses a distinct exit
status for this absence; the runner treats it as no publication regardless of
SSH informational stderr. Other command failures, linked files, oversized data
and invalid JSON remain errors. SSH warnings do not change a successful remote
command's meaning. This preserves startup without masking transport failures.


## Stable wheel imports and early stopping (accepted 2026-10-09)

Explicit user request supersedes the job-specific-import requirement above for
new exports. New wheels expose Model from the same `nnm_<normalized-project-id>`
Python package used by their distribution. Keep the previous job-specific name
as a thin compatibility alias within a newly exported wheel. Existing archived
wheels retain their original bytes and names. Installing a replacement wheel
for the same project replaces that model without changing caller imports; use
a fresh interpreter and force reinstall/relock when version 0.1.0 is unchanged.
Consumers support selecting and installing a downloaded wheel without editing
Python code or dependency source paths. Historical legacy wheels remain usable
through distribution metadata discovery, never by hardcoding a job ID.

Training accepts `early_stopping_patience` (integer 0..10000; default 0 disables)
and `early_stopping_min_delta` (finite nonnegative number; default 0). When
enabled, evaluate stopping once per completed epoch using the existing full
validation result, independently of step-publication cadence. Save every strict
new validation minimum; reset patience only for improvement greater than
min_delta relative to the previous best. First epoch establishes the baseline.
Stop successfully after patience consecutive epochs without such improvement.
Restore the minimum-validation weights before the single final test and export,
also when the epoch/step budget ends. No test result selects weights.
Checkpoint handoff preserves best tensors, best epoch/loss and patience count;
resume must produce the same chosen weights and stopping epoch as uninterrupted
training. Legacy checkpoints and disabled requests retain existing behavior.
Metrics add optional `early_stopping` with best_epoch, best_validation_loss,
non_improving_epochs and stopped_epoch (null unless patience ended training).
Reader validates these fields and keeps them visible; old records remain valid.
Qt exposes patience and min delta with explicit units/defaults and displays
selection/stopping evidence. Nonfinite or malformed settings fail visibly.

Tiny Shakespeare adapter shuffles training windows with the seeded torch RNG per
epoch, keeping validation/test ordered. Replay from epoch-start RNG recreates
the same training order on Slurm handoff. Existing stochastic text decoding
remains unchanged.


## Portable per-user service profile (accepted 2026-10-12)

The single launcher/service profile is `$XDG_CONFIG_HOME/nnmodelling/config.json`
when XDG_CONFIG_HOME is set, otherwise `~/.config/nnmodelling/config.json`.
`NNMODELLING_CONFIG` selects an explicit alternate file. Missing default profile
uses local Docker defaults; an explicit missing file, malformed JSON, unknown
profile keys or invalid field types/ranges produce visible errors before launch.
No institution, account, SSH alias, remote path or partition is preconfigured.
Host, remote root, image and partition are empty until explicitly supplied.

The root JSON object contains `backend` and `slurm` objects. `backend` contains
`executor`, `job_root`, `container_runtime`, `worker_image`; `slurm` contains
`host`, `root`, `image`, `partition`, `cpus`, `memory`, `time`, `gpus`,
`ssh_executable`. Strings are JSON strings; cpus/gpus are integers (cpus 1..256,
gpus 0..8). Executor is docker or slurm. Local job root expands a leading tilde.
Defaults retain local Docker, user-local job storage, two CPUs, 4G memory,
00:30:00 walltime, zero GPUs and ssh. No credentials are profile keys.
Standalone service precedence is explicit NNMODELLING_* environment > JSON >
generic defaults. Bearer credentials stay environment/session-only.

Python `backend.user_config` uses the standard json module for profile loading
and validation. Qt uses its existing QJsonDocument and QSaveFile directly for
profile reading and atomic updates; no parser dependency or subprocess helper.
Both implementations enforce the same schema and defaults. UI shows profile
path, loads non-secret fields, and saves atomically before starting/restarting
the managed local service. Updates preserve unrelated root JSON settings.
Invalid/unreadable profiles are never silently replaced; load/save errors prevent
managed launch. Reopening configuration retries a failed load so users can fix
the file externally. Prior Qt launcher settings are no longer read: users
configure the portable profile explicitly. UI-selected fields become explicit
child environment overrides. Existing backend credentials remain absent from
disk and logs. Profile errors must not echo configured values.

The removed large model example and its dedicated resources/consumers/probes
are no longer shipped. Generic parameter counts, GPU support, remote acquisition
and optimizers remain supported. Verification uses local bounded tests; remote
training needs an explicitly configured reachable cluster.
