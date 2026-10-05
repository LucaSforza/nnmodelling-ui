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
  `training` contains epochs, batch_size, learning_rate, seed. Response is Job.
* `GET /v1/jobs` returns `{jobs:[Job]}` for current owner.
* `GET /v1/jobs/{id}` returns Job: id, status, created_at, error, metrics.
  Status is queued/running/completed/failed/cancelled. Metrics is
  `{epochs:[{epoch,training_loss,validation_loss}],test_loss:null}`; test_loss
  becomes finite numeric value after final evaluation.
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

## Generic graph execution and export

Resolve exact package id/version and declared dependencies from snapshot.
Load manifest pytorch entrypoint `build(parameters,context,services)`; existing
`stereotype_runtime.pytorch` ABI supplies DType/torch_dtype, BuildContext,
StereotypeReference and subflow/stereotype services. Defaults come from definition.
Generic DAG schedule keyed by scope and source handle; numeric join handle
ordering, multi-output dicts, nested boundary mapping and independent repeat
instances. ModuleDict/ModuleList register all trainable parameters. Root Input
reads binding; loss external inputs follow definition objective.externalInputs
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
exposing `Model(weights_path=None)`. Default loads bundled weights.safetensors;
explicit path loads alternate compatible weights. `Model.infer(value)` and alias
`Model.inference(value)` call dataset tokenize, prediction under eval/inference
mode, then untokenize. Bundle exact graph, required Python resources, adapter,
private runtime and weights; wheel needs torch/safetensors plus declared resource
dependencies, not service or original repository. Include valid dist-info/RECORD.
Keep copied project files in a private snapshot directory separate from wheel's
Model/runtime/weights/core folders, so valid project-relative resource/asset paths
cannot overwrite generated package code or bundled core resources.
Never package training data, tokens, absolute paths or pickle checkpoints.

## UI and resource authoring

Backend dialog: endpoint/token, connection status, job list, submit controls,
refresh/poll, epoch training/validation losses, final test loss, downloads and
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
