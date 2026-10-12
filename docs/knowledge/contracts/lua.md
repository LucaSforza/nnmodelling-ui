# Embedded Lua contract

Official Lua 5.5.1 source is compiled in-tree by `justfile` as a static C
library. The build
must not fetch Lua or link a system Lua. Its original MIT notice remains in
the vendored directory. See [decision](../decisions/native-client.md).

Package `manifest.json` declares exact `entrypoints.inference.file`. Loader
validates relative path, package identity/version, dependency closure and
file size before Lua activation. It never looks up a package by display name.
Each rule is an isolated callable `function(context, parameters, services)`.
The host supplies `tensor` primitives and a narrow `services` table. Host
tables contain validated typed values only. Rule result is `success` with a
tensor or keyed typed outputs per typed-outputs.md, `error` with message, or a
distinguished runtime fault. Missing input
or parameter is unresolved editor state, not a fake tensor.

Runtime implementation must limit memory and instructions, protect calls,
remove filesystem/process/network/module-loading APIs, and convert Lua errors
to structured diagnostics. No Python execution or fallback. A package may
not mutate model directly. Mutable Lua state must not leak between unrelated
package applications. Static dependencies resolve only within active core plus
model-owned set. This release runs inference locally; previous bootstrap only
proved interpreter linkage.

`src/inference/inference.h` exposes owned `NNInferenceReport *nn_infer_project(const
NNProject *)`, `nn_inference_free`, count and borrowed per-node result accessors.
Each result has node ID, status (`success`, `Lua compilation error`, `semantic
error`, `unresolved`, `runtime fault`), message and owned handle-keyed tensors
(plus a primary/terminal tensor view)
dtype/dimensions. Analysis
orders graph topologically; it never edits graph or project. Package paths come
from staged catalog, not ambient lookup. The application invalidates and lazily
reruns after semantic mutations.
Application now owns report caching/invalidation per diagnostics.md. Results
add Lua compilation error, stable code, source file/line and root cause node ID.
The report remains owned C data; successful tensors are separate from problems.

Formal: `infer(rule, inputs, params) -> Result<HandleTensorMap, CompilationError, SemanticError,
RuntimeFault, Unresolved>` and `modelAfterInfer=modelBeforeInfer`.
Natural: Lua computes type results but never changes graph.

Formal: `∀node: package(node)=(id,version)∈activeCatalog`, and
`compilationError ≠ runtimeFault ≠ semanticError ≠ unresolved`.
Natural: exact package identity and diagnostic class are preserved.

Accepted 2026-09-30: services.infer_subflow evaluates the caller's immediate
scope using inherited input and mapped typed terminals, with depth/invocation
limits and fault propagation specified in typed-outputs.md and
resource-authoring.md. New authored Lua
is validated as a bounded returned function before resource activation.
`nn_inference_validate_source(const char *source,char *error,size_t capacity)`
returns bool, compiles <=1 MiB source in protected isolated bounded Lua state,
evaluates only package initialization, requires a returned function, then frees
state. Application calls it before the project module stages resource files.
No project-module inference dependency is required.

Accepted 2026-10-03 after the GUI LLM trial exposed missing host functions used
by preserved core Embedding/Cast rules: `tensor.append_dimension(tensor,size)`
returns a fresh tensor retaining dtype and prior dimensions with one positive
integer dimension appended (maximum rank 64). `tensor.with_dtype(tensor,dtype)`
returns a fresh tensor retaining shape and replacing dtype with a nonempty
string. Invalid tensor/arguments/rank return nil plus an English error; neither
helper mutates its input. Both run inside the existing protected bounded Lua
state and preserve host allocation-failure reporting and temporary ownership.
`tensor.equal(first,second)`, used by the preserved core Add rule, compares
dtype, rank and corresponding dimension values exactly (numeric dimensions
compare by their integer value; symbolic dimensions by text), returning a
boolean or nil/error for invalid tensors. It performs no broadcasting or
numerical tensor comparison and never mutates either argument.
No numerical embedding/cast or model mutation is introduced. Core package
rules remain unchanged; the host must supply these declared primitives.

Accepted 2026-10-04: generic infer_stereotype reference service and separate
subflow visualization recipe runtime follow [3D contract](visualization-3d.md).
Inference never executes visualization entrypoints.

## Parameter counts (accepted 2026-10-08)

Package manifests may independently declare
`entrypoints.parameters={language:"lua",file:"parameters.lua"}`. This optional
entrypoint returns `function(context, parameters, services)` and runs in the
existing isolated bounded Lua sandbox, without Python, file or network access.
It returns `{status="success", total=N, trainable=M}` with nonnegative exact
integers and `M<=N`, or an explicit error/unresolved result. Missing entrypoints
mean unknown, never zero. Arithmetic overflow and malformed results are errors.
Counts exclude buffers and optimizer state and describe registered PyTorch
parameters, not per-token active MoE parameters. Context supplies inferred input
and output shape metadata when available; parameter-only formulas need not
require a complete graph. Rules receive effective validated defaulted parameters.

Only subflows receive `services.count_subflow()`: it recursively counts one
immediate body; the package rule expresses replication. Repeat multiplies one
body by times; Horizontal Repeat additionally counts its referenced join once
using `services.count_stereotype(reference)`. Proxy counts one body. Recursion
has depth 32 and a bounded invocation budget; malformed containment fails.
No package-ID branching in the C host. Each root occurrence contributes once;
subflow descendants are included through their owner, never counted twice.
The read-only application query returns owned JSON (caller frees with
nn_app_free_text), including total/trainable, completeness and per-node rows;
unknown contributions preserve known subtotal and mark aggregate incomplete.
Qt Model > Parameter counts displays total/trainable and node breakdown,
including nested owners; refresh queries C and never mutates graph or dirty state.

Accepted 2026-10-05: Concat exposes zero-based nonnegative axes and negative
axes from the end. Its rule normalizes axes to negative indices before calling
the existing tensor host (whose positive dimension indices are one-based).
This preserves unrelated rules and symbolic batch dimensions.

A `count_stereotype` reference inherits the calling graph node’s inferred
input/output context. Its target package validates owned parameters, but
does not reinterpret caller edge handles according to the target package
kind. In particular a referenced join module must not require the caller’s
ordinary `in` edge to be an `in-N` handle.
