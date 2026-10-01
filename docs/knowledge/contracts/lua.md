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

`src/inference.h` exposes owned `NNInferenceReport *nn_infer_project(const
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
