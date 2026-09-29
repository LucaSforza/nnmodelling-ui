# Native client bootstrap plan

Historical bootstrap record; superseded by [Qt migration](ui-rewrite.md).
Status at bootstrap: architecture and minimal scaffolding. Current user scope: UI/client
only; no backend implementation.

## Completed in this phase

1. Inspect legacy repository at pinned commit; classify current package,
   graph, browser, command and backend concepts.
2. Copy dual license, immutable `stereotype-packages/core` and original
   Visual Paradigm file; transcribe historical UML and reconcile native UML.
3. Establish KB, contract-first skill, root agent guidance and this plan.
4. Vendor official Lua 5.5.1 source, compile in-tree with `just` and prove C
   embedding with test source under `tests/`.
5. Establish SDL3 client target through private platform adapter. Main and
   public headers never include SDL3. Headless open/close lifecycle smoke
   passes with installed SDL3 3.4.16.

## Milestone 1: narrow native editor slice

Prerequisite: update any ambiguous public operations in KB before code.
Implement one typed in-memory graph with two model-backed package nodes and
one edge, draw through graphics layer, pan/zoom, select/drag, render diagnostics.
Use actual core package definitions; no package-ID switch. Establish graph
operations and tests first. Type inference follows only after Lua tensor host
contract is complete. Do not start command server or backend connection in
this milestone.

Suggested narrow implementation tasks for GPT-6 Luna after principal updates
contracts: (1) typed graph and validation, (2) SDL graphics and camera,
(3) editor interactions, (4) Lua package host and conformance tests. Each task
must cite relevant KB/UML and exact acceptance criteria. Principal reviews
each diff and tests against KB before next task.

## Deferred

Project persistence, custom package authoring, `nnmodelctl`, local IPC,
backend connection, training, datasets and wheel download. They require
separate accepted contracts. `nnmodelctl` conceptual design is documented to
preserve one application authority, not an implementation authorization.

## Open questions

- Exact model JSON compatibility and migration policy for native persistence.
- Cross-platform local automation transport and command protocol.
- Future backend endpoint, authentication, training UX and response semantics.
- Text/font dependency and UI accessibility API for implemented widgets.
- Whether package Lua 5.5 incompatibilities arise in full corpus; use corpus
  tests before treating runtime substitution as semantic parity.
