# Typed outputs implementation plan

Accepted 2026-10-01. Principal owns normative contract/UML, assets/examples,
automation integration, review and final verification. Commit KB before code.

Three bounded GPT-6 Luna background assignments, disjoint ownership:

1. C catalog/model/project/application: normalized output definitions, validation,
   boundary mapping persistence/API, transactional subflow terminal spawn and
   root seeding; focused catalog/application/resource/model/project tests.
2. C inference/report: per-handle tensors, strict Lua results, mapped recursive
   subflows, root/nested completion diagnostics and inference/allocation tests.
3. Qt frontend: circle boundaries, typed ports/edges/drafts, outputs authoring,
   boundary inspector and multi-output inspection; Qt tests.

Each worker personally reads the named KB/UML and git show/git diff of this KB
commit versus its parent before implementation. No workers change docs/assets
or other workers' files. Principal supplies new core.loss-output and rewrites
MLP/VAE/assets, CLI metadata/results/mapping and end-to-end tests while workers
run. Preserve historical core assets; no legacy model migration. Run just test,
just test-ui, sanitizer core gate, diff --check and inspect rendered evidence.

## Implementation and review

KB/UML committed before delegation as `a846cca`; graph-level empty-root
completion clarified in `648c1d3`. Three registered GPT-6 Luna sessions read
the contract diff and implemented disjoint C topology, C inference and Qt
assignments. Principal supplied assets/examples and CLI integration.

Implemented normalized defaults/overrides, transactional mapped terminals,
source-handle inference, circle/typed fanout rendering, resource form, inspector
and CLI parity. Principal review corrected rollback collision safety, candidate
edge validation, exact output ID/key handling, boundary diagnostic visibility,
label/hotspot geometry and queued inspector refresh. Sanitizer review identified
OOM-path qsort and temporary-cause cleanup defects, corrected before release.

Status: completed. Final principal `just test` (10/10), `just test-ui` (3/3),
`just test-swarm` (11/11), fresh Clang ASan/UBSan/leak core gate (10/10) and
`git diff --check` passed. Actual Wayland and small offscreen screenshots were
reviewed; evidence and limitations are in docs/knowledge/testing/qa-results.md.
