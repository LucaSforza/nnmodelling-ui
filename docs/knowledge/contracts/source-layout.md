# Source modules and private helpers

Accepted 2026-10-02 from the user's source-organization request. This is a
behavior-preserving refactor: public C symbols/types, ownership, limits, failure
atomicity, schema, Lua isolation and UI/CLI behavior remain unchanged. Backend
and training remain deferred. Historical assets and vendored sources are not moved.

## Layout and boundaries

All application implementation lives under `src/`, one directory per module:

| Directory | Responsibility / independent translation units |
| --- | --- |
| `model/` | graph lifecycle/snapshots, nodes, edges/DAG, typed values/parameters |
| `catalog/` | lifecycle/loading, manifest, normalized definitions and shared package queries, dependencies |
| `project/` | lifecycle/open, graph JSON, atomic save, datasets, resource transactions, templates, filesystem helpers |
| `inference/` | report/results, scope evaluation, rule execution, Lua sandbox/budget, Lua tensor operations, owned tensors |
| `application/` | lifecycle/cache, graph commands, parameter validation/text/defaults, ports/boundaries, resource commands |
| `automation/` | protocol/dispatch, project snapshot, diagnostics JSON, nonblocking Unix transport |
| `nnmodelctl/` | CLI arguments/request construction, bounded Unix exchange, entry point |
| `utils/` | global allocation-free errors, owned text/path joining, join-handle parsing |
| `gui/qt/` | Qt frontend only: existing canvas classes; window composition, project actions, inspector/resources, diagnostics, authoring dialogs, UI automation |

Public headers stay named after their module and live inside its directory.
Cross-module includes are explicit (`application/application.h`, etc.) relative
to the exported `src` include root. No flat compatibility forwarding headers or
per-module public include search directories conceal dependencies. GUI classes
remain accessible through the canvas target's `src/gui/qt` include root.
C++/Qt remains confined to `src/gui/qt`; core/CLI remain C11 and build without
Qt or a C++ compiler. CMake lists translation units explicitly, sharing those
lists with fixtures such as the catalog-stub project test; no `.c` inclusions,
unity-build splitting, generated fragments or source globs.

## Private helpers

An opaque owner's concrete representation may be in `<module>_internal.h`
inside its module. That header is private, never included by another module,
GUI or tests. Share only types/prototypes actually needed by sibling units.
Helpers used by one unit remain static. Helpers shared by sibling units use
module-qualified names (e.g. `nn_project_*`) and a private header, not public
ABI additions. Module-wide mechanical helpers may live in
`<module>_utils.c/.h`; concept-specific helpers belong to that concept's unit.
Do not copy implementations into multiple units, make a kitchen-sink helper
file, or move domain validation into global utils. Global helpers are included
as `utils/utils.h`; local helpers have distinct names and include guards.
Private headers must not bundle every library/system header for convenience.

Splitting is by responsibility, not arbitrary line chunks; keep independent
concepts readable and avoid replacing one monolith with another. Tiny cohesive
units need not be split further. Do not change allocation order or error
classification merely to obtain a cleaner split. Preserve test behavior and
test sources under `tests/`.

## Verification

Run `just test`, `just test-ui`, `just test-cli`, a fresh C-only build, sanitizer
checks, source-layout checks and `git diff --check`. Inspect rendered UI evidence
and report desktop/tooling limitations accurately. Update paths in build,
current documentation and agent guidance; preserve clearly historical paths.
The existing [architecture diagram](../architecture/overview.md) and
[editor/graphics UML](../uml/editor.md) describe these boundaries; update those
diagrams rather than creating another overlapping UML document.

Accepted 2026-10-04: `visualization/` owns C11 expanded scene/composition,
layout and camera/projection/picking per visualization-3d.md. Qt adapter stays
in gui/qt. No graphics-library dependency enters core.
