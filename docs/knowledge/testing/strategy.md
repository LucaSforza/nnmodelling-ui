# Verification strategy

Local backend (accepted 2026-10-05): [backend contract](../contracts/backend.md)
adds Python API/worker/runtime/wheel tests, Qt HTTP tests, resource uv scaffolds
and actual container training gate. A direct worker test is diagnostic evidence;
only actual container job proves deployment path. Keep service running locally
when requested and document startup/shutdown in backend management skill.

Core gate: `just test` configures a C-only build without Qt/C++ and runs Lua,
model, catalog, project, application, resource, automation and inference tests. Existing project test
uses its catalog fixture. Application tests use real packages and verify C-only
validation/defaults, mutation failure atomicity, joins, scopes, dirty state and
save/reopen. Preserve existing test coverage.

GUI gate: `just build` and `just test-ui` compile Qt Widgets frontend and run
separate offscreen Qt interaction tests. Exercise real scene item movement,
connections, selection/deletion, scope navigation and project lifetimes. Core
tests never instantiate QApplication. Capture actual widget rendering and
inspect at normal/small size; offscreen is not proof of desktop integration.
Both `just build` and `just core` export the active CMake compilation database
to the repository-root `compile_commands.json` for clangd; the file is generated
and ignored by git.
Run `git diff --check`. Report missing tooling/desktop limitations honestly.

Source reorganization gate (2026-10-02): source_layout runs in both configurations,
auditing the module directories, private-header boundaries, separately compiled
units and explicit target source coverage. `just test-sanitize` configures a
separate instrumented C-only build using Clang by default (override
NN_SANITIZER_CC), then runs the same core suite with ASan/UBSan and leak checks.

Resource gate covers indented JSON, positioned definition metadata, typed
authoring defaults, invalid Lua/dependency/schema/slot rejection, symlink
confinement, save rollback (disk and dirty flag), save/reopen and VAE copying.
Inference tests cover inherited subflow tensors, repeated/nested scopes, all VAE
shapes, empty/duplicate boundaries, orphan scopes and depth/invocation budgets.
`just test-cli` runs the actual UI and nnmodelctl, creates and reopens resources,
authors/connects nodes, navigates VAE scopes and captures a committed screenshot.
The same end-to-end test is part of `just test-ui`. Socket tests verify private
permissions, existing-path preservation, malformed requests and idle clients.
Diagnostics gates additionally cover cached reports, Lua compile locations,
causal propagation, optional error buffers, partial allocation cleanup, text
ownership, retained authoring forms, severity markers, filters and CLI parity.

Typed outputs (2026-10-01): test normalized defaults/overrides and duplicate
type/ID rejection, source-handle-sensitive tensors, terminal-only compatibility,
single-input collectors, root completion, transactional subflow terminal spawn,
mapping persistence/diagnostics, two-output recursion, circle/edge/draft colors,
output form and CLI parity. Rewrite templates/fixtures; no legacy migration.

Editor revision gate (2026-10-03): orthogonal segment/obstacle tests, unrelated
node rerouting, hierarchical scopes, camera-only Fit, read-only expansion and
C-backed multi-output hover. Browser Computer Use through existing .computer-use
Qt VNC/noVNC integration verifies actual menus, grid, pan/zoom, previews, routing,
Arrange and Fit. Project-local launch skill owns repeatable bridge lifecycle.

History gate: graph commands, deep snapshot values/topology, grouped drag/Delete/
Arrange atomicity, failed/no-op redo preservation, save/undo/redo dirty revisions,
project replacement and resource barriers, C-only and sanitizer tests. GUI checks
Edit actions/shortcuts, rerouted restored geometry and scope fallback. Shared
layout tests check nonoverlap, feed-forward ranks, branch/join centering and compact
preview bounds without C coordinate mutation; restart Qt/bridge after every change.

Dataset manager gate (2026-10-05): verify prefilled metadata editing, stable
identity/active selection, preserved opaque root/batch/slot metadata and Python/data
assets, normal write/save rollback, symlink rejection and save/reopen. GUI exercises
Dataset/New dataset entrypoints, empty selection, Edit/Select, invalid retained
forms and cancellation. Use a disposable project for real browser UI editing.
