# Verification strategy

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

Resource gate covers indented JSON, positioned definition metadata, typed
authoring defaults, invalid Lua/dependency/schema/slot rejection, symlink
confinement, save rollback (disk and dirty flag), save/reopen and VAE copying.
Inference tests cover inherited subflow tensors, repeated/nested scopes, all VAE
shapes, empty/duplicate boundaries, orphan scopes and depth/invocation budgets.
`just test-cli` runs the actual UI and nnmodelctl, creates and reopens resources,
authors/connects nodes, navigates VAE scopes and captures a committed screenshot.
The same end-to-end test is part of `just test-ui`. Socket tests verify private
permissions, existing-path preservation, malformed requests and idle clients.
