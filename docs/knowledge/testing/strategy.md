# Verification strategy

Core gate: `just test` configures a C-only build without Qt/C++ and runs Lua,
model, catalog, project, application and inference tests. Existing project test
uses its catalog fixture. Application tests use real packages and verify C-only
validation/defaults, mutation failure atomicity, joins, scopes, dirty state and
save/reopen. Preserve existing test coverage.

GUI gate: `just build` and `just test-ui` compile Qt Widgets frontend and run
separate offscreen Qt interaction tests. Exercise real scene item movement,
connections, selection/deletion, scope navigation and project lifetimes. Core
tests never instantiate QApplication. Capture actual widget rendering and
inspect at normal/small size; offscreen is not proof of desktop integration.
Run `git diff --check`. Report missing tooling/desktop limitations honestly.
