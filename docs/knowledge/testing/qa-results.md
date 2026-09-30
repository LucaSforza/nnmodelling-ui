# UI/resources/VAE/local CLI QA — 2026-09-30

## Environment

Linux; GCC 16.2.1, Qt Widgets/Test 6.11.2. Desktop session is Wayland,
WAYLAND_DISPLAY=wayland-0. Visible launch used QT_QPA_PLATFORM=wayland; automated
interaction tests used offscreen. Desktop captures are 2000x1250 pixels;
offscreen captures use 1360x850 and 900x560 windows. Native device scale was
not explicitly measured. No core reference assets
were modified. Backend, numerical execution and training remain absent.

## Commands and results

* `just test`: exit 0, 8/8 C tests (model, catalog, inference, application,
  resources, automation, project, Lua smoke).
* `just build`: exit 0, C core, nnmodelctl and Qt client build without downloads;
  also rebuilt by the GUI test recipe after final changes.
* `just test-ui`: exit 0, 3/3 tests (Qt window, Qt canvas, real CLI/UI integration).
* `just test-cli`: exit 0, real nnmodelctl resource authoring, save/reopen,
  graph edits, dirty-replacement rejection, VAE scopes and screenshot capture.
* `just test-swarm`: exit 0, repository mailbox tests.
* `git diff --check`: exit 0.
* Clang 22.1.8 C-only build with AddressSanitizer/UndefinedBehaviorSanitizer,
  `ASAN_OPTIONS=detect_leaks=1 ctest --test-dir build/asan-clang -L core`:
  exit 0, 8/8 tests, no sanitizer/leak diagnostics. Initial GCC sanitizer
  configure failed because -lasan and -lubsan are missing; Clang runtimes worked.

## Functional evidence

Visual forms create stereotypes and datasets without editing definition JSON;
the default returned Lua rule infers a connected dataset-backed tensor. New
resource transactions persist unsaved nodes, reject bad Lua/dependencies/slots,
and retain graph/resources/dirty state after forced model.json replacement failure.
Symlink category directories are rejected and left untouched. Integer defaults
above 2^53 round-trip exactly through the Qt form. Saved JSON is indented two
spaces with a final newline. Created VAE copies include their own packages and
dataset metadata and all 21 nodes infer successfully.

Inference coverage includes root multiple Inputs, inherited nested Input tensors,
Repeated subflows, disconnected/incomplete boundaries, duplicate boundaries,
orphan scopes, semantic/runtime propagation, recursion and invocation limits.
MNIST MLP remains [B,10]; VAE encoder is [B,2,32], sampled shape [B,32],
decoder/reconstruction [B,784], KL [B]. These are types/shapes, not execution.

Socket tests cover 0600 mode, existing-path preservation, malformed JSON,
duplicate keys, NUL strings and idle clients. UI/CLI uses the same application
owner. Read-only inspection does not refresh away user selection. Dirty project
replacement requires save or explicit discard; replacement resets scope root.
Qt window tests cover save/cancel/close/reopen and invalid parameter preservation.
Zoom anchor assertions run offscreen too (no skipped original assertion).

## Visible review

Native Wayland startup, root/encoder/decoder navigation, committed screenshots
and project.close succeeded without platform/compositor errors. The standalone
`--capture ... --quit-after-capture` path also exited successfully. Evidence:

* `/tmp/opencode/nn-vae-reviewed-wayland.png`: root VAE.
* `/tmp/opencode/nn-encoder-reviewed-wayland.png`: mean/log-variance branch and
  ordered Gaussian join, with visible top/bottom feature badges.
* `/tmp/opencode/nn-decoder-reviewed-wayland.png`: nested decoder.
* `/tmp/opencode/nn-final-offscreen.png`: full editor/inspector normal size.
* `/tmp/opencode/nn-final-small-offscreen.png`: minimum 900x560 window.

Review found and fixed dark native button backgrounds, low-contrast package
labels, Proxy footer overlap, oversized-card layout overlap and global-cursor
zoom dependence. Package colors retain identity while text meets contrast on
white. Small-window controls remain usable through scrolling and toolbar
overflow; long labels are elided, and tall graphs require pan/zoom. Offscreen
form coverage and visible graph review do not claim exhaustive manual testing
of every native desktop dialog. Evidence is local temporary output, not tracked
project assets.

## SDL legacy cleanup — 2026-09-30

Removed the six unused SDL adapter/editor/entry-point/smoke sources listed in
the architecture and Qt migration plan. No active C core, Qt, build recipe,
package or vendored asset changed. Repository reference audit found only
historical/removal documentation references, with no remaining SDL adapter or
old editor references in active code/build/tests.

Post-removal verification:

* `just test`: exit 0, C-only configure/build and 8/8 core tests.
* `just test-ui`: exit 0, Qt configure/build and 3/3 GUI tests, including live
  CLI/UI integration.
* `git diff --check`: exit 0.

These are automated regression checks for source cleanup; no new manual desktop
or visual review was performed. The visible review above is prior Qt migration
evidence, not a newly repeated check.
