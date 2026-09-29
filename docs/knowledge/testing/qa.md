# Native client QA

Status: release gate. Run after implementation, record result and evidence in
`docs/knowledge/testing/qa-results.md`. A build or dummy-driver smoke test does
not establish that visible application works.

## Environment and launch

1. Run `just test`, `just test-ui`, `just build` and `git diff --check`; record
   compiler, Qt version and failures.
2. On actual desktop session, run client with `QT_QPA_PLATFORM=wayland` and
   without driver override. Confirm window appears, responds, paints every
   panel and closes cleanly. Record `XDG_SESSION_TYPE`, `WAYLAND_DISPLAY`, Qt
   platform plugin and any compositor error. If Wayland fails, diagnose
   Qt error and check X11 fallback separately; do not count offscreen plugin as
   visible launch.
3. Capture screenshot from visible window after a full frame. Inspect text
   legibility, clipping, node/edge placement, panel bounds, contrast and font
   glyphs. Test at 1280x800 and a smaller window; note display scale.

## Project lifecycle

4. Start with no project: chooser appears. Create blank project and new MNIST
   MLP under fresh parent. Verify folder names, `model.json`, dataset files,
   editable graph and project title.
5. Save, close, reopen through UI and startup path argument. Move node, rename
   node, edit parameter, connect/disconnect edge; save and compare reopened
   graph and manifest. Dirty state clears only after successful save.
6. Open missing/corrupt project, unsupported schema, undeclared package,
   duplicate resource identity, bad dependency and resource symlink/traversal.
   Existing active project must survive failure with visible diagnostic.
7. Force save failure using unwritable destination or removed parent. Existing
   file must remain valid; dirty state stays true; close requires save or
   explicit discard.

## Graph, resources and analysis

8. Palette lists exact active core/custom stereotypes. Add/select/move/delete
   nodes; handles connect only valid same-scope endpoints. Reject cycles and
   occupied target inputs. Verify viewport pan, zoom, fit and edge hit targets.
9. Inspector uses definition types, defaults, minimums and choices. Invalid
   value leaves graph unchanged. Dataset panel shows MNIST `image` input
   `[B,1,28,28]` float32 and `target` `[B]` int64. Package panel shows exact
   identities and dependency links from project scope.
10. MNIST path must infer `Flatten` to `[B,784]`, hidden layers to `[B,128]`
    and `[B,64]`, final logits to `[B,10]`; show typed result. Break a Linear
    dimension and verify node-specific semantic error; missing Input binding
    must be unresolved rather than success. Lua fault must be distinct from
    semantic error. Confirm no Python/backend/training process runs.

## Evidence and exit

Record commands, exit codes, screenshot path, observed defects and fixes in
`qa-results.md`. Rerun affected cases after each fix. Release passes only when
visible Wayland launch or documented equivalent desktop launch, project round
trip, MNIST analysis, all relevant automated tests and visual review pass.
