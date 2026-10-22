# Native client QA

Status: release gate. Run after implementation, record result and evidence in
`docs/knowledge/testing/qa-results.md`. A build or dummy-driver smoke test does
not establish that visible application works.

## Environment and launch

1. Run `just test`, `just test-ui`, `CCACHE_DISABLE=1 just build` and
   `git diff --check`; record compiler, Qt version and failures.
2. For visible UI trials, follow `.agents/skills/nnmodelling-browser/SKILL.md`:
   launch native Qt client through its noVNC browser bridge and interact with
   the visible app. CLI or offscreen checks do not replace this trial. After
   every code/build change, rebuild and restart client and bridge before more
   GUI checks.
3. Capture screenshot after a full frame. Inspect text legibility, clipping,
   node/edge placement, panel bounds, contrast and glyphs at 1280x800 and a
   smaller window; note display scale and screenshot path.

## Project lifecycle

4. Start with no project: choose Create or Open. Create offers MNIST MLP,
   MNIST VAE and tiny LLM only. Check initial version `0.1.0` and live
   destination `<parent>/<model ID>`; create each template under a fresh
   parent, then verify folder names, `model.json`, project-owned resources,
   editable graph and project title. Open a non-project directory and confirm
   a specific explanation while the current project remains intact.
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

8. Palette lists exact active core/custom stereotypes and component guide.
   Verify descriptions for Input, Output, Loss and Loss Output. Add/select/
   move/delete nodes; handles connect only valid same-scope endpoints. Reject
   cycles and occupied target inputs. Verify visible Zoom In/Out/Fit, arrow-key
   pan, vertical/horizontal Arrange shortcuts and mouse pan/zoom/fit. Right-click
   edge and remove it from context menu. Enter a subflow using its visible
   button; confirm editor shows only that scope, with no child preview.
9. Inspector uses definition types, defaults, minimums and choices. Invalid
   value leaves graph unchanged. Dataset authoring explains `B` inline with
   examples `B,4` and `B,1`. Dataset panel shows MNIST `image` input
   `[B,1,28,28]` float32 and `target` `[B]` int64. Package panel shows exact
   identities and dependency links from project scope.
10. MNIST path must infer `Flatten` to `[B,784]`, hidden layers to `[B,128]`
    and `[B,64]`, final logits to `[B,10]`; show typed result. Break a Linear
    dimension and verify node-specific semantic error; missing Input binding
    must be unresolved rather than success. Lua fault must be distinct from
    semantic error. Ordinary graph editing must not launch Python/training.
    Explicit backend job submission follows backend.md and its separate gates.

## Evidence and exit

Record commands, exit codes, screenshot path, observed defects and fixes in
`qa-results.md`. Rerun affected cases after each fix. Release passes only when
visible Wayland launch or documented equivalent desktop launch, project round
trip, MNIST analysis, all relevant automated tests and visual review pass.
