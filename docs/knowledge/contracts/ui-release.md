# Native editor release contract

Status: accepted from 2026-09-28 request. Complete UI/client rewrite target;
training/backend execution remains external future work.

## Project lifecycle and workspace

Startup displays project chooser. `New project` asks parent directory, valid
model ID, version and display name; creates a new child directory named by ID
and `model.json`, rejecting existing child. `Open project` accepts a project
directory containing `model.json`; invalid project leaves current session
unchanged. `Save` writes atomically and clears dirty state only on success.
`Close project` returns to chooser after saving; save failure keeps project
open with visible error. An explicit `Discard` path may close dirty project
without saving. `New MNIST MLP` creates a copied editable project from bundled
template with its own dataset metadata and model graph. No backend is contacted.

Native path chooser may be an in-window path field. Users can also pass a
project directory as startup argument. Project folder, custom package and
dataset paths are validated as confined relative paths, with no symlink escape.
Schema v2 manifest is authoritative. Core stereotypes are immutable; custom
package/dataset lists contain exact ID, version and relative path. Loading is
staged and atomic. Project JSON is not the in-memory model.
Optional `manifest.activeDataset` stores exact selected `{id,version}`; Input
bindings use its named input slots.

## Graph editor

Main window has toolbar, package palette, graph viewport, node inspector,
project resources panel and status/diagnostics. User can create package-backed
nodes, select/delete/move them, pan/zoom/fit/arrange view, connect/disconnect
handles, edit names and typed parameters, save, close and reopen project.
Edges render under nodes; input handle occupancy and cycle/scope validation
apply before commit. Multiple Input nodes are allowed with distinct named
bindings. Palette comes from active catalog, never package-ID switches.
Inspector controls follow stereotype parameter schema, defaults and choices.
Selected dataset exposes its named input/target tensor slots to client type
analysis. Package and dataset panels show project-owned resources and exact
identities/dependencies; authoring/deletion require project transaction rules
in [project UML](../uml/project.md).

Graph view and inspector remain usable with unresolved tensors or Lua semantic
errors. Diagnostics identify node and category. Native client does not execute
`pytorch.py` or `dataset.py`, train model, download weights, or predict digits.
MNIST MLP example is an editable classifier *design* for 28×28 grayscale
inputs and ten digit logits, not trained model or live classifier.

## Completion conditions

1. Build and run Qt 6 client through a pure C application API and `justfile`.
2. Create, save, close and reopen project through UI without graph or resource
   loss; invalid open/save reports error and preserves active state.
3. Edit MNIST MLP example in UI: inspect package parameters, move a node,
   connect an edge, save, reopen and retain edits.
4. Load core package metadata and project-owned dataset/dependency metadata;
   reject invalid paths, identities and dependency closure.
5. Run package Lua inference for MLP path with selected MNIST dataset; show
   shape/dtype diagnostics without backend.
6. Test model/project/graph invariants, Lua result classes, Qt interactions
   and real headless UI lifecycle. Review a screenshot from rendered client.

These conditions define current release stop point. Additional backend,
training and command automation remain separate future milestones.
