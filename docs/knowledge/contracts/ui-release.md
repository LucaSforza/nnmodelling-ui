# Native editor release contract

Status: native rewrite accepted from 2026-09-28 request. Accepted 2026-10-05:
[backend contract](backend.md) extends UI with HTTP jobs, loss monitoring,
snapshot restore and downloads; Python training runs in backend containers.

## Project lifecycle and workspace

On a normal launch without a project path, show only the startup chooser; keep
the editor window hidden until project creation or opening succeeds. Dismissing
the chooser exits the application. A canceled or failed create/open returns to
the chooser without exposing an empty editor. Startup first asks whether to
create or open a project. Creation then offers only the bundled MNIST MLP,
MNIST VAE and tiny LLM templates; the GUI never creates an empty project. The
creation form shows parent directory, model ID, display name, and editable
initial version defaulting to `0.1.0`, plus the exact destination
`<parent>/<model ID>`. It rejects an existing destination before copying.
Canceling the parent-directory picker returns to the same creation form and
retains all entered values.
Opening accepts a project directory containing `model.json`; an invalid
directory shows a specific explanation and leaves the current project intact.
`Save` writes atomically and clears dirty state only on success. `Close project`
returns to the chooser after saving; save failure keeps the project open with a
visible error. An explicit `Discard` path may close dirty project without
saving. Each template creates an editable copy with its own resources and graph;
the bundled source stays unchanged and no backend is contacted. Template copy
failure preserves the active project and removes only the new copy.

Accepted 2026-10-22: project chooser exposes Create/Open, and Create proceeds to
the three-template selection before opening the editor. Blank-project creation
is not a GUI action. Initial version is editable and starts at `0.1.0`; the
destination preview updates as parent or model ID changes.

Native path chooser may be an in-window path field. Users can also pass a
project directory as startup argument. Project folder, custom package and
dataset paths are validated as confined relative paths, with no symlink escape.
Schema v2 manifest is authoritative. Core stereotypes are immutable; custom
package/dataset lists contain exact ID, version and relative path. Loading is
staged and atomic. Project JSON is not the in-memory model.
Optional `manifest.activeDataset` stores exact selected `{id,version}`; Input
bindings use its named input slots.

## Graph editor

Main window has toolbar, package palette, contextual component guide, graph
viewport, node inspector, project resources panel and status/diagnostics.
The selected package guide and selected node Inspector show their definition
description; core Input, Output, Loss and Loss Output descriptions explain
dataset binding, model result and objective roles. User can create package-backed
nodes, select/delete/move them, pan/zoom/fit/arrange view, connect/disconnect
handles, edit names and typed parameters, save, close and reopen project.
Visible shortcuts are Ctrl++, Ctrl+-, Ctrl+F, arrow keys for viewport panning,
Ctrl+Shift+V for vertical Arrange and Ctrl+Shift+H for horizontal Arrange.
Right-clicking a node or edge opens actions for that item; edge context menu
supports mouse-only connection removal.
Edges render under nodes; input handle occupancy and cycle/scope validation
apply before commit. Accepted 2026-10-01: typed-outputs.md governs root's exactly
one Output/one Loss Output, spawned subflow boundaries and circle/edge visuals.
Multiple Input nodes are allowed with distinct named
bindings. Palette comes from active catalog, never package-ID switches.
Inspector controls follow stereotype parameter schema, defaults and choices.
Selected dataset exposes its named input/target tensor slots to client type
analysis. Dataset creation explains shape entry inline: comma-separated
dimensions, `B` for batch, with `B,4` and `B,1` examples. Package and dataset
panels show project-owned resources and exact identities/dependencies;
authoring/deletion require project transaction rules in
[project UML](../uml/project.md).

Graph view and inspector remain usable with unresolved tensors or Lua semantic
errors. Diagnostics identify node and category. Native client does not execute
`pytorch.py` or `dataset.py` or train models itself. Explicit backend connection
adds job monitoring, snapshot restore and wheel/weight downloads per backend.md.
MNIST MLP example is an editable classifier *design* for 28×28 grayscale
inputs and ten digit logits, not trained model or live classifier.

## Completion conditions

Accepted 2026-09-30 additions: New MNIST VAE, indented JSON, positioned card
parameters, visual stereotype/dataset authoring and local CLI service follow
resource-authoring.md and automation.md. These supersede earlier authoring and
command deferrals; training/backend remain separate.

1. Build and run Qt 6 client through a pure C application API and `justfile`.
2. Create, save, close and reopen project through UI without graph or resource
   loss; invalid open/save reports error and preserves active state.
3. Edit MNIST MLP example in UI: inspect package parameters, move a node,
   connect an edge, save, reopen and retain edits.
4. Load core package metadata and project-owned dataset/dependency metadata;
   reject invalid paths, identities and dependency closure.
5. Run package Lua inference for MLP path with selected MNIST dataset; show
    problems in diagnostics and successful shape/dtype in inspector, without backend.
6. Test model/project/graph invariants, Lua result classes, Qt interactions
    and real headless UI lifecycle. Review a screenshot from rendered client.

Accepted 2026-09-30: diagnostics.md replaces per-node success rows with Model
problems. Shape/dtype remain in the inspector; Lua compilation, model errors,
Incomplete and internal unavailability have distinct presentation and CLI parity.

These conditions define native editor milestone. Local backend milestone adds
acceptance conditions in backend.md; ordinary editing starts no backend job.
