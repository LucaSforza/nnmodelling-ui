# Interaction sequences

These diagrams describe how the implemented native client components cooperate
for common user and automation flows. Read the short purpose note above each
diagram, then follow its concrete Qt widget, C API, model, catalog, or runtime
lifelines.

The diagrams record current implementation behavior. Normative rules remain in
the linked contracts; an implementation/design mismatch must be reported and
resolved there rather than hidden by a diagram.

| Area | Sequences |
| --- | --- |
| Project lifecycle | [Startup and editor synchronization](project-lifecycle.md#startup-and-editor-synchronization), [Open and replace a project](project-lifecycle.md#open-and-replace-a-project), [Create a project](project-lifecycle.md#create-a-project), [Save and close](project-lifecycle.md#save-and-close) |
| Graph editing | [Add a node](graph-editing.md#add-a-node), [Connect nodes](graph-editing.md#connect-nodes), [Edit a parameter](graph-editing.md#edit-a-parameter), [Move nodes](graph-editing.md#move-nodes), [Adjust visible join slots](graph-editing.md#adjust-visible-join-slots), [Arrange current scope](graph-editing.md#arrange-current-scope), [Fit current scope](graph-editing.md#fit-current-scope), [Single-command history](graph-editing.md#single-command-history), [Grouped edit history](graph-editing.md#grouped-edit-history), [Undo and redo](graph-editing.md#undo-and-redo) |
| Analysis and navigation | [Analysis and problem navigation](analysis-navigation.md#analysis-and-problem-navigation), [Preview and hover](analysis-navigation.md#preview-and-hover) |
| Resources | [Create a stereotype or dataset](resource-authoring.md#create-a-stereotype-or-dataset) |
| Local command service | [Request transport](command-service.md#request-transport), [Route a request](command-service.md#route-a-request) |
| 3D explorer | [Build the 3D view](network-3d.md#build-the-3d-view), [Navigate the 3D camera](network-3d.md#navigate-the-3d-camera), [Select a 3D occurrence](network-3d.md#select-a-3d-occurrence) |

## Shared invariants

Graph mutations go through the C NNApplication that owns the active project.
Qt retains presentation state and copied identifiers, not a second graph.
Failed mutations leave the committed graph unchanged; failed project opens
preserve the active project. Rendering and 3D exploration are read-only.

Connection ordering follows target-handle order. A requested automation
screenshot is captured after requested arrangement, widget refresh, and layout
have completed.

## Contracts

Project and graph behavior follows [model](../../contracts/model.md),
[editor](../../contracts/editor.md), and [typed outputs](../../contracts/typed-outputs.md).
Analysis follows [diagnostics](../../contracts/diagnostics.md). Local commands
follow [automation](../../contracts/automation.md). The 3D explorer follows
[visualization-3d](../../contracts/visualization-3d.md).
