# Interaction sequences

Sequence diagrams live in the dedicated [sequence catalog](sequences/README.md).
Each file covers one subsystem boundary and names the implementation entrypoints
that own its work. Structural ownership remains in [editor](editor.md),
[project](project.md), [metamodel](metamodel.md), [automation](automation.md),
[backend](backend.md) and [agent-swarm](agent-swarm.md).

## Sequence catalog

- [Project and editor operations](sequences/project-editor.md): startup, project
  lifecycle, graph edits, layout, join slots and undo.
- [Analysis cache and problem navigation](sequences/analysis-navigation.md):
  lazy C analysis, copied diagnostic identity, scope navigation, hover and preview.
- [Resource authoring and local automation](sequences/resource-automation.md):
  project-owned datasets/stereotypes, CLI dispatch, UI inspection and capture.
- [Backend jobs and training dashboard](sequences/backend-training.md):
  immutable submission, queue/container execution, curves, cancellation, export
  and restore.
- [Python dataset and numerical graph runtime](sequences/python-runtime.md):
  adapter boundaries, generic DAG construction, training and standalone inference.
- [Network 3D explorer](sequences/visualization-3d.md): read-only scene creation,
  camera, picking and teardown.
- [Development agents](sequences/development-agents.md): Codex delegation and
  OpenCode V2 mailbox flows.

## Analysis cache and problem navigation

This anchor is retained for existing links. The flow is documented in
[analysis-navigation.md](sequences/analysis-navigation.md#lazy-analysis-and-problem-navigation).
