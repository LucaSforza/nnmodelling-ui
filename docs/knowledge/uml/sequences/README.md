# Sequence catalog

These diagrams describe observed ordering and ownership across the current
client, local automation, backend and Python runtime. The sequence tool checks
all Markdown Mermaid fences under `docs/knowledge/`; use `--write` only when the
whitespace-only normalization is intended.

On a clean checkout, run `just mermaid-sync` once to install the pinned Node
dependencies. `just mermaid-check` validates all KB fences, `just mermaid-write`
normalizes their whitespace and validates them, and `just test-mermaid-check`
checks formatter idempotence and malformed-diagram reporting. The parser does
not infer missing software behavior or automatically rewrite diagram semantics.

| File | Flow boundary | Main source owners |
| --- | --- | --- |
| [Project and editor](project-editor.md) | Open/save, graph edits, layout and history | `src/application/`, `src/project/`, `src/model/`, `src/gui/qt/` |
| [Analysis and navigation](analysis-navigation.md) | Lazy report, diagnostics, preview and hover | `src/application/`, `src/inference/`, `src/gui/qt/` |
| [Resources and local automation](resource-automation.md) | Resource transaction, AF_UNIX dispatch, UI introspection and capture | `src/project/`, `src/catalog/`, `src/automation/`, `src/nnmodelctl/` |
| [Backend jobs and dashboard](backend-training.md) | Snapshot, queue, container, metrics, artifacts and restore | `backend/`, `src/gui/qt/BackendDialog.cpp` |
| [Python runtime](python-runtime.md) | Dataset boundary, graph construction/execution and wheel inference | `python/nnmodelling-runtime/`, `backend/worker.py` |
| [Network 3D](visualization-3d.md) | Disposable occurrence scene, camera and picking | `src/visualization/`, `src/gui/qt/` |
| [Development agents](development-agents.md) | Codex and OpenCode delegation | Codex collaboration API / OpenCode V2 mailbox |
