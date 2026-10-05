# Project and editor sequences

These flows follow `src/application/application.c`, `src/application/graph_commands.c`,
`src/application/history.c`, `src/project/project.c`, `src/model/model.c` and the
Qt entrypoints in `src/gui/qt/MainWindow*.cpp`, `GraphScene.cpp` and
`DataflowLayout.cpp`. C owns the active model and persisted integer positions;
Qt owns selection, scope, camera, drafts and presentation-only expansion state.

## Startup and project replacement

```mermaid
  sequenceDiagram
    actor User
    participant Qt as QApplication/MainWindow
    participant App as NNApplication
    participant Project as NNProject
    participant Catalog as Package and dataset catalogs
    participant Scene as GraphScene
    Qt->>App: create(core root)
    opt Open a project
      User->>Qt: choose project path
      Qt->>App: open(path)
      App->>Project: parse model.json and manifest
      Project->>Catalog: stage exact resources and dependencies
      Catalog->>Project: resolved catalog or error
      Project->>Project: validate graph and bindings
      alt staged project is valid
        Project->>App: complete candidate project
        App->>App: swap active project and invalidate prior report
        App-->>Qt: project active
        Qt->>Scene: reset scope and refresh items
        Scene->>App: query borrowed model/catalog snapshots
        App-->>Scene: authoritative state
        Scene->>Scene: copy IDs and synchronize items
      else parse, resolution or validation fails
        Project-->>App: error
        App-->>Qt: error and prior project remains active
      end
    end
```

An open failure discards staged state and preserves the active project. Graph
snapshots borrowed from C expire on the next mutation; Qt retains copied stable
IDs. Relevant owners are `nn_app_open` and project staging in
`src/application/application.c` and `src/project/project.c`.

## Create, save and close

```mermaid
  sequenceDiagram
    actor User
    participant Qt as MainWindow
    participant App as NNApplication
    participant Project as NNProject
    participant Catalog as Resource catalogs
    User->>Qt: create blank/template project
    Qt->>App: create(parent, id, name, template)
    App->>Project: stage directory and schema-v2 model
    Project->>Catalog: resolve copied dataset and core packages
    Catalog-->>Project: exact active scope or error
    alt valid candidate
      Project->>Project: publish staged model
      Project-->>App: committed project
      App-->>Qt: refresh root scope and resources
    else failure
      Project-->>App: error and candidate removed
      App-->>Qt: prior project remains active
    end
    User->>Qt: Save
    Qt->>App: save()
    App->>Project: write temporary sibling then atomic rename
    Project-->>App: success or error
    App-->>Qt: clear dirty only on success
    User->>Qt: Close
    Qt->>App: close(save or explicit discard)
    App->>Project: release model, catalogs and directory
    App-->>Qt: project chooser
```

Save failure retains both the old file and dirty state. Successful project
replacement or close releases the previous catalog and resets scope to root.

## Add and connect nodes

```mermaid
  sequenceDiagram
    actor User
    participant Qt as Palette/GraphScene
    participant App as NNApplication
    participant Catalog as Package catalog
    participant Model as NNModel
    participant Analysis as Lazy inference report
    User->>Qt: choose package and values
    Qt->>App: add node in current scope
    App->>Catalog: resolve exact package and defaults
    Catalog-->>App: definition and validated parameters
    App->>Model: add node
    opt New node kind is subflow
      App->>Model: add one inherited Input and mapped terminals atomically
    end
    Model-->>App: node ID or error
    alt mutation succeeded
      App->>App: invalidate lazy analysis report
      App-->>Qt: successful command result
      Qt->>Analysis: query diagnostics through lazy application report
    else mutation rejected
      App-->>Qt: error with no graph change
    end
    User->>Qt: drag source handle to target handle
    Qt->>App: connect(source, sourceHandle, target, targetHandle)
    App->>Catalog: resolve handle direction and output type
    App->>Model: validate scope, occupancy and cycle
    alt valid
      Model->>Model: insert one edge ordered by target handle
      App->>App: invalidate lazy analysis report
      App-->>Qt: edge ID
      Qt->>Qt: refresh items and routes, query lazy diagnostics
    else invalid
      Model-->>App: structural error without mutation
      App-->>Qt: visible error dialog
    end
```

Qt submits graph commands but does not decide tensor compatibility. Add/connect
and spawned terminal operations are owned by `src/application/graph_commands.c`
and model operations in `src/model/`; report refresh remains lazy.

## Parameter edit, layout and join slots

```mermaid
  sequenceDiagram
    actor User
    participant Qt as Inspector/GraphScene
    participant App as NNApplication
    participant Catalog as Package catalog
    participant Model as NNModel
    User->>Qt: edit schema field
    Qt->>App: setParameter(node, key, value)
    App->>Catalog: validate declared type, range and choices
    App->>Model: commit one typed value
    alt accepted
      App->>App: invalidate lazy analysis report
      App-->>Qt: refresh inspector and problems
    else rejected
      App-->>Qt: reason and committed value is retained
    end
    User->>Qt: Arrange vertical or horizontal
    Qt->>Qt: compute deterministic DAG ranks and lanes
    Qt->>App: move nodes with integer grid positions
    App->>Model: validate and persist scope-local positions
    App-->>Qt: result, then refresh routes and frame content
    User->>Qt: Fit
    Qt->>Qt: recompute orthogonal routes and frame content
    Note over Qt: Fit changes camera only and Arrange changes positions
    User->>Qt: add/remove visible join slot
    Qt->>App: query occupied numeric in-N handles
    App-->>Qt: occupied handle IDs
    Qt->>Qt: retain occupied slots and edit free-slot presentation state
```

Position/label changes do not invalidate semantic analysis. Coordinates persist
as signed 32-bit multiples of 20; direction and visible free join slots remain
Qt session state. Join kind, not package identity, selects the junction UI.

## Drag and persist node positions

```mermaid
  sequenceDiagram
    actor User
    participant Item as NodeItem
    participant Scene as GraphScene
    participant App as NNApplication
    participant Model as NNModel
    User->>Item: drag one or more nodes
    Item->>Item: preview positions snapped to the 20-unit grid
    Item->>Scene: update connected edge paths for preview geometry
    User->>Scene: release drag
    Scene->>App: move copied stable IDs and integer coordinates
    App->>Model: normalize and range-check position values
    alt every position is valid
      Model-->>App: persist scope-local coordinates
      App-->>Scene: committed positions
      Scene->>Scene: refresh obstacles and orthogonal routes
    else any position is invalid
      Model-->>App: error and no graph mutation
      App-->>Scene: restore committed positions
    end
```

Drag preview is Qt-only. Release is the model mutation boundary; failure restores
the model-backed item positions and does not dirty the graph.

## Undo and grouped edits

```mermaid
  sequenceDiagram
    actor User
    participant Qt as GraphScene
    participant App as NNApplication
    participant History as NNEditHistory
    participant Model as NNModel
    User->>Qt: multi-node drag, Delete or Arrange
    Qt->>App: beginEdit()
    App->>History: retain pre-edit model snapshot
    loop Commands in gesture
      Qt->>App: validated model operation
      App->>Model: mutate active graph
    end
    Qt->>App: endEdit(commit)
    alt all commands succeed
      App->>History: publish one revision
      App->>App: invalidate analysis if semantics changed
    else one command fails
      App->>History: restore pre-edit graph
    end
    User->>Qt: Undo or Redo
    Qt->>App: undo() or redo()
    App->>History: swap active model snapshot
    App->>App: update dirty revision and invalidate report
    App-->>Qt: refresh items, scopes, inspector and diagnostics
```

`NNApplication` owns the bounded history. Qt holds no parallel graph history;
camera, selection, expansion and direction are not restored by undo.
