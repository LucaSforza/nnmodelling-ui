# Graph editing sequences

## Add a node

**Purpose.** Show how a palette choice becomes a C-owned graph node with
catalog defaults and, for subflows, transactional boundary nodes.

```mermaid
sequenceDiagram
    actor User
    participant Palette as Package palette Qt
    participant Main as MainWindow Qt Widgets
    participant App as NNApplication C API
    participant Catalog as Active package catalog
    participant Model as C graph model
    User->>Palette: select package and version
    User->>Main: add package to current scope
    Main->>App: nn_app_add_node(id, package, version, scope, position)
    App->>Catalog: resolve active package and parameter defaults
    App->>Model: add node and validated default values
    opt package kind is subflow
        App->>Catalog: resolve Input and typed boundary packages
        App->>Model: add inherited Input and one terminal per output
        Note over App,Model: Any failure rolls back the whole node creation
    end
    Model-->>App: committed node ID or error
    App->>App: record history and invalidate analysis on success
    App-->>Main: result or error
    Main->>Main: refresh panels after successful change
```

## Connect nodes

**Purpose.** Trace a visible port gesture through C validation, and show that
inference refresh happens after a successful mutation.

```mermaid
sequenceDiagram
    actor User
    participant Source as Source NodeItem Qt
    participant Scene as GraphScene Qt
    participant Main as MainWindow Qt Widgets
    participant Target as Target NodeItem Qt
    participant App as NNApplication C API
    participant Catalog as Active package catalog
    participant Model as C graph model
    User->>Source: press visible output handle
    Source->>Scene: begin connection draft for handle ID
    loop while pointer moves
        User->>Scene: move pointer
        Scene->>Target: update draft and target-port cue
    end
    User->>Target: release on visible input handle
    Scene->>App: nn_app_connect(source, sourceHandle, target, targetHandle)
    App->>Catalog: resolve package port kinds and output/loss compatibility
    App->>Model: validate scope, direction, occupancy, and cycle
    alt valid connection
        Model->>Model: insert edge in target-handle order
        Model-->>App: committed edge
        App->>App: record history and invalidate analysis
        App-->>Scene: success
        Scene->>Scene: refresh routes and emit modelChanged
        Scene-->>Main: modelChanged
        Main->>Scene: queued refresh and lazy analysis query
    else invalid connection
        App-->>Scene: error, committed graph unchanged
        Scene-->>Main: report operation error
        Main-->>User: show operation error
    end
```

## Edit a parameter

**Purpose.** Show that the inspector sends text to the shared C validator and
that only a successful value update invalidates graph analysis.

```mermaid
sequenceDiagram
    actor User
    participant Inspector as MainWindow inspector Qt
    participant App as NNApplication C API
    participant Catalog as Active package catalog
    participant Model as C graph model
    User->>Inspector: edit parameter value
    Inspector->>App: nn_app_set_parameter_text(node, key, text)
    App->>Catalog: resolve parameter schema and parse typed value
    App->>Model: validate and set one parameter
    alt value accepted and changed
        Model-->>App: success
        App->>App: record history and invalidate analysis
        App-->>Inspector: success
        Inspector->>Inspector: refresh inspector and diagnostics
    else invalid, rejected, or unchanged
        App-->>Inspector: error or unchanged result
        Note over App,Model: Rejected input does not alter the graph
    end
```

## Move nodes

**Purpose.** Distinguish Qt drag preview from the grouped C mutation committed
on release.

```mermaid
sequenceDiagram
    actor User
    participant Item as NodeItem Qt
    participant Scene as GraphScene Qt
    participant App as NNApplication C API
    participant Model as C graph model
    User->>Item: drag one or more selected nodes
    loop while pointer moves
        Item->>Item: preview position snapped to 20-unit grid
        Item->>Scene: update connected routes from preview geometry
    end
    User->>Scene: release drag
    Scene->>App: nn_app_begin_edit()
    loop each moved node
        Scene->>App: nn_app_move_node(id, committed position)
        App->>Model: normalize and validate integer grid coordinates
    end
    Scene->>App: nn_app_end_edit(commit=allMovesSucceeded)
    App-->>Scene: one history revision, or full rollback and error
    Scene->>Scene: refresh committed positions and routes
```

## Adjust visible join slots

**Purpose.** Separate Qt’s temporary empty-slot controls from model-backed
occupied handles and actual graph connections.

```mermaid
sequenceDiagram
    actor User
    participant Join as Join NodeItem Qt
    participant Scene as GraphScene Qt
    participant App as NNApplication C API
    participant Model as C graph model
    Scene->>App: read defined and occupied input handles
    App->>Model: inspect graph ports and edges
    Model-->>App: copied port IDs and occupancy
    App-->>Scene: port snapshot
    Scene->>Scene: combine occupied ports with retained free-slot count
    Note over Scene: At least two free slots, occupied ports remain visible
    User->>Join: click plus or minus
    Join->>Scene: change presentation-only free-slot count
    Scene->>Scene: queue refresh and redraw ordered handles
    Note over Scene,Model: Slot controls create no edge, connection uses nn_app_connect
```

## Arrange current scope

**Purpose.** Show that Arrange computes positions in Qt but commits them as one
C-owned edit, with rollback if a move fails.

```mermaid
sequenceDiagram
    participant Main as MainWindow Qt Widgets
    participant Scene as GraphScene Qt
    participant Layout as DataflowLayout Qt
    participant App as NNApplication C API
    participant Model as C graph model
    Main->>Scene: refresh current geometry and routes
    Main->>App: read current-scope model and package snapshots
    App-->>Main: borrowed node, edge, and package data
    Main->>Layout: compute DAG ranks, port lanes, and grid positions
    Layout-->>Main: proposed positions
    Main->>App: begin grouped edit
    loop each proposed node position
        Main->>App: nn_app_move_node(id, position)
        App->>Model: validate and commit candidate coordinate
    end
    Main->>App: nn_app_end_edit(commit=allMovesSucceeded)
    App-->>Main: one committed history edit or rollback error
    Main->>Main: refreshAll and fitGraph after committed layout
    Note over Scene,Model: Arrange persists positions, any failed move rolls back group
```

## Fit current scope

**Purpose.** Show that Fit changes only the 2D view camera and does not move
nodes or write model state.

```mermaid
sequenceDiagram
    actor User
    participant View as GraphView QGraphicsView
    participant Scene as GraphScene Qt
    User->>View: choose Fit
    View->>Scene: recompute routes from current geometry
    View->>View: frame scene content with margin
    Note over View,Scene: Camera adjustment only, persisted node coordinates stay unchanged
```

## Single-command history

**Purpose.** Show how a single graph command records history before mutation,
and how C restores snapshots while Qt refreshes its views.

```mermaid
sequenceDiagram
    actor User
    participant Actions as MainWindow graph action handlers
    participant App as NNApplication C API
    participant History as C edit history
    participant Model as C graph model
    Actions->>App: validated palette or inspector command
    App->>History: snapshot before changed command
    App->>Model: apply mutation
    Model-->>App: success or error
    App->>History: retain one revision only when model changed
    Note over App,History: Multi-node drag, Delete, and Arrange use grouped edits, see next sequence
```

## Grouped edit history

**Purpose.** Show how multi-command canvas actions commit one history revision
and roll back together if any command fails.

```mermaid
sequenceDiagram
    participant Scene as GraphScene or MainWindow Qt
    participant App as NNApplication C API
    participant History as C edit history
    participant Model as C graph model
    Scene->>App: nn_app_begin_edit()
    loop each command in the grouped drag, Delete, or Arrange action
        Scene->>App: validated graph command or move
        App->>Model: mutate active graph
    end
    Scene->>App: nn_app_end_edit(commit=allCommandsSucceeded)
    App->>History: store one snapshot on commit, restore start on failure
```

## Undo and redo

**Purpose.** Show how restoring a C-owned snapshot updates model revision and
analysis state before Qt refreshes its views.

```mermaid
sequenceDiagram
    actor User
    participant Main as MainWindow Qt Widgets
    participant App as NNApplication C API
    participant History as C edit history
    participant Model as C graph model
    User->>Main: choose Undo or Redo
    Main->>App: nn_app_undo() or nn_app_redo()
    App->>History: retrieve owned snapshot and update revision
    App->>Model: swap active graph with snapshot
    App->>App: update dirty revision and invalidate analysis
    Main->>Main: refresh graph, scope tree, inspector, and diagnostics
```
