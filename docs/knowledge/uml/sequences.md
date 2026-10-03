# Interaction sequences

## Purpose

Test ownership and ordering for important client flows. Dashed future actors
remain deferred. Each sequence references operation contracts in
[model](../contracts/model.md), [editor](../contracts/editor.md) and
[future interfaces](../contracts/future-interfaces.md).

## Startup and graphical synchronization

```mermaid
sequenceDiagram
    participant Qt as QApplication / MainWindow
    participant App as C NNApplication
    participant Project as C Project
    participant Scene as QGraphicsScene
    Qt->>App: new(core_root)
    Qt->>App: open(path) optional
    App->>Project: stage and validate
    Project-->>App: project or error
    Qt->>Scene: synchronize borrowed C snapshots
    Note over Scene: Retain IDs and graphical state only
    Scene->>App: mutation on completed gesture
    App-->>Scene: result / error
    Scene->>Scene: refresh from C state
```

## Create node

```mermaid
sequenceDiagram
    actor User
    participant Palette
    participant App as Application
    participant Catalog
    participant Model
    participant Types as Lua type analysis
    User->>Palette: choose exact package and parameters
    Palette->>App: addNode(ref, values, position)
    App->>Catalog: resolve definition + defaults
    App->>Model: validate and insert node
    opt kind=subflow
      App->>Model: insert mapped terminal per declared output atomically
    end
    Model-->>App: NodeId or error
    App->>Types: invalidate dependent region
    App-->>Palette: result + diagnostics
```

## Connect nodes

```mermaid
sequenceDiagram
    actor User
    participant Editor
    participant App as Application
    participant Model
    participant Types as Lua type analysis
    Editor-->>User: Visible contrasting handles aligned to flow direction, hover cue
    User->>Editor: drag source output to target input
    Note over User,Editor: Visible PortItem and generous hit region identify the same handle
    Editor->>App: connect(source, sourceHandle, target, targetHandle)
    App->>Catalog: resolve output type and terminal-only restriction
    App->>Model: check same scope, handle directions, occupancy, cycle
    alt valid
      Model->>Model: insert edge ordered by target handle
      App->>Types: invalidate descendants and infer
      Types-->>App: result or diagnostics
      App-->>Editor: EdgeId + diagnostics
    else invalid
      Model-->>App: structural error, no mutation
      App-->>Editor: error
    end
```

## Fit and arrange scope

```mermaid
sequenceDiagram
    actor User
    participant Window as MainWindow
    participant Scene as GraphScene
    participant App as C NNApplication
    participant View as GraphView
    User->>Window: Fit or Arrange(Vertical/Horizontal)
    Note over Window: Fit always chooses Vertical
    Window->>Scene: set presentation direction
    Window->>App: read current-scope nodes and edges
    Window->>Window: deterministic DAG ranks and spaced lanes
    Window->>App: move nodes using copied stable IDs
    App-->>Window: committed positions or visible error
    Window->>Scene: refresh matching handles and edge tangents
    Window->>View: frame content with margin and final centering
    Note over App,Scene: Positions persist; direction is session presentation state
```

## Edit parameter

```mermaid
sequenceDiagram
    actor User
    participant Inspector
    participant App as Application
    participant Catalog
    participant Model
    participant Types as Lua type analysis
    User->>Inspector: edit field
    Inspector->>App: setParameter(node,key,value)
    App->>Catalog: get definition and validate type/range
    App->>Model: commit one value
    App->>Types: invalidate dependent region
    App-->>Inspector: result + diagnostics
```

## Project loading

```mermaid
sequenceDiagram
    actor User
    participant App as Application
    participant Project
    participant Catalog
    participant Lua
    participant Model
    User->>App: open(path)
    App->>Project: parse typed model + manifest
    Project->>Catalog: stage core + exact custom closure
    Catalog->>Lua: prepare isolated rules
    Catalog->>Model: validate graph against staged scope
    alt all valid
      App->>App: atomically swap model and catalog
      App-->>User: project active
    else error
      App->>App: discard staged state
      App-->>User: prior project unchanged + diagnostic
    end
```

## Create, save and close project

```mermaid
sequenceDiagram
    actor User
    participant Editor
    participant App as Application
    participant Project
    participant Catalog
    User->>Editor: New MNIST MLP(parent, id, name)
    Editor->>App: createProject(...)
    App->>Project: stage owned folder and schema-v2 model
    Project->>Catalog: validate core and copied dataset reference
    Catalog-->>Project: exact active scope
    Project-->>App: committed project
    App-->>Editor: editable graph and resource panels
    User->>Editor: Save
    Editor->>App: save()
    App->>Project: write temporary model.json, flush, rename
    Project-->>App: success or error
    App-->>Editor: clear dirty only on success
    User->>Editor: Close
    Editor->>App: close(save or explicit discard)
    App->>Project: release graph, datasets and package scope
    App-->>Editor: project chooser
```

## Resolve project resource dependencies

```mermaid
sequenceDiagram
    participant Project
    participant Manifest
    participant Packages as PackageCatalog
    participant Datasets as DatasetCatalog
    participant Graph
    Project->>Manifest: parse customPackages and customDatasets
    Manifest->>Packages: load exact references under project directory
    Packages->>Packages: validate identities and dependency DAG
    Manifest->>Datasets: load exact references under project directory
    Datasets->>Datasets: validate named input and target slots
    Packages-->>Project: staged active package set
    Datasets-->>Project: staged active dataset set
    Project->>Graph: validate node packages and Input bindings
    Note over Project,Graph: Any failure discards entire staged project
```

## Local command, introspection and screenshot

Accepted 2026-09-30 per automation.md; transport is nonblocking AF_UNIX and
dispatch occurs on Qt application thread. Resource creation sequence:

```mermaid
sequenceDiagram
    actor Author as User or LLM
    participant Entry as Qt form or nnmodelctl
    participant App as C application
    participant Project as C project
    participant Catalog
    Author->>Entry: create stereotype or dataset
    Entry->>App: typed identity + definition payload
    App->>App: validate definition and bounded Lua initialization
    App->>Project: stage generated resource and candidate manifest
    Project->>Catalog: validate identities, schema and dependencies
    alt valid
        Project->>Project: atomic candidate model.json save
        Project-->>App: publish owned resources
        App-->>Entry: success and refreshed snapshot
    else invalid or write failure
        Project->>Project: remove transaction-created files only
        App-->>Entry: error, prior graph/resources/dirty preserved
    end
```

```mermaid
sequenceDiagram
    actor Agent
    participant CLI as nnmodelctl
    participant IPC as Local IPC
    participant Adapter as Command adapter
    participant App as Application
    participant Editor
    participant Graphics
    Agent->>CLI: operation or inspect UI ID
    CLI->>IPC: typed request
    IPC->>Adapter: dispatch
    alt model operation
      Adapter->>App: same operation as UI
      App-->>Adapter: result + diagnostics
    else introspection
      Adapter->>Editor: snapshot semantic UI tree
      Editor-->>Adapter: tree
    else screenshot
      Adapter->>Editor: arrange if requested, await layout/frame
      Editor->>Graphics: capture completed frame
      Graphics-->>Adapter: image or error
    end
    Adapter-->>CLI: response
    CLI-->>Agent: result
```

## Constraints and C mapping

Formal: `operationFailure ⇒ graph'=graph`; `projectOpenFailure ⇒
(graph',catalog')=(graph,catalog)`; `renderFrame ⇒ graph'=graph`.
Natural: failed commands/load and frame rendering do not corrupt active model.

Formal: `∀join: inputOrder=targetHandleOrder`; `screenshotFrame≥layoutFrame`.
Natural: semantic join order and visible screenshot order are deterministic.

In C, sequences call explicit application functions; event handlers and local
transport adapters hold pointers/IDs to the same application owner. Project
staging uses temporary owned structs, swapped only after validation. No second
graph is allocated for command service beyond immutable response snapshots.
Analysis queries/invalidation and problem navigation follow diagnostics.md.

## Analysis cache and problem navigation

Accepted 2026-09-30, consolidated 2026-10-02 from diagnostics UML; owner diagram
is editor.md and result types are metamodel.md. contracts/diagnostics.md remains
normative. Typed spawning is already included in Create node above.

```mermaid
sequenceDiagram
    actor User
    participant Entry as Qt or CLI
    participant App as C NNApplication
    participant Analysis as C Lua analysis
    participant Scene as Qt GraphScene
    Entry->>App: semantic mutation
    App->>App: invalidate report on success
    Entry->>App: analysis query
    alt cache missing
        App->>Analysis: infer immutable project
        Analysis-->>App: owned report or failure
    end
    App-->>Entry: borrowed outcomes or explicit failure
    Note over Entry: Problems grouped by root cause; tensors separate
    User->>Entry: click problem or ui.reveal(node)
    Entry->>Scene: validate ID, switch scope, queued refresh
    Scene->>Scene: select and center node
```
