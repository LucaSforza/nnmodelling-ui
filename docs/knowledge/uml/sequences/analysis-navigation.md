# Analysis and problem navigation

The C `NNApplication` owns one lazy `NNInferenceReport`. Analysis is computed
only when queried; a successful semantic mutation invalidates the cache but does
not infer immediately. Position, label and save operations keep the report.
Sources: `src/application/application.c`, `src/application/graph_commands.c`,
`src/inference/inference.c`, `src/gui/qt/MainWindowDiagnostics.cpp` and
`src/gui/qt/GraphScene.cpp`.

## Lazy analysis and problem navigation

```mermaid
  sequenceDiagram
    actor User
    participant Entry as Qt or nnmodelctl
    participant App as NNApplication
    participant Analysis as C Lua inference
    participant Scene as GraphScene
    User->>Entry: request problems or analysis.diagnostics
    Entry->>App: nn_app_analysis(app)
    alt report cache is valid
      App-->>Entry: borrowed cached report
    else cache is absent or invalidated
      App->>Analysis: infer active immutable model
      alt complete analysis succeeds
        Analysis-->>App: owned node results, tensors and root status
        App->>App: cache report
        App-->>Entry: borrowed report
      else analysis construction or execution fails
        Analysis-->>App: explicit failure and error
        App-->>Entry: unavailable result and no empty-success report
      end
    end
    Entry->>Entry: group problem rows by stable root cause
    User->>Entry: select problem or call ui.reveal(node ID)
    Entry->>App: validate stable node ID
    App-->>Entry: accepted or unknown ID
    Entry->>Scene: switch scope, refresh, select and center node
```

Problems distinguish Lua compilation, model semantic, incomplete and internal
failures. Successful typed tensors stay available in the inspector and port
hover; they do not become success rows. Borrowed report references expire on
invalidation or application release. Qt copies stable IDs and category markers;
it never keeps report pointers across mutation. A failed mutation preserves
the old report because active model state did not change.

## Repeated shape evaluation (accepted 2026-10-12)

```mermaid
  sequenceDiagram
    participant Rule as Subflow Lua rule
    participant Analysis as Report-local C evaluation
    participant Cache as Bounded successful scope cache
    participant Body as Immediate scoped graph
    Rule->>Analysis: infer_subflow(inherited tensor)
    Analysis->>Analysis: check depth and charge request work
    Analysis->>Cache: lookup owner, tensor metadata, depth
    alt successful cached scope
      Cache-->>Analysis: owned typed outputs and descendant rows
      Analysis->>Analysis: charge replay work and restore descendant outcomes
    else cache miss
      Analysis->>Body: evaluate nodes with shared work/instruction budgets
      Body-->>Analysis: mapped outputs and descendant outcomes
      opt all descendants succeeded and cache has capacity
        Analysis->>Cache: deep-copy outputs and descendant rows
      end
    end
    Analysis-->>Rule: copied mapped outputs or explicit budget/runtime fault
```

Evaluation owns and frees cache at report completion; report keeps its normal
owned final rows. Cached results never survive model/parameter/dataset mutation.
Deterministic Lua has no random/randomseed functions. Per-rule limits remain;
100000 work units and 10000000 cumulative instruction units replace call count.

## Diagnostics query through local automation

```mermaid
  sequenceDiagram
    actor Author as User or model author
    participant CLI as nnmodelctl
    participant IPC as AF_UNIX service
    participant Adapter as CommandAdapter
    participant App as NNApplication
    Author->>CLI: analysis.diagnostics {}
    CLI->>IPC: newline-delimited JSON request
    IPC->>Adapter: validate request and dispatch on Qt thread
    Adapter->>App: query the same lazy report as the UI
    alt report available
      App-->>Adapter: complete flag, problems and typed tensors
      Adapter-->>IPC: structured response
      IPC-->>CLI: JSON response
    else whole report unavailable
      App-->>Adapter: explicit analysis error
      Adapter-->>IPC: ok:false and error
      IPC-->>CLI: non-success response
    end
```

The command is read-only and uses the active UI application's single report.
Transport and command details are in
[resource-automation.md](resource-automation.md) and
`contracts/diagnostics.md`.

## Preview and tensor hover

```mermaid
  sequenceDiagram
    actor User
    participant Scene as GraphScene
    participant App as NNApplication
    participant Report as NNInferenceReport
    User->>Scene: expand subflow card
    Scene->>App: read actual immediate-scope nodes and edges
    App-->>Scene: borrowed model snapshot
    Scene->>Scene: build read-only preview items and temporary layout
    Scene->>Scene: derive visual offsets for overlapping peers
    User->>Scene: hover node or preview child
    Scene->>App: request analysis report if needed
    App->>Report: reuse valid report or compute lazily
    Report-->>App: typed outcomes and consumed tensors
    App-->>Scene: borrowed report values
    Scene->>Scene: copy handle, dtype, shape and display text
    User->>Scene: leave, collapse or refresh
    Scene->>Scene: hide popup or discard temporary items
```

Preview borrows identity from the existing graph; it does not create a second
graph or mutate child positions. Hover text is copied into Qt-owned storage.
Expansion, camera and popup state remain transient presentation state.
