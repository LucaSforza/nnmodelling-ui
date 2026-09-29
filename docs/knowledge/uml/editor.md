# Editor and C application boundary

```mermaid
classDiagram
    class NNApplication {
      -NNProject project
      +open(path) bool
      +create(parent,id,name,template) bool
      +save() bool
      +close(discard) bool
      +addNode(id,package,scope,position) bool
      +connect(id,source,handle,target,handle) bool
      +setParameter(node,key,value) bool
    }
    class MainWindow {
      +palette
      +inspector
      +resources
      +diagnostics
    }
    class GraphScene {
      +scopeId
      +selection
      +connectionDraft
      +refresh()
    }
    MainWindow --> NNApplication : pure C ABI
    MainWindow *-- GraphScene
    GraphScene --> NNApplication : snapshots / mutations
    NNApplication *-- NNProject
    NNProject *-- NNModel
```

C application owns active project and validates commands. Qt owns widget state,
selection, scope navigation and gestures. Snapshots are borrowed only until the
next C mutation; Qt copies stable IDs when retaining references. Parameter
widgets follow schema, but C remains final validator. ID-based refresh restores
selection where entities survive. Failed operations preserve committed state.
