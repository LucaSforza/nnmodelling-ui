# Network 3D explorer

The C11 visualization module builds a disposable occurrence scene from the
existing model and active package catalog. Qt translates input and paints C
projections. Sources: `src/visualization/`, `src/gui/qt/Network3D*` and
`src/gui/qt/MainWindow*`.

## Activate, inspect and rebuild

```mermaid
  sequenceDiagram
    actor User
    participant Qt as Network 3D tab
    participant App as NNApplication
    participant Scene as C expanded scene
    participant Catalog as Active package catalog
    participant Lua as Bounded visualization Lua
    User->>Qt: activate Network 3D
    Qt->>App: read active model and catalog
    App-->>Qt: borrowed model/catalog view
    Qt->>Scene: build disposable scene
    Scene->>Catalog: resolve subflow plans and referenced operators
    Catalog-->>Scene: exact definitions and visualization entrypoints
    loop Each nested subflow occurrence
      Scene->>Lua: evaluate plan with validated owner parameters
      Lua-->>Scene: body instances, synthetic operators and mapped edges
      Scene->>Scene: recursively expand body occurrences and validate limits
    end
    alt scene is valid
      Scene->>Scene: layout groups and occurrence paths
      Scene-->>Qt: owned scene snapshot
      Qt->>Qt: paint projected geometry and labels
    else plan, containment or bound fails
      Scene-->>Qt: explicit error and no partial scene
      Qt->>Qt: clear stale scene and show error
    end
    User->>Qt: edit graph in 2D and reactivate 3D
    Qt->>Scene: discard prior presentation scene and rebuild
```

Expanded bodies are occurrences of the source graph, not copied model nodes or
claims about parameter sharing. Occurrence paths combine source IDs and recipe
instance IDs. No write-back, dirty-state, history, analysis or persistence
change occurs. Missing visualization Lua uses the default single-body plan.

## Camera and picking

```mermaid
  sequenceDiagram
    actor User
    participant Qt as Qt input and painter
    participant Camera as C camera and projection
    participant Scene as C occurrence scene
    User->>Qt: fly, look, arrow turn, Fit or Home
    Qt->>Camera: translate widget input into camera operation
    Camera->>Scene: update transient camera or request frame
    Scene->>Camera: scene geometry
    Camera-->>Qt: sorted projected polygons, lines and labels
    Qt->>Qt: paint frame and current selection
    User->>Qt: pick projected object
    Qt->>Camera: query hit at viewport point
    Camera->>Scene: resolve projected hit
    Scene-->>Qt: source node, owner group and occurrence path
    Qt-->>User: show selected occurrence details
```

Only C owns camera math, projection, depth order and picking. Camera and
selection are transient. The independent 2D editor continues to own persisted
scope-local layout; switching tabs does not mutate it.
