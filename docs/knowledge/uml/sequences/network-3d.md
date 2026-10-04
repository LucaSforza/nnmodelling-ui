# Network 3D sequences

## Build the 3D view

**Purpose.** Show how the concrete Network3DView widget rebuilds its C-owned
scene from the active project and presents a fresh camera frame.

```mermaid
sequenceDiagram
    actor User
    participant Main as MainWindow Qt Widgets
    participant View as Network3DView Qt widget
    participant App as NNApplication C API
    participant Project as Active NNProject
    participant Viz as C 3D visualization API
    participant Lua as Sandboxed Lua visualization runtime
    User->>Main: select Network 3D tab
    Main->>App: nn_app_project()
    App-->>Main: borrowed active project or null
    Main->>View: rebuild(project)
    View->>View: dispose prior frame and scene
    View->>Viz: nn_3d_build(project) when project exists
    Viz->>Project: read graph and active catalog
    Viz->>Lua: evaluate bounded recipe when subflow package declares entrypoint
    Lua-->>Viz: declarative composition or script error
    Viz->>Viz: validate plans, expand occurrences, and lay out scene
    Viz-->>View: owned scene or explicit error
    View->>Viz: nn_3d_camera_home() and nn_3d_frame(...) when scene built
    Viz-->>View: depth-sorted projected primitives
    View->>View: paint current frame or show empty/error state
    Note over View,Project: Null project skips nn_3d_build, build failure leaves no stale scene displayed
    Note over Viz,Lua: Lua composition is optional per package and bounded by the 3D contract
```

## Navigate the 3D camera

**Purpose.** Trace repeated camera input through the widget’s C camera API and
frame repaint without changing the graph model.

```mermaid
sequenceDiagram
    actor User
    participant View as Network3DView Qt widget
    participant Viz as C 3D visualization API
    loop while user explores current scene
        User->>View: Fit, Home, look, turn, fly, or zoom input
        View->>Viz: nn_3d_camera_fit/home or update camera from input delta
        View->>Viz: nn_3d_frame(scene, camera, viewport size)
        Viz-->>View: depth-sorted projected primitives
        View->>View: paint frame with QPainter
    end
    Note over User,View: Controls: Fit/Home, right-drag, arrows, WASD, Q/E, and mouse wheel
    Note over View,Viz: Camera and frame are presentation state, graph remains unchanged
```

## Select a 3D occurrence

**Purpose.** Show how a pointer hit maps from the current projected frame to a
specific expanded occurrence and its copied source details.

```mermaid
sequenceDiagram
    actor User
    participant View as Network3DView Qt widget
    participant Viz as C 3D visualization API
    User->>View: click projected operator face
    View->>Viz: nn_3d_pick(current frame, pointer position)
    Viz-->>View: occurrence index or no hit
    View->>Viz: nn_3d_node_at(index) after a hit
    Viz-->>View: source ID, occurrence path, and parameters
    View->>View: copy C strings and display selection details
    Note over View,Viz: Picking queries disposable scene identity, it does not mutate project graph
```
