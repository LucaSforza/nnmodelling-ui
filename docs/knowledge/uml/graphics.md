# Qt graphics ownership

```mermaid
classDiagram
    QMainWindow <|-- MainWindow
    QGraphicsView <|-- GraphView
    QGraphicsScene <|-- GraphScene
    QGraphicsItem <|-- NodeItem
    NodeItem <|-- SubflowItem
    QGraphicsItem <|-- PortItem
    QGraphicsPathItem <|-- EdgeItem
    MainWindow *-- GraphView
    GraphView --> GraphScene
    GraphScene *-- NodeItem
    GraphScene *-- EdgeItem
    NodeItem *-- PortItem
    GraphScene --> NNApplication : C ABI / stable IDs
    NNApplication *-- NNProject
    NNProject *-- NNModel
```

Qt owns items, clipping, fonts, transforms and event delivery.
Card geometry includes definition-positioned top/bottom parameter rows and
central title; Qt copies text for painting and adapts port positions to height.
C owns graph, coordinates and application mutations. Edge endpoints follow PortItem scene
positions only for rendering. On release, positions commit to C and failed
commits restore C positions. Painting is read-only. SubflowItem owns no child
model: navigation filters C nodes by scope ID, and collapse is transient.
The graphical entry point is `gui/qt/main.cpp`; the C application/core has no
graphical entry point. The former `NNPlatform` adapter and `nn_editor_run` loop
were removed on 2026-09-30, together with their SDL lifecycle smoke test.
No SDL adapter, draw list, custom camera math or C++ graph exists.
