# Graphics and editor contract

Accepted 2026-09-29: Qt 6 Widgets and Graphics View replace the SDL draw API.
Cleanup accepted 2026-09-30: remove the superseded SDL adapter, C editor loop,
C graphical entry point and platform smoke test. Qt is the only graphical
frontend; no compatibility platform/draw API is retained. C application/model
APIs and behavior are unchanged; vendored font/rasterization assets remain
preserved per the native-client decision.

## Ownership

QApplication owns GUI lifecycle. MainWindow owns native toolbars, palette,
inspector, resources and diagnostics. GraphView/QGraphicsView owns pan, bounded
positive zoom and fit; GraphScene/QGraphicsScene owns selectable retained items.
NodeItem and SubflowItem reference stable C node IDs; PortItem references node
and handle IDs; EdgeItem/QGraphicsPathItem references a stable C edge ID.
No Qt or C++ type crosses application.h or appears in public core headers.

Painting and selection do not mutate C data. All graph changes go through
NNApplication. Dragging uses transient item positions; release commits positions
through C, then resynchronizes. Connected edge paths follow ports during preview.
Failed changes display an error and restore the model-backed view. Refresh must
not recursively submit movement or destroy an item inside its event handler.
Selection uses Qt, including rubber-band and modifier-based multiple selection.
Delete removes selected edges/nodes through C. Draft edges are visual only until
C validates direction, handle existence, occupancy, same scope and acyclicity.
Qt does not independently decide neural-network compatibility; Lua diagnostics
remain in C and incomplete graphs stay editable.

## Subflows and coordinates

Persisted data.scope remains unchanged: empty means root; child nodes reference
their owning subflow's stable ID. Existing named scopes remain accessible even
without a matching parent, without silently reparenting imported data. Show one
immediate scope at a time. SubflowItem is a visually distinct framed card with
child count and an entry affordance; double-click enters its scope, scope
navigation returns to parent/root. Collapsing or leaving a scope never mutates
its children. Item parentage is graphical only; never infer graph membership
from a QGraphicsItem parent. Child positions remain scope-local C values.
Existing boundary Input/output/proxy nodes and external edges remain ordinary
package-backed nodes and same-scope edges. This migration does not introduce
cross-scope edges or new compiler/subflow semantics.

New child creation requires an existing package-kind subflow scope. Deleting a
nonempty subflow fails visibly until children are removed, preventing orphaning.
Join inputs expose existing numerically ordered in-N handles and the first free
positive slot (at least two initially); C generates and validates these IDs.

## Appearance and interaction

2026-09-30 additions: [resource authoring](resource-authoring.md) governs
positioned parameter rows, theme-independent contrast, visual creation dialogs,
dataset selection and New MNIST VAE. [Automation](automation.md) governs local
CLI integration. Subflow navigation remains scope-local; recursive type analysis
now delegates through the existing Proxy Lua rule.

Use white canvas, light gray dock panels, compact text, restrained blue accents,
colored stereotype cards, visible dark ports and directed Bezier connections.
Respect package colors and show package identity/details in inspector. Subflows
must look like containers and remain distinguishable from ordinary layers.
Use Qt font metrics, clipping, high-DPI support and events directly, without a
new event dispatch or drawing abstraction. All errors stay visible.
The diagnostics presentation is now governed by diagnostics.md: problems only,
distinct categories, root-cause grouping, scope filtering and node navigation.
Canvas command rejection opens an English error dialog, not just a status bar.

Wheel zoom is bounded using the actual next/current scale ratio and anchored to
the wheel event's viewport position with native QGraphicsView mapping/transforms,
not global cursor state (which is unavailable in offscreen tests). This is Qt
presentation state only; model coordinates remain unchanged. Native scrollbar
rounding is tolerated within two scene units by interaction tests.
