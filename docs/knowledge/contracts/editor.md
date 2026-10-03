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
Boundary nodes remain package-backed and edges remain same-scope. Accepted
2026-10-01: typed-outputs.md governs mapped internal terminals, spawning, circles
and black/red handles/fanout. No cross-scope edges or compiler are introduced.

New child creation requires an existing package-kind subflow scope. Deleting a
nonempty subflow fails visibly until children are removed, preventing orphaning.
Join inputs use numerically ordered in-N handles; C generates and validates
these IDs. Accepted user refinement 2026-10-03: the GUI initially shows exactly
in-1 and in-2 plus any additional occupied handles. Connecting an input does
not automatically expose another empty input. Extra empty handles appear only
through the + control and survive refresh/scope navigation during the session.
C's dynamic port enumeration may expose a first-free candidate; that candidate
does not itself increase the GUI's visible slot count.
Accepted 2026-10-03 from the user's Add screenshot: every package with
kind=join uses a dark junction bar without a computational card backing,
ordered inputs separated from its receiving side and typed outputs separated
from its outgoing side, with its name below/outside the bar. This is
kind-driven, never an Add/Concat/Fork package-ID special case. Vertical flow
uses a horizontal bar with inputs above and outputs below; horizontal leftward
flow rotates the junction geometry with inputs on the right and outputs left.
In vertical flow place circular minus to the left and plus to the right of
the bar, matching the supplied reference; in horizontal flow rotate their
positions above/below the bar. Place parameter rows outside the junction and
keep contrast against the canvas. Plus exposes one additional
free in-N slot; minus removes only the highest displayed free slot, keeping at
least two and every occupied handle. Disable minus if reduction would hide an
occupied handle, and disable plus at 128 displayed inputs. Existing occupied
handles remain visible even beyond that authoring bound. Additional empty slots
are Qt editing state retained by node ID across refresh/scope navigation, not
new graph edges or a new model field; connected handles persist through normal
edges. C remains final connection validator. Name, schema parameter editing,
typed colors, selection, drag and problem markers remain available; controls
must not begin a node drag or connection and refresh is queued after events.

## Appearance and interaction

2026-09-30 additions: [resource authoring](resource-authoring.md) governs
positioned parameter rows, theme-independent contrast, visual creation dialogs,
dataset selection and New MNIST VAE. [Automation](automation.md) governs local
CLI integration. Subflow navigation remains scope-local; recursive type analysis
now delegates through the existing Proxy Lua rule.

Use white canvas, light gray dock panels, compact text, restrained blue accents,
colored computational cards and directed Bezier connections. Boundary circles
and typed output/edge colors override package colors per typed-outputs.md.
Show package identity/details in inspector. Subflows
must look like containers and remain distinguishable from ordinary layers.
Use Qt font metrics, clipping, high-DPI support and events directly, without a
new event dispatch or drawing abstraction. All errors stay visible.
Accepted 2026-10-03: terminal circles retain visible contrasting rim handles
and hover cues per typed-outputs.md; invisible connection hit regions are not
an acceptable substitute for discoverable mouse targets.

The diagnostics presentation is governed by diagnostics.md: problems only,
distinct categories, root-cause grouping, scope filtering and node navigation.
Canvas command rejection opens an English error dialog, not just a status bar.

## Direction and layout

Accepted 2026-10-03 after the user's computer-use review: the default direction
is vertical, from top to bottom. Every incoming handle, including Output and
Loss Output circle handles, sits on the top edge/rim; outgoing handles,
including Input, sit on the bottom edge/rim. Multiple handles keep definition
order and remain separate. Horizontal arrangement flows towards the left:
incoming handles on the right and outgoing handles on the left for all node kinds. Committed and
draft Bezier curves use tangents matching the selected direction.

The toolbar Fit action arranges the current scope vertically, then navigates
to its first Input in stable model order. Accepted user refinement 2026-10-03:
Input must be visible near the top of the viewport with readable scale (at
least 0.7 scene-to-viewport scale), so a long graph may extend below the view.
If no Input exists, Fit frames the scope content instead. Arrange exposes
explicit Vertical and Horizontal choices; either
choice arranges and frames the scope using matching handles. Arrangement is
deterministic and topology-aware: DAG depth orders connected nodes along the
flow axis, branches occupy separate lanes, and disconnected terminal
collectors follow the computational ranks. Node dimensions and spacing prevent
overlap. It never changes edges, handle IDs, tensor semantics or other scopes.
Positions are committed through NNApplication and saved normally. Direction
is Qt presentation state, defaults to vertical when opening a project, and
survives refresh/scope navigation during that window session; no model schema
or C ABI field is added. Internal framing on open/navigation remains camera-only.

Content framing (Arrange and internal framing) uses bounds with margin, recenters after the final
scale is applied, and must not clip large graphs through a minimum zoom clamp.
Empty scopes fit harmlessly. Interactive wheel zoom remains positive and
bounded, including when Fit produced a scale below the normal wheel minimum.
Visible Zoom In (+) and Zoom Out (-) toolbar actions provide the same bounded
camera scaling without requiring a mouse wheel, anchored to viewport center.
The canvas shows a restrained 20-scene-unit grid. Node creation, dragging and
arrangement snap positions to this grid; drag previews and connected edges
follow the snapped positions. Multi-node moves preserve grid-aligned offsets.
The authoritative integer coordinate contract is in model.md; Qt may keep
floating transforms and item dimensions but cannot persist fractional positions.

Wheel zoom is bounded using the actual next/current scale ratio and anchored to
the wheel event's viewport position with native QGraphicsView mapping/transforms,
not global cursor state (which is unavailable in offscreen tests). This is Qt
presentation state only; model coordinates remain unchanged. Native scrollbar
rounding is tolerated within two scene units by interaction tests.
