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
colored computational cards and directed orthogonal connections. Boundary circles
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
incoming handles on the right and outgoing handles on the left for all node kinds. Committed and draft connections use only horizontal/vertical segments aligned with the selected port direction.

Accepted 2026-10-03 editor revision: Fit is camera-only. It first updates routes,
then frames nodes, labels, inline previews and routed edges with margin. It never
moves nodes or changes flow direction. Arrange retains Vertical/Horizontal choices,
uses deterministic DAG ranks, barycentric branch ordering, port-centered lanes and spaced corridors,
commits positions through C, refreshes routes and frames content. Node rectangles
must not overlap. Direction is session Qt presentation state, default vertical.
No edge/handle/tensor semantics or other-scope coordinates change.

## Inline previews, tensors and navigation

Expand/collapse and enter are separate actions. Subflow cards expose expand/collapse;
double-click and an Enter action still navigate scope. Expansion retains current scope,
shows real immediate children and edges within a labelled boundary and may recursively
show expanded nested owners. Child items borrow model identity, never create another
NNModel; preview geometry uses the same dataflow layout as Arrange, normalized and uniformly
scaled to a bounded preview area, without persisting child coordinates. Put the
owner header above its contents; align external ports with the container center.
Expanding temporarily pushes overlapping current-scope peers along flow direction,
with grid-aligned visual offsets derived from C positions and item bounds. Recompute
these offsets on refresh; collapse restores compact model-backed placement. Drag
commits subtract the visual offset; Arrange persists its computed scope layout,
then refreshes offsets (zero for nonoverlapping placement). Fit never computes a
new placement. These offsets are Qt presentation only, not another graph/history.
Preview is read-only:
no drag, selection mutation, edge drafting or join-slot mutation inside it. Hover remains
available. Expansion IDs survive refresh/navigation during the project session and reset
on project replacement; deleting an owner removes its expansion state.

Node hover shows all successful C-inferred outputs beside the node: handle name, dtype,
shape. Terminal consumed tensors may also be shown. Copy report text; retain no borrowed
report pointers across mutation. Leave hides popup; refresh clears it. No Qt inference.
This applies to current-scope nodes and preview children. Hover popup is excluded from Fit.

Scope selector uses a real Qt tree/model-view with Root and actual scope_id containment,
current item highlighted, and refreshed labels/membership after rename/add/remove.
Imported orphan scopes remain explicitly accessible; malformed/cyclic containment must
not hang tree building or preview recursion.

Application actions use File, Model and View QMenuBar/QMenu/QAction menus with standard
shortcuts. File includes template submenu; Save As remains omitted until supported.
Graph toolbar retains compact scope/fit/zoom/arrange controls without application buttons.

## Orthogonal routing

Routes are explicit Qt editor polylines, separate from painting and C graph semantics.
Only horizontal/vertical segments and right-angle bends; no Bezier/diagonal connection
segments. Rectangles including labels and preview boundaries plus a small margin are
obstacles. Source/target allow only port escape/entry; unrelated interiors are forbidden.
Route current and preview scopes independently, excluding graphical ancestor boundaries
from their internal obstacles. Recompute all affected routes on geometry/direction,
expansion, refresh and drag (including unrelated obstacle movement), before Fit.
Deterministic rectilinear visibility routing prefers obstacle-free paths first, then
penalizes shared tracks and crossings, bends and distance. Use parallel tracks where
space permits. Shared source stems may be unavoidable; crossings show no junction dots.
If endpoints/obstacles overlap and no safe route exists, omit failed route and expose a
visible routing diagnostic rather than draw through nodes. Cache remains Qt-only;
no persistence/schema/C ABI changes.

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

## Undo and redo (accepted 2026-10-03)

C NNApplication owns one bounded, project-local graph edit history (100 edits).
Graph add/remove (including incident edges and spawned subflow terminals), connect/
disconnect, move, rename, parameter and boundary-mapping changes are reversible.
Each successful changed command records one edit; failed and no-op commands preserve
history and redo. Multi-node drag, Delete selection and Arrange use a C edit group,
committing one history entry or rolling back the entire group on failure.
Qt exposes Edit > Undo/Redo using standard shortcuts and enabled state from C;
Qt stores no parallel command/model history. Undo/redo restore owned C snapshots,
invalidate analysis, refresh graph/tree/inspector/diagnostics and recompute routes.
If restored graph removes current scope, return to Root. Camera, expansion, hover,
selection and flow direction remain transient UI state and are not semantic history.

Save retains history and marks the saved revision; undo/redo dirty state reflects
whether current revision equals that saved revision. Successful project replacement/
close clears history; failed replacement preserves it. Resource creation persists
files and all graph edits, so success creates a history barrier. Dataset selection
also creates a barrier in this initial graph-history implementation. Neither barrier
pretends to undo filesystem/resource operations. Failure preserves history.
History lives in memory only; schema v2 and package semantics stay unchanged.
