# Graphics and editor contract

## Metavariable authoring (accepted 2026-10-14)

Model > Manage metavariables opens a project-local staged list with New, Edit,
Remove, Save and Cancel, matching Operations. Forms use Name, Type and Value;
Save calls C validation and keeps invalid forms open with clear errors. Cancel
does not mutate project. Project resources lists declarations and current values.
Every Inspector parameter has one value editor accepting a literal, `$name` or
`$(Lua expression)` in that same control, including nested stereotype fields.
Numeric/text/array fields use text input; boolean, choice and stereotype selectors
are editable so their text area accepts binding syntax directly. No separate
binding field or mode selector. A visible compact hint near Parameters explains
the syntax; controls also expose placeholders/tooltips. Stored bindings display
their source in the editor and a resolved value or diagnostic below it. Literal
references display a friendly package name and nested fields, not raw reference
JSON; bound references show a friendly resolved package and read-only nested
values until explicitly replaced. Preview labels and editors must fit their tree
row at normal and enlarged fonts, without overlapping following rows. Core owns
parsing, resolution and validation. Literal strings starting with $ use $$.
Refresh after variable edits updates shapes, counts and Inspector previews.

Positioned parameter rows on canvas nodes in the active scope display concrete
values from the C resolved-parameter query.
They never display internal `$ref`/`$lua` JSON or Lua source. Stereotype values
display friendly package names; arrays retain compact readable list notation.
Resolution failure displays `?` with the diagnostic in a tooltip and normal
problem markers. Displaying a value never replaces the authored binding or
changes project dirty/history state.

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
child count and a prominent `Enter subflow` button. It never expands child
previews; the button enters its scope, and scope navigation returns to
parent/root. Leaving a scope never mutates its children. Item parentage is
graphical only; never infer graph membership from a QGraphicsItem parent.
Child positions remain scope-local C values.
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
dataset selection and template creation. [Automation](automation.md) governs
local CLI integration. Subflow navigation remains scope-local; recursive type
analysis now delegates through the existing Proxy Lua rule.

Accepted 2026-10-05: the shared MainWindow stylesheet owns an explicit light
background and dark foreground for all app-owned QDialog surfaces, including
resource forms, project chooser, input and message dialogs. Platform dark
palettes must not combine with inherited dark label/checkbox text. Verify actual
dialog rendering under a dark application palette, not only stylesheet strings.

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
then frames nodes, labels and routed edges with margin. It never
moves nodes or changes flow direction. Arrange retains Vertical/Horizontal choices,
uses deterministic DAG ranks, barycentric branch ordering, port-centered lanes and spaced corridors,
commits positions through C, refreshes routes and frames content. Node rectangles
must not overlap. Direction is session Qt presentation state, default vertical.
No edge/handle/tensor semantics or other-scope coordinates change.

## Descriptions, tensors and navigation

The left palette includes a component guide. It starts with concise instructions,
then shows the selected package description. Selecting a graph node shows its
package description in the right Inspector. Package descriptions are user-facing
and explain Input dataset binding, Output model result, Loss computation and
Loss Output objective. Descriptions do not replace package identity or parameter
schemas.

Node hover shows all successful C-inferred outputs beside the node: handle name,
dtype and shape. Terminal consumed tensors may also be shown. Copy report text;
retain no borrowed report pointers across mutation. Leave hides popup; refresh
clears it. No Qt inference.

Scope selector uses a real Qt tree/model-view with Root and actual scope_id containment,
current item highlighted, and refreshed labels/membership after rename/add/remove.
Imported orphan scopes remain explicitly accessible; malformed/cyclic containment must
not hang tree building.

Application actions use File, Model and View QMenuBar/QMenu/QAction menus with
standard shortcuts. File includes template submenu; Save As remains omitted
until supported. Visible graph controls show Ctrl++ for Zoom In, Ctrl+- for
Zoom Out, Ctrl+F for Fit, arrow keys for panning, Ctrl+Shift+V for vertical
Arrange and Ctrl+Shift+H for horizontal Arrange. Arrow keys pan while the graph
view has focus. Graph toolbar retains compact scope/fit/zoom/arrange controls
without application buttons.
On the Qt VNC platform, recreate closed popup containers before reuse to avoid
stale window composition. Preserve their QAction and tree objects and state;
the frontend still owns them. This does not change desktop menu semantics.

## Orthogonal routing

Routes are explicit Qt editor polylines, separate from painting and C graph semantics.
Only horizontal/vertical segments and right-angle bends; no Bezier/diagonal connection
segments. Rectangles including labels plus a small margin are
obstacles. Source/target allow only port escape/entry; unrelated interiors are forbidden.
Route scopes independently, excluding graphical ancestor boundaries from their
internal obstacles. Recompute all affected routes on geometry/direction, refresh
and drag (including unrelated obstacle movement), before Fit.
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

Right-clicking a node or edge opens an item-specific menu. A node menu includes
Delete and, for subflows, `Enter subflow`; an edge menu includes Delete
connection. Menus select the target item before action and submit changes through
the existing C application API and edit history.

Accepted 2026-10-22: the palette's component guide updates from package selection,
and the Inspector shows a selected node's package description. Visible shortcuts
include Ctrl++, Ctrl+-, Ctrl+F, arrow-key panning, Ctrl+Shift+V vertical Arrange
and Ctrl+Shift+H horizontal Arrange. Menus and toolbar tooltips display each
shortcut.

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
If restored graph removes current scope, return to Root. Camera, hover,
selection and flow direction remain transient UI state and are not semantic history.

Save retains history and marks the saved revision; undo/redo dirty state reflects
whether current revision equals that saved revision. Successful project replacement/
close clears history; failed replacement preserves it. Resource creation persists
files and all graph edits, so success creates a history barrier. Dataset selection
also creates a barrier in this initial graph-history implementation. Neither barrier
pretends to undo filesystem/resource operations. Failure preserves history.
History lives in memory only; schema v2 and package semantics stay unchanged.

Accepted 2026-10-08: changing exported operation metadata also creates an
unsaved history barrier. Reapplying an identical list or rejecting invalid JSON
preserves history and dirty state. Later graph undo/redo cannot clear these
unsaved manifest edits; normal save or close persists them.

## Stereotype reference inspector (accepted 2026-10-05)

Stereotype parameters expose an active-catalog package selector filtered by the
parameter's optional kind, plus schema-derived fields for its parameters. The selector
shows exact ID/version; choosing a new package seeds its defaults. Existing
references retain their parameters until explicitly edited. Boolean and choice
fields use editable selectors, while integers, finite numbers and strings use
text fields; the shared metavariable editor contract above governs both literals
and bindings in each same control.
Each field submits one updated reference using current committed sibling values.
JSON-array parameters retain their existing array editor; no whole-reference
JSON entry is required. Missing imported references remain visibly unset and can be repaired. Submit the object through
normal C parameter validation; rejection preserves graph/history and fields.
No package-ID switches or string-encoded object storage. Core owns reference
resolution, validation, defaults and bounded inference; Qt owns controls only.

## Dataset manager (accepted 2026-10-05)

Accepted 2026-10-08: Model > Parameter counts opens a read-only network total
and per-node total/trainable breakdown from the C application query specified
in lua.md. Unknown/error rows stay visible and incomplete totals are explicitly
labelled. Subflow owner totals include replicated descendants, with no double
addition into the network total. Refresh uses current graph, including unsaved
parameter edits; closing the dialog writes nothing.

Project resources retains New dataset and adds Dataset, opening the manager in
resource-authoring.md. Exact dataset identity is stable while a shared prefilled
form edits metadata/tensor slots. C validates, persists and refreshes analysis;
Qt only owns form/selection state. Shape fields accept comma-separated
dimensions; `B` means batch dimension. Explain `B,4` as four features per
example and `B,1` as one scalar per example beside the field. No dataset
deletion or Python/data editor.
