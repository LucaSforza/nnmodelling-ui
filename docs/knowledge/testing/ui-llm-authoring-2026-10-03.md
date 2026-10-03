# Computer-use LLM authoring trial — 2026-10-03

The user requested a new blank project, an LLM graph authored through the GUI,
fixes for blocking bugs by GPT-6 Luna high workers, and separate commits for
this report and the resulting example. The project is
`examples/tiny-decoder-llm`; no template or existing example was opened to build
it. Numerical execution and training remain outside this client's scope.

## Environment and method

Codex exposed browser computer use but no native desktop surface in this Linux
session. The actual Qt application ran through Qt's VNC platform plugin and a
local noVNC browser bridge. Mouse clicks, drags and keyboard input were sent
through Codex computer use to the GUI. The bridge is in the ignored repository
folder `.computer-use/`, as requested, rather than `/tmp`. VNC and HTTP listen
only on loopback; bridge dependencies and logs are not committed.

Project creation, dataset/stereotype authoring, parameter edits, node creation,
scope navigation, edge dragging and saving used the GUI. File inspection,
builds, tests and commits used terminal tools. Any structured final analysis
capture supplements GUI verification; it does not author the graph.

## Observed problems

| Problem and evidence | Resolution / practical impact |
| --- | --- |
| Agent skill assumed OpenCode swarm tooling, which is absent in Codex. | Skill and swarm contract/UML now distinguish Codex collaboration from OpenCode V2. `6ebc869`. |
| Boundary circles had hidden port glyphs, including on hover, making mouse connection targets undiscoverable. | Visible contrasting rim handles and hover cues, with unchanged generous hit areas. Luna fix `7f6b115`; real Input-to-Embedding and loss-to-terminal drags exercised. |
| Circles had side ports while computational cards used top/bottom ports; curves disagreed with flow. | Default vertical direction applies to every kind. Horizontal mode flows right to left, matching the user's refinement. Contracts and UML updated before Qt changes. |
| Arrange used a four-column insertion-order grid; terminals came before Input and branches crossed repeatedly. | Topology-aware ranks and separated lanes replace insertion-order wrapping. Both arrangement directions expose matching handles. |
| Fit only framed existing positions and could miscenter after scale clamping; it did not navigate to Input. Qt regression checks also exposed scene-extent constraints after fitting very large graphs. | Toolbar Fit arranges vertically and focuses the first Input at readable scale. Arrange frames the whole scope. Camera extents permit accurate centering and navigation; internal framing remains camera-only. |
| No visible zoom buttons; wheel interaction was unavailable through the VNC bridge. | Explicit Zoom In/Out controls provide centered bounded scaling. |
| Positions saved arbitrary doubles, including fractions introduced by GUI placement. | C model stores signed 32-bit positions on a 20-unit grid, with finite/range checks; save/snapshot emit integers. Qt shows and snaps to the same grid. Luna commit `146be06`. |
| New Repeat contained its terminal but no Input, although inherited analysis requires exactly one. | New subflows now atomically create one Input plus declared terminals at local grid coordinates. Existing scopes are not silently repaired. `146be06`. Trial's Input was inserted manually before the fix. |
| Generic joins appeared as computational cards and lacked explicit input-slot controls. User supplied Add junction reference; subsequent review rejected a third automatically exposed empty input. | Kind-driven junction bars and adjacent +/- controls apply to every `kind=join`, preserving occupied handles and parameter editing. Start with two inputs plus occupied handles; extra empty inputs require explicit +. Empty-slot counts are editing state; connections persist normally. |
| Embedding analysis failed with `attempt to call a nil value (field 'append_dimension')` at its preserved Lua rule, line 17. Host also lacked `with_dtype` and residual Add's `equal`. | Host primitives added without altering core package assets. Shape/dtype preservation, invalid arguments, rank cap, equality and core-package integration are tested. Luna commit `cfdb40a`. |
| Dataset form rows were shallow and text clipped at the initial display size. | Keyboard entry and explicit saved-file inspection allowed authoring. Usability issue remains; dataset metadata was verified. |
| Stereotype authoring dialog exceeded the initial screen height; Save/Lua bottom could be clipped. | Larger Qt screen and keyboard confirmation allowed saving. Dialog sizing/scrolling remains an improvement to make. |
| Inspector parameter controls became clipped after panel resizing; refreshing fields also replaces the widgets. | Focus from Name then Tab allowed reliable editing; values were checked on cards and in saved JSON. Panel discoverability remains limited. |
| Toolbar Fit/Arrange were hidden at the original 1360-pixel window width. | Larger trial window exposed controls. Narrow-window toolbar discoverability remains an improvement. |
| New-project dialog did not prompt for version although UI-release text describes a version field. | Creation uses version `0.1.0`. No data loss; dialog/contract mismatch remains recorded. |

Bridge-specific limitations are distinct from application defects: whole-string
browser typing reached noVNC as only its first character, so input was sent
character by character; the Qt VNC pixel channels needed a browser display
correction; detached local processes were needed to keep the bridge alive.
No app model changes were made to compensate for those bridge limitations.

## Verification record

The core gate passed 11/11 tests after integer grid and automatic Input changes.
Sandbox runs initially failed because ccache's configured temporary directory
was read-only or AF_UNIX bind was denied. Authorized runs outside that sandbox
passed; these are tooling restrictions, not application regressions.
ASan/UBSan with leak detection also passed all 11 tests outside the sandbox;
inside it, LeakSanitizer fails because the sandbox uses ptrace.

A read-only probe linked to the current C library loaded the GUI-saved project
and validated all 19 nodes successfully, including all hidden decoder children
and both root terminals. No project write or graph mutation was performed by
that probe. Output is `float32[B,128,32000]`, loss `float32[]`, inherited decoder
input/output `float32[B,128,512]`, and expanded FFN `float32[B,128,2048]`.

Qt direction/layout/zoom/junction changes passed the full GUI gate, 3/3, and
were committed by Luna as `2df88c9`. The app was restarted with the rebuilt
executable; visible +/− toolbar actions confirmed the new build. GUI checks
exercised vertical arrangement, leftward horizontal arrangement, Fit near
Input, both zoom actions, entering Repeat, junction +/- and preservation of
occupied inputs. Both scopes were arranged vertically and saved. Close/Open
through the GUI reopened the same project successfully; all 19 saved positions
were integers divisible by 20. Structured read-only analysis of the live app
returned `available=true`, `complete=true`, 19 tensors and `problems=[]`.

The two-input join refinement was committed by Luna as `5f17cc4`, after
contracts/UML commit `962c4bf`; the full GUI gate again passed 3/3. The rebuilt
app was restarted again. Both saved Add nodes displayed two receiving handles;
GUI + exposed a third, and - returned to two without losing either connection.
Regression checks cover refresh, connection without automatic expansion and
reopening with two occupied inputs. Final `git diff --check` passed.

The selected dataset has `tokens: int64[B,128]` and next-token
`target: int64[B,128]`. The graph uses vocabulary 32000, width 512, six repeated
decoder blocks, eight-head causal-attention metadata and FFN width 2048.
Expected logits are `float32[B,128,32000]`; Cross Entropy consumes logits and
publishes scalar loss. Its preserved core rule checks floating logits and rank,
but does not deeply verify target compatibility. This project proves client
shape analysis and authoring, not a trained or numerically executed LLM.
