---
name: nnmodelling-browser
description: Build, launch and interact with the native NNModelling Qt app through its local noVNC browser integration, then shut down the test session. Use for real GUI trials in Codex when native desktop surfaces are unavailable.
---

# NNModelling browser trial

Read `../use-nnmodelling-kb/SKILL.md` and relevant KB before changing code.
This drives the actual Qt Widgets app, not a web reimplementation. Runtime files,
logs, sockets, screenshots and dependencies stay in ignored `.computer-use/`.
Linux Qt VNC platform plugin, C compiler, Node/npm, Python 3 and Qt build tools
must be available. Other platforms are unsupported by this bridge.

## Mandatory fresh-session rule

After EVERY code/build change, rebuild and restart Qt AND browser bridge before
any GUI verification. Never test an already running old session. If any session
exists, close/save its project through GUI first, run `bridge.py stop`, build,
then `bridge.py start PROJECT_DIR` and reload browser tab. Session executable
SHA-256 is recorded; `status` reports stale if rebuilt executable differs.
A matching hash does not prove source changes were built: always rebuild first.
Reconfirm actual UI after restart; screenshots from earlier process are invalid
for changed implementation. Preserve user work when closing an old session.

From repository root:

1. Build: `CCACHE_DISABLE=1 just build`. Run relevant `just test`/`just test-ui`
   separately; compiler flags already enforce warnings. No formatter/linter recipe
   currently exists. Local socket/network tests need execution outside sandbox
   when sandbox blocks bind; report actual failures.
2. Start: `python3 .agents/skills/nnmodelling-browser/scripts/bridge.py start PROJECT_DIR`.
   Optional `--build` runs `just build`; omit project for an empty app. Start
   prepares existing `.computer-use/` integration, installs locked npm dependencies
   only if missing, compiles loopback confinement and waits up to 15 seconds for
   VNC/HTTP readiness. It refuses live managed processes; stop session first.
   Use a disposable copy of examples for edits, preserving user changes.
3. Use `mcp__cua_repl.js`: first `await cua.getState()` to discover surfaces;
   then `let tab = await cua.createBrowserTab("iab", "http://127.0.0.1:8797", {visible:true})`.
   Read returned browser documentation. Snapshot/screenshot before coordinate
   decisions. Native surfaces can be absent while browser input works.
4. App is a canvas: use documented tab Computer Use screenshot/coordinate APIs.
   Inspect fresh UI after actions. For DOM-only browsers use documented
   `tab.playwright`/`tab.cua` methods. Do not invent browser APIs or manipulate
   noVNC RFB internals. Whole-string typing may send only first character:
   send characters individually using documented keyboard/type API and verify.
   Browser screenshots need included SVG red/blue channel correction for Qt VNC.
   Menus, scrollbar pan, zoom buttons, node drag and hover operate actual Qt.
   If ports remain occupied without PID files, inspect process command lines and
   confirm they belong to this disposable session before stopping them.
   CLI inspection can supplement proof; it cannot replace requested UI trials.
5. Close project/window through GUI, saving or discarding disposable edits as
   intended. `python3 .agents/skills/nnmodelling-browser/scripts/bridge.py stop`
   then stops only identified local test Qt/bridge processes, waits and cleans
   their inactive socket/PID files. It never deletes project data. `status`
   reports process/port readiness. Close browser tab through Computer Use.

Bridge binds HTTP/VNC only to loopback. Never expose unauthenticated VNC publicly.
If start fails, inspect `.computer-use/qt.log` and `browser.log`; report unsupported
Qt VNC/tooling rather than substitute CLI model edits for GUI verification.
