---
name: nnmodelling-vnc
description: Drive the real NNModelling Qt app for GUI/QA trials in Claude Code (or any agent without browser/desktop computer use) through Qt's VNC platform plugin and a stdlib Python RFB client - real clicks, drags, keys and PNG screenshots.
---

# NNModelling VNC trial (Claude Code)

Read `../use-nnmodelling-kb/SKILL.md` and the relevant KB first. This is the
Claude Code counterpart of `nnmodelling-browser` (Codex drove noVNC in a
browser): here the agent itself is the VNC client. It drives the actual Qt
Widgets app; it is not a reimplementation. `nnmodelctl` or direct project-file
edits must not replace a requested GUI trial.

1. Rebuild first after every code change: `just build`. Restart the session after
   each rebuild; screenshots from an older process are invalid evidence.
2. Use a disposable COPY of the project (e.g. `cp -r examples/test .computer-use/vnc/study`),
   never edit `examples/` in place.
3. Start: `.agents/skills/nnmodelling-vnc/scripts/session.sh start [--socket PATH] [PROJECT]`.
   Without a project the startup chooser appears (preferred for human-open trials).
   Environment variables pass through, e.g. `NNMODELLING_ACTIVITY_LOG=.computer-use/vnc/a.log`.
   Defaults: port 5998 (`NN_VNC_PORT`), 1600x1000 (`NN_VNC_SIZE`), loopback-only bind
   via the `nnmodelling-browser` loopback shim. Log: `.computer-use/vnc/qt.log`.
4. Drive: `python3 .agents/skills/nnmodelling-vnc/scripts/vnc.py --port 5998 STEP...`
   with steps `shot PATH`, `crop X Y W H PATH` (2x zoom), `click X Y`, `dclick`,
   `rclick`, `drag X1 Y1 X2 Y2`, `wheel X Y N`, `type TEXT`, `key Return|Escape|ctrl+z|...`,
   `wait S`. Always take a fresh `shot` and read the PNG before choosing
   coordinates; verify each action's effect visually. Coordinates are framebuffer pixels.
5. Optional `--socket` lets `nnmodelctl ui.inspect`/`project.snapshot` supplement
   (never replace) visual verification.
6. Close the project/window through the GUI when the trial needs it, then
   `session.sh stop` (kills only the recorded nnmodelling-ui pid).

Limitations: no window manager, so top-level dialogs appear without decorations;
drag-and-drop between windows is not available. Report tooling limitations
separately from application defects.
