# Anonymous user activity log (accepted 2026-11-12)

Explicit user request for a usability study: bachelor students perform the
guided ten-task MNIST MLP assessment on Ubuntu lab machines using the AppImage,
starting from `examples/test` (only Output and Loss Output, MNIST small dataset).
The native client records every user action as an anonymous, parsable append-only
log. Collection and analysis are offline tools; nothing is sent over the network.

## Location, activation and failure policy

The graphical client always logs unless disabled. `NNMODELLING_ACTIVITY_LOG`
selects the target: unset or empty means `$HOME/nnmodelling.log`; `off`, `0`,
`false` or `none` (case-insensitive) disables logging; any other value is the
file path. Without `HOME`, logging is disabled. Only `src/gui/qt/main.cpp` opens
the log; `nnmodelctl`, core tests and Qt widget tests never log unless they
explicitly open one. CTest sets `NNMODELLING_ACTIVITY_LOG=off` for tests that
launch the GUI binary.

The file is opened with `O_WRONLY|O_APPEND|O_CREAT|O_CLOEXEC`, mode `0600`, and
never truncated. Each record is one complete line written by one `write()` call
(no stdio buffering, no fsync), so a crash loses at most the record being written
and concurrent instances never interleave inside short lines. Logging must never
block, abort or alter application behavior: open failure prints one warning on
stderr and leaves logging disabled; a write failure prints one warning and
disables the log for the rest of the process. All logging runs on the Qt
application thread; the C log owner is not thread-safe.

## Ownership and C ABI

New C11 module `src/activity/` (part of `nnmodelling_core`, depends only on
`utils/` and yyjson) owns the writer through public `activity/activity.h`:

```c
typedef struct NNActivityLog NNActivityLog;
/* false + empty error when disabled; false + error on failure; true + path. */
bool nn_activity_resolve_path(char *buffer, size_t capacity, char *error, size_t cap);
NNActivityLog *nn_activity_open(const char *path, char *error, size_t cap);
void nn_activity_close(NNActivityLog *log);
const char *nn_activity_session(const NNActivityLog *log);
/* data_json: NULL or JSON object text. Returns false on write failure. */
bool nn_activity_emit(NNActivityLog *log, const char *source, const char *event,
                      bool ok, const char *error_text, const char *data_json);
/* Owned copy with $HOME prefix replaced by "~"; release with free. */
char *nn_activity_redact_path(const char *path);
```

`NULL` logs are accepted by `emit`/`close` as no-ops. `application.h` adds
`nn_app_set_activity_log(app, log)` (borrowed; caller keeps it alive until
`nn_app_free` or reset to NULL) and `nn_app_set_activity_source(app, source)`
(static string `core` or `ipc`; default `core`) with getter
`nn_app_activity_source(app)`. The automation dispatcher sets `ipc` for the
duration of each request and restores the previous source. Public operations
wrap an unchanged implementation; with a log attached they use a local error
buffer so failures are logged even when the caller passed none, then copy the
text back. `nn_activity_open` only reports its error; `main.cpp` prints the
single stderr warning. A record that cannot be serialized (invalid UTF-8) is
skipped without disabling the log. Qt
wraps the same `NNActivityLog` in a C++ helper and serializes `data` with
QJsonDocument; Qt types never enter the C ABI.

## Record format: JSON Lines, schema version 1

One UTF-8 JSON object per line, terminated by `\n`, keys in this order:

| Key | Type | Meaning |
| --- | --- | --- |
| `v` | integer | schema version, `1` |
| `ts` | string | local wall time `YYYY-MM-DDTHH:MM:SS.mmm+HH:MM` |
| `t` | integer | milliseconds since log open, `CLOCK_MONOTONIC` |
| `session` | string | random UUID v4 generated per process open |
| `seq` | integer | per-session record counter starting at 1 |
| `src` | string | `ui` (Qt presentation event), `core` (application operation requested by the GUI), `ipc` (application operation requested by local automation) |
| `event` | string | dotted event name from the catalog below |
| `ok` | boolean | operation outcome; `true` for pure observations |
| `error` | string | present only when `ok` is false: the error text shown or returned |
| `data` | object | event-specific fields, always present (possibly `{}`) |

Strings in `data` longer than 1024 bytes are replaced by
`{"truncated":true,"len":N,"fnv1a64":"<16 hex digits>"}` (hash of the full UTF-8
bytes). Invalid `data` JSON is replaced by `{"invalid_data":true}`. Consumers must
ignore unknown keys and events; a new incompatible layout increments `v`.

### Anonymity

Records never contain the login name, hostname, machine ID, IP, home directory
or environment. Paths are redacted: a `$HOME` prefix becomes `~`. No raw key
presses, pointer coordinates, clipboard or screenshots are logged. Values the
user deliberately enters (parameter text, names, palette search) are logged
because they are the studied interaction. Participant identity exists only
through the anonymous file name assigned at collection.

## Event catalog

Core events are emitted by `src/application/` after the operation finishes, with
its result, for every call whether successful or failed (a failure logs `ok:false`
and the error buffer text). Field names use snake_case.

| Event | `data` | Application function |
| --- | --- | --- |
| `project.open` | `path` | `nn_app_open` |
| `project.create` | `parent`,`id`,`name`,`version`,`template` (`blank`,`mnist-mlp`,`mnist-vae`,`tiny-llm`) | `nn_app_create`, `nn_app_create_vae`, `nn_app_create_llm` |
| `project.save` | `{}` | `nn_app_save` |
| `project.close` | `discard` | `nn_app_close` |
| `node.add` | `id`,`package`,`version`,`scope`,`x`,`y` | `nn_app_add_node` |
| `node.remove` | `id`,`package` (when known) | `nn_app_remove_node` |
| `node.move` | `id`,`x`,`y` | `nn_app_move_node` |
| `node.rename` | `id`,`label` | `nn_app_rename_node` |
| `edge.connect` | `id`,`source`,`source_handle`,`target`,`target_handle` | `nn_app_connect` |
| `edge.disconnect` | `id` | `nn_app_disconnect` |
| `param.set` | `node`,`key`,`old` (text or null),`new` (text) | `nn_app_set_parameter_text`, `nn_app_set_parameter` (both values use the parameter-text form: raw strings, compact JSON otherwise) |
| `boundary.set` | `node`,`handle` | `nn_app_set_boundary_handle` |
| `edit.undo`, `edit.redo` | `{}` | `nn_app_undo`, `nn_app_redo` |
| `edit.begin`, `edit.end` | `{}` / `commit` | `nn_app_begin_edit`, `nn_app_end_edit` |
| `dataset.create`, `dataset.update` | `id`,`version`(,`select`) | `nn_app_create_dataset`, `nn_app_update_dataset` |
| `dataset.select`, `dataset.delete` | `id`,`version` | `nn_app_select_dataset`, `nn_app_delete_dataset` |
| `dataset.import` | `source_dir` | `nn_app_import_dataset` |
| `stereotype.create` | `id`,`version`,`python` | `nn_app_create_stereotype*` |
| `stereotype.update_source` | `id`,`version`,`kind`,`source` | `nn_app_update_stereotype_source` |
| `operations.set`, `metavariables.set` | `json` | `nn_app_set_operations_json`, `nn_app_set_metavariables_json` |
| `model.snapshot` | see below | after successful semantic mutations |
| `analysis.result` | see below | `nn_app_analysis` |

`model.snapshot` follows (same session, next `seq`) every successful core event
except `node.move`, `node.rename`, `edit.begin`, `project.save` and
`project.close`; inside an open edit group it is deferred to a committing
`edit.end`, and an aborting `edit.end` emits it because the model is restored.
`project.close` emits nothing further. Data: `project` (manifest `id`),
`dirty`, `active_dataset` (`{id,version}` or null), `nodes`
(`[{id,scope,package,version,name,params}]`, params as in model.json) and
`edges` (`[{id,source,source_handle,target,target_handle}]`), in model order.
Positions are excluded; moves carry them. The snapshot lets consumers rebuild
state after undo/redo, because history entries are unnamed model copies.

`analysis.result` is emitted when `nn_app_analysis` computes a new report (not on
cache hits) and its problem signature differs from the last emitted one for the
active project. Data: `problems` (non-success node results plus one for a
non-success root), `root_status`, `nodes` (`[{id,status,message}]` for
non-success node results, capped at 50). Status strings are
`nn_inference_category` values: `success`, `model`, `incomplete`,
`lua-compilation`, `internal`.

Qt `src:"ui"` events, emitted by `src/gui/qt/ActivityLog.*`:

| Event | `data` | Trigger |
| --- | --- | --- |
| `session.start` | `app`,`version`,`qt`,`os` (`PRETTY_NAME` of /etc/os-release or null),`locale`,`platform`,`screen` `[w,h]`,`project_arg`,`automation` (booleans) | after QApplication construction |
| `session.end` | `exit_code` | after the event loop returns |
| `app.focus`, `app.blur` | `{}` | application becomes active/inactive |
| `ui.action` | `action` (objectName, else text),`via` (`shortcut`,`menu`,`toolbar`,`button`,`other`) | QAction trigger or toolbar/inspector button |
| `ui.status` | `text` | status bar message |
| `error.shown` | `where` (`dialog`,`canvas`),`message` | warning/critical message box, `GraphScene::errorOccurred` |
| `dialog.open`, `dialog.close` | `dialog`,`result` (close only: `accepted`,`rejected`, or the chosen button such as `save`,`discard`,`cancel`) | each modal dialog, file chooser and confirmation |
| `palette.search` | `query` | palette filter text, 500 ms debounce |
| `palette.select` | `package`,`version` | palette current item change |
| `palette.add` | `package`,`version`,`via` (`button`,`double_click`) | add request, before `node.add` |
| `selection.change` | `nodes`,`edges` (ID arrays) | coalesced scene selection change |
| `scope.change` | `scope` | active scope change |
| `problems.toggle` | `expanded` | diagnostics panel toggle |
| `problem.click` | `node`,`message` | diagnostics row activation |
| `connect.start` | `node`,`handle`,`output` | port press starts a draft edge |
| `connect.cancel` | `reason` (`escape`,`no_target`) | draft discarded without `nn_app_connect` |
| `drag.start`, `drag.cancel` | `nodes` | node drag begins / is cancelled |
| `join.slots` | `node`,`count` | join `+`/`-` slot adjustment |
| `view.change` | `zoom`,`center` `[x,y]` (scene) | wheel/pan/zoom, 1 s debounce |
| `view.fit`, `layout.arrange` | `{}` / `direction` | fit or arrange |
| `backend.request` | `method`,`path`,`status` | every non-GET backend request and explicit Connect |

## Collection and analysis tools

`tools/activity_log/collect.sh DEST` copies the current user's log to
`DEST/P-<random hex>.jsonl` (non-identifying name) and prints that name.
`tools/activity_log/parse_activity_log.py LOG... --out DIR [--tasks FILE]` is
Python 3.10+ standard library only. It skips and reports malformed or truncated
lines and `seq` gaps, and writes `events.csv`, `sessions.csv`, `tasks.csv` and
`summary.md`. One participant is one input file; sessions within it are ordered
by `ts` and concatenated. Task predicates live in
`tools/activity_log/tasks_bachelor.json`: a task completes at the first record
where its predicate holds over the latest `model.snapshot` (state predicates) or
at a matching successful event (event predicates). Per task: completion time,
elapsed time since previous completion, active time from the first action after
the previous completion, actions, failed operations, undo/redo, errors shown,
palette queries, completion and out-of-order flag. Its tests are in
`tests/test_activity_log_parser.py`.

## Verification

`tests/activity_log_test.c` (core label) covers path resolution, format,
append, escaping, truncation, open failure and application events/snapshots
including undo and a failed operation. A Qt window test checks UI events with a
temporary log. QA runs the GUI with a temporary `NNMODELLING_ACTIVITY_LOG`,
drives the ten tasks, validates every line with `jq` and parses the result.
