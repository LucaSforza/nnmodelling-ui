# Deferred interfaces

## Backend connection

Current repository has no backend implementation, backend client, HTTP server,
or training service. Future backend adapter will consume a validated immutable
project/package snapshot from application API. It cannot own or mutate active
graph; responses become application diagnostics or job state, not a second
model. Endpoint, authentication, protocol, jobs and dataset transfer are OPEN.
No project file stores credentials. Existing PyTorch package files are copied
only for compatibility with a future resolved bundle.

## Local automation

`nnmodelctl` is deferred. If implemented, companion CLI sends typed requests
over local IPC to running application. Unix domain socket is preferred on
Unix-like systems; other-platform transport is OPEN. The application command
adapter calls the same model/editor operations as UI. No MCP, WebSocket or
browser automation layer is planned. Semantic target IDs are stable and
UI introspection returns a read-only tree. Screenshot waits for layout/frame
completion before capture. Commands, payload schema, permissions, lifecycle,
error codes and screenshot format are OPEN until a milestone defines them.

Formal: `GraphAuthority(UI)=GraphAuthority(CLI)=Application`.
Natural: command interface never keeps a second graph.

Formal: `screenshot(request) ⇒ captureFrame ≥ layoutCommittedFrame`.
Natural: screenshot reflects completed layout, including requested arrange.

Legacy MCP use cases for node creation, connection, parameter editing, layout,
project creation/opening and screenshot are candidates for adaptation. Training,
monitoring and wheel download depend on future backend integration and are out
of current scope. Deleting or authoring project resources requires its own
accepted transaction contract before exposure by either UI or CLI.
