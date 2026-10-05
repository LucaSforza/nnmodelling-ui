# External interfaces

## Backend connection

Accepted 2026-10-05: [backend](backend.md) defines local FastAPI/OpenAPI server,
container jobs, immutable snapshots, generic PyTorch execution, dataset adapters
and wheel export. Qt consumes job state through asynchronous HTTP; C application
remains active graph authority. No project file stores credentials. Multi-user
remote hosting and distributed scheduling remain deferred.

## Local automation

Implemented for Linux AF_UNIX per [automation](automation.md), accepted
2026-09-30. The C command adapter calls the same application operations as UI.
No MCP, WebSocket or browser automation layer is used. Other-platform transport
remains deferred. Introspection uses stable semantic IDs; screenshot commits
layout and scene synchronization before capture. Request schema, permissions,
commands and lifecycle are no longer OPEN: automation.md specifies them.

Formal: `GraphAuthority(UI)=GraphAuthority(CLI)=Application`.
Natural: command interface never keeps a second graph.

Formal: `screenshot(request) ⇒ captureFrame ≥ layoutCommittedFrame`.
Natural: screenshot reflects completed layout, including requested arrange.

Node creation, connection, parameter editing, layout, project lifecycle,
screenshot and resource creation are supported. Authoring uses the accepted
[resource transaction](resource-authoring.md). Resource deletion remains deferred.
Training, monitoring and wheel download use HTTP backend, not local CLI protocol.

Accepted 2026-10-05: [backend contract](backend.md) and [backend UML](../uml/backend.md) supersede earlier backend/training deferrals. Native C11 graph authority and Lua shape analysis remain unchanged.
