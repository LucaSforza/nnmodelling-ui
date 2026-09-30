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
[resource transaction](resource-authoring.md). Resource deletion, training,
monitoring and wheel download remain deferred.
