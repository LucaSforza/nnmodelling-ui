# Development swarm sequence

See the [accepted tooling contract](../contracts/agent-swarm.md).

## Runtime routing

```mermaid
flowchart TD
    Request[User requests delegation] --> KB[Principal reads KB and assigns bounded work]
    KB --> Runtime{Current runtime}
    Runtime -->|OpenCode V2| Init[Discover swarm tools and complete swarm.init]
    Init --> Native[Launch native background subagent]
    Runtime -->|Codex| Spawn[collaboration.spawn_agent: Luna high implementation worker]
    Spawn --> Messages[Native collaboration messages and completion]
    Native --> Mailbox[OpenCode mailbox and native completion]
    Messages --> Review[Principal reviews results and runs final checks]
    Mailbox --> Review
```

Shared-checkout workers own disjoint files and serialize Git index/commit
mutations. Runtime limits govern concurrency. Codex requires no OpenCode
initialization or enrollment. The sequences below describe OpenCode V2 only.

```mermaid
sequenceDiagram
    actor User
    participant Principal as Sol orchestrator
    participant Runtime as OpenCode V2
    participant Mailbox as Local swarm plugin
    participant A as Luna child A
    participant B as Luna child B
    User->>Principal: /swarm objective
    Principal->>Principal: Read KB, accept contract, assign disjoint files
    Principal->>Mailbox: init({}) before any child launch
    Mailbox->>Runtime: Validate principal session
    Mailbox->>Mailbox: Persist initialized registry (preserve existing state)
    Mailbox-->>Principal: Initialization complete
    Principal->>Runtime: subagent(A, background=true)
    Runtime-->>Principal: A sessionID
    A->>A: Personally read assigned KB documents
    A->>Mailbox: send(parent, actionable question), possibly before B launches
    Mailbox->>Runtime: Validate direct parent, project, directory and active session
    Mailbox->>Mailbox: Automatically enroll A unless explicitly excluded
    Mailbox->>Runtime: synthetic(principal, resume=true, delivery=steer)
    Runtime-->>Principal: Deliver question / schedule continuation
    Principal->>Runtime: subagent(B, background=true)
    Runtime-->>Principal: B sessionID
    B->>B: Personally read assigned KB documents
    B->>Mailbox: members()
    Mailbox->>Mailbox: Automatically enroll B (serialized registry update)
    Mailbox-->>B: Principal and currently enrolled peers
    Principal->>Mailbox: send(A, clarification)
    Mailbox->>Runtime: synthetic(A, resume=true)
    A->>Mailbox: send(B, coordination message)
    Mailbox->>Runtime: synthetic(B, resume=true)
    Runtime-->>B: Deliver / wake if idle
    Principal->>Mailbox: wait(children=outstanding IDs, timeout_seconds<=270)
    Mailbox->>Runtime: Subscribe to native events before notification snapshot
    Mailbox->>Mailbox: Start one deadline timer
    Mailbox->>Runtime: Observe child idleness
    Mailbox->>Mailbox: Read persisted notification / observed inbox ID
    alt Incoming swarm message
        B->>Mailbox: send(parent, actionable blocker)
        Mailbox->>Runtime: synthetic(principal, resume=true)
        Runtime-->>Mailbox: session.inbox.enqueued
        Mailbox-->>Principal: reason=message (without consuming inbox)
    else Child loop becomes idle
        Runtime-->>Mailbox: Native session.wait resolves
        Mailbox-->>Principal: reason=idle (not necessarily success)
    else Deadline expires
        Mailbox-->>Principal: reason=timeout
        Principal->>Mailbox: status(outstanding IDs), one diagnostic snapshot
        Mailbox->>Runtime: Read child metadata/tools, active sessions if API exposed
        Mailbox-->>Principal: Activity/outcome/tools, no transcript or reasoning
    end
    Mailbox->>Mailbox: Remove listener/timer, abort observation requests only
    A-->>Runtime: Final implementation result
    Runtime-->>Principal: Native background completion
    Principal->>Principal: Review, integrate, run relevant checks
```

The number of independent children is not capped by project instructions.
Each launch follows successful initialization; no later `register` call is needed.
Registry updates serialize within the plugin storage context. Optional explicit
replacement with `register` excludes omitted enrolled children until the principal
explicitly restores them. Repeating `init` preserves membership and exclusions.

## CLI model-authoring agent

`model-builder` is a Sol high agent with `mode: all`; native primary selection
and child invocation both use the same KB-first CLI authoring instructions.
It does not replace the orchestrator's architecture/review ownership.

```mermaid
sequenceDiagram
    actor User
    participant Principal as Orchestrator
    participant Runtime as OpenCode V2
    participant Mailbox as Local swarm plugin
    participant Builder as model-builder (Sol high)
    participant CLI as nnmodelctl
    participant App as Existing GUI / C application
    alt Selected as primary
        User->>Runtime: Select visible model-builder
        Runtime->>Builder: Model-authoring objective
    else Delegated by orchestrator
        Principal->>Principal: Assign exclusive application/project mutation ownership
        Principal->>Mailbox: init({}) before child launch
        Principal->>Runtime: subagent(model-builder, background=true, bounded objective)
        Runtime-->>Principal: Child sessionID
        Runtime->>Builder: Socket, project, allowed mutations, KB and acceptance criteria
        Builder->>Mailbox: members() / actionable send(parent) when needed
    end
    Builder->>Builder: Read KB, verify CLI and assigned socket
    Builder->>CLI: project.snapshot
    CLI->>App: Read authoritative project state
    App-->>Builder: Identity, dirty flag, graph and resources via CLI
    Builder->>CLI: Authorized project/resource/node/edge operations
    CLI->>App: Validate and commit each operation
    App-->>Builder: Success or explicit error via CLI
    Builder->>CLI: analysis.diagnostics, then arrange/save/capture as requested
    CLI->>App: Analyze and persist through existing application authority
    App-->>Builder: Diagnostics, save/capture results via CLI
    alt Primary session
        Builder-->>User: Actual authored model, save status and limitations
    else Child session
        Builder-->>Runtime: Final authoring report
        Runtime-->>Principal: Native completion
        Principal->>Principal: Review against assignment and actual diagnostics
    end
```

One mutating author owns a socket's application and writable project at a time;
disjoint node IDs do not provide independent state. Builder owns no separate
graph and cannot edit project files or spawn workers. CLI failure does not undo
earlier successful commands. Numerical execution/training remain deferred.
