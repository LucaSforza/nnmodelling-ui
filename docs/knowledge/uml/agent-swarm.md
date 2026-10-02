# Development swarm sequence

See the [accepted tooling contract](../contracts/agent-swarm.md).

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
        Mailbox->>Runtime: Read child metadata/tools; active sessions if API exposed
        Mailbox-->>Principal: Activity/outcome/tools, no transcript or reasoning
    end
    Mailbox->>Mailbox: Remove listener/timer; abort observation requests only
    A-->>Runtime: Final implementation result
    Runtime-->>Principal: Native background completion
    Principal->>Principal: Review, integrate, run relevant checks
```

The number of independent children is not capped by project instructions.
Each launch follows successful initialization; no later `register` call is needed.
Registry updates serialize within the plugin storage context. Optional explicit
replacement with `register` excludes omitted enrolled children until the principal
explicitly restores them. Repeating `init` preserves membership and exclusions.
