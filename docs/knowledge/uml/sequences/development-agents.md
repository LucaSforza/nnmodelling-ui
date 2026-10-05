# Development agent sequences

The KB-first rules apply in Codex and OpenCode V2, but their delegation APIs are
separate. The principal owns architecture, accepted contracts, knowledge-base
changes and result review. Workers implement only bounded assignments and do
not delegate further.

## Codex bounded implementation

```mermaid
  sequenceDiagram
    actor User
    participant Principal as Codex principal
    participant Collaboration as Native collaboration API
    participant Child as Luna high worker
    User->>Principal: request implementation or explicit parallel work
    Principal->>Principal: read relevant KB and accept bounded contract
    Principal->>Collaboration: spawn_agent with disjoint file ownership
    Collaboration-->>Principal: child identity
    Child->>Child: personally read assigned KB documents
    Child->>Principal: actionable blocker or contract ambiguity
    Principal->>Child: clarification or accepted correction
    Child->>Child: implement assigned files and run focused checks
    Child-->>Collaboration: final result and verification
    Collaboration-->>Principal: completion result
    Principal->>Principal: review changes and run final checks
```

Codex workers use native `collaboration` tools. OpenCode `swarm.init`, mailbox
enrollment and session IDs do not apply to this path. Shared-checkout workers
own disjoint files; Git staging and commits are serialized by the principal.

## OpenCode V2 initialization, launch and mailbox

```mermaid
  sequenceDiagram
    actor User
    participant Principal as Sol orchestrator
    participant Runtime as OpenCode V2
    participant Mailbox as Local swarm plugin
    participant A as Luna child A
    participant B as Luna child B
    User->>Principal: request swarm objective
    Principal->>Principal: read KB, accept contract and assign disjoint files
    Principal->>Mailbox: init before launching any child
    Mailbox->>Runtime: validate principal session
    Mailbox->>Mailbox: persist initialized registry and preserve exclusions
    Mailbox-->>Principal: initialization complete
    Principal->>Runtime: launch child A in background
    Runtime-->>Principal: A session ID
    A->>A: personally read assigned KB documents
    A->>Mailbox: send actionable parent question
    Mailbox->>Mailbox: enroll A automatically on first mailbox use
    Mailbox->>Runtime: deliver synthetic input with resume
    Principal->>Runtime: launch child B in background
    Runtime-->>Principal: B session ID
    B->>B: personally read assigned KB documents
    B->>Mailbox: members or send actionable coordination
    Mailbox->>Mailbox: enroll B automatically
    Principal->>Mailbox: wait for outstanding children and messages
    Mailbox->>Runtime: observe native child waits and inbox events
    alt mailbox message arrives
      B->>Mailbox: send parent actionable blocker
      Mailbox-->>Principal: message event without consuming inbox input
    else child becomes idle
      Runtime-->>Mailbox: native wait completion
      Mailbox-->>Principal: idle event, not proof of success
    else bounded wait expires
      Mailbox-->>Principal: timeout
      Principal->>Mailbox: inspect one status snapshot
      Mailbox-->>Principal: activity, outcome and tool names only
    end
    A-->>Runtime: final implementation result
    Runtime-->>Principal: native child completion
    Principal->>Principal: review implementation and required checks
```

Initialization must succeed before any OpenCode child launch. Direct children
enroll on mailbox use or when addressed; no subsequent register barrier is
needed. Wait observes events and does not consume messages, cancel children or
prove completion. Registry membership/exclusions and all role limits follow
`contracts/agent-swarm.md`.

## CLI model-authoring agent

```mermaid
  sequenceDiagram
    actor User
    participant Principal as Orchestrator
    participant Runtime as OpenCode V2
    participant Mailbox as Local swarm plugin
    participant Builder as model-builder Sol high
    participant CLI as nnmodelctl
    participant App as Existing GUI and C application
    alt user selects model-builder as primary
      User->>Runtime: select visible model-builder
      Runtime->>Builder: provide model-authoring objective
    else principal delegates model authoring
      Principal->>Principal: assign exclusive application and project ownership
      Principal->>Mailbox: initialize before launch
      Principal->>Runtime: launch bounded model-builder child
      Runtime-->>Principal: child session ID
      Runtime->>Builder: socket, project, allowed operations and acceptance checks
      Builder->>Mailbox: discover peers or send actionable blocker
    end
    Builder->>Builder: read KB and verify assigned socket
    Builder->>CLI: project.snapshot
    CLI->>App: read authoritative project state
    App-->>Builder: identity, dirty state, graph and resources
    Builder->>CLI: authorized project/resource/node/edge operations
    CLI->>App: validate and commit each operation
    App-->>Builder: success or explicit error
    Builder->>CLI: diagnostics, arrange, save or capture as requested
    CLI->>App: analyze and persist through existing authority
    App-->>Builder: diagnostics, save and capture results
    alt primary session
      Builder-->>User: actual authored result and limitations
    else child session
      Builder-->>Runtime: final authoring report
      Runtime-->>Principal: native completion
      Principal->>Principal: review diagnostics and save status
    end
```

Only one mutating author may own an application socket and writable project at
a time. Builder has no second graph and cannot edit project files or spawn
workers. Each successful CLI command commits independently; later command
failure does not undo earlier operations.
