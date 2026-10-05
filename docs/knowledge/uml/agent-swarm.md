# Development swarm ownership

See the [accepted tooling contract](../contracts/agent-swarm.md). Detailed
interaction sequences are maintained in [development-agents](sequences/development-agents.md).

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
mutations. Runtime limits govern concurrency. Codex uses native collaboration
tools and has no OpenCode initialization or enrollment step. OpenCode V2
protocol sequences live in [development-agents.md](sequences/development-agents.md).
