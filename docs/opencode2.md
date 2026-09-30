# OpenCode V2 swarm

The repository uses `opencode2` (tested with V2.0.1), not the legacy `opencode`
executable. Configuration is project-local; provider credentials remain in your
existing OpenCode authentication store.

## Start

Requirements: OpenCode V2, Node.js with `node --test`, `just`, a working
OpenAI connection, and the fff MCP already configured in OpenCode. The project
inherits the MCP configuration; it does not install a machine-specific fff binary.

```sh
just swarm-setup
just test-swarm
opencode2 .
```

`swarm-setup` verifies the local plugin entrypoint. The plugin has no npm/runtime
package dependencies to install.

In OpenCode:

```text
/swarm <implementation objective>
```

`opencode.jsonc` selects `orchestrator` with OpenAI Sol 6.1. `/swarm` explicitly
selects Sol 6.1 high even in an existing session. Worker definitions select
OpenAI GPT-6 Luna high. The command requests up to three concurrent background
workers; this maximum is an instruction, not a runtime semaphore.

The default-agent setting applies to new sessions, not to an agent already stored
on an existing session. `/swarm` selects the orchestrator in the current session.
If a newly added plugin is not visible in an already running location, start a
new session in this checkout or reload that location's configuration.

## Messaging

After native background launches return session IDs, the principal discovers the
`swarm` namespace in Code Mode and registers the complete child list:

```js
await tools.swarm.register({ children: [childA, childB] });
```

A worker discovers peers and sends a mid-task question without waiting for its
own final response:

```js
await tools.swarm.members();
await tools.swarm.send({ to: "parent", message: "Which accepted contract covers this case?" });
await tools.swarm.send({ to: peerSessionID, message: "The assigned interface is ready." });
```

These examples are Code Mode tool calls, not slash commands. Use the exact names
shown by the tool catalog. `delivery: "steer"` is the default; `"queue"` is also
available. `swarm.broadcast({ message: "..." })` excludes the sender and reports
admission separately for each recipient.

Messages use OpenCode's durable synthetic inbox with `resume: true`: the runtime
schedules the recipient's agent loop. The mailbox returns on admission rather
than waiting for a response. Native background child completion is handled by
OpenCode; the plugin does not duplicate its completion notifications.

Only the principal and its registered direct children in the same project and
checkout can communicate. Register new children before using them and retain
existing IDs when updating the registry. Removed, moved, deleted or archived
sessions cannot be used as recipients. Send actionable updates; avoid automated
acknowledgements and broadcast echoes.

## Files and verification

- `opencode.jsonc`: default model, agent, KB skill directory and local plugin.
- `.opencode/agents/`: principal and bounded implementation workers.
- `.opencode/commands/swarm.md`: explicit swarm workflow.
- `.opencode/swarm-mailbox/`: dependency-free JavaScript plugin for the V2.0.1 API.
- `tests/swarm_mailbox_test.mjs`: transport and membership tests.
- [Accepted contract](knowledge/contracts/agent-swarm.md) and
  [sequence](knowledge/uml/agent-swarm.md): normative development-tooling behavior.

The worker edit-tool rules restrict modules; assignment prompts additionally
define exact file ownership. Shell remains permission-controlled and is not a
filesystem sandbox. Core and Qt builds do not depend on this JavaScript tooling.

`just test-swarm` verifies routing, family isolation, resume/delivery parameters,
registration persistence, cancellation and partial broadcast failure. A live
swarm smoke test must additionally observe native completion, principal wake-up
and idle sibling wake-up with the configured models. Automated transport tests
alone do not prove provider authentication or successful live model execution.

### Verified 2026-09-30

On OpenCode V2.0.1 the local plugin became active and its four tools were available
in the running session. Runtime discovery found all four agents and `/swarm`.
`just swarm-setup`, all 11 `just test-swarm` tests and `git diff --check` passed.

A read-only live smoke test used OpenAI GPT-6 Luna high native background children:

- Both initial child completions arrived automatically at the principal.
- A child sent `SWARM_A_MIDTASK_OK` to the principal before returning its result.
- Continuing that same child by session ID, it sent a message to an already
  completed, idle sibling. Admission succeeded; the sibling resumed and sent
  `SWARM_IDLE_WAKE_OK` to the principal through the mailbox.

This verifies sibling idle wake and live mid-task delivery. A separately idle
principal wake scenario was not exercised; its `resume: true` request is covered
by the transport tests and installed runtime API contract.
