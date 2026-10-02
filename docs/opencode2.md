# OpenCode V2 swarm

The repository uses `opencode2` (tested with V2.0.1 and V2.0.22), not the legacy `opencode`
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
OpenAI GPT-6 Luna high. There is no fixed project-level maximum for concurrent
background workers. The three worker definitions are roles; multiple sessions of
the same role may run with disjoint files. The principal sizes concurrency to
independent work, resources and actual runtime/provider limits.

The default-agent setting applies to new sessions, not to an agent already stored
on an existing session. `/swarm` selects the orchestrator in the current session.
If a newly added plugin is not visible in an already running location, start a
new session in this checkout or reload that location's configuration.

## Messaging

BEFORE launching any native background child, the principal discovers the `swarm`
namespace in Code Mode and successfully initializes communication:

```js
await tools.swarm.init({});
```

Only after successful initialization does the principal launch children and retain
their returned session IDs. If the tool is unavailable or initialization fails,
resolve/report that blocker before launching. Initialization is idempotent and
preserves membership/exclusions; an old array registry is migrated on `init`.

A direct child automatically enrolls on its first mailbox operation or when
addressed by session ID, after parent/project/checkout validation. It can send a
mid-task question immediately, without waiting for a later registration step:

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

Only the initialized principal and its eligible direct children in the same
project and checkout can communicate. Peer discovery is progressive: `members`
and broadcasts see enrolled children, not all native sessions. The principal can
address a newly returned session ID immediately. Registry updates are serialized
within the local plugin storage context, so concurrent enrollment cannot lose
members; this is not a cross-service distributed lock.

`swarm.register({children: [...]})` remains available for explicit principal-only
membership replacement. Omitted enrolled children are excluded and cannot silently
rejoin; explicitly registering them restores access. Do not call it routinely
with a stale launch list. `register({children: []})` does not initialize the swarm.
Moved, deleted or archived sessions cannot be used as recipients. A child encountering
missing initialization returns a blocked report rather than polling. Send actionable
updates; avoid automated acknowledgements and broadcast echoes.

## Waiting without busywork

When independent principal work is exhausted, suspend explicitly rather than
polling or filling time with unnecessary work:

```js
await tools.swarm.wait({ children: outstandingSessionIDs, timeout_seconds: 120 });
// If reason is timeout and a progress check would help:
await tools.swarm.status({ children: outstandingSessionIDs });
```

`timeout_seconds` is optional (default 270), integer-only, minimum 1 and maximum
270 seconds (4 minutes 30 seconds). The deadline includes preparation. `children`
is optional: omission watches enrolled children, `[]` waits for messages only.
Use outstanding IDs to avoid immediately returning for previously reviewed idle
children. At most one wait may be active per principal.

Wait returns `reason: "message" | "idle" | "timeout" | "unavailable"`, watched IDs
and elapsed milliseconds. It observes native child wait requests, incoming mailbox
native inbox-enqueued events and the latest persisted admission notification without
periodic polling. The authenticated event subscription starts before the notification
snapshot, closing the startup race even across plugin instances. An observed inbox ID
is saved separately, so that notification is not replayed. It may refer to an
already-delivered input and is not an unread-inbox claim.
The V2.0.22 plugin adapter does not emit `server.connected`; waiting must not
block on that marker. Its first actual event triggers one extra persisted-notification
reconciliation, without periodic polling.
Idle includes failed/interrupted loops; native completion reports remain authoritative.
Wait does not consume inbox messages or stop agents on cancellation. Missing native
wait/event API support is an explicit error, not a silent fallback to shell sleep.
Native event stream failure returns unavailable, not a successful completion.

`status` is a separate one-time diagnostic read after timeout, not part of the
270-second waiting deadline. It reports running/idle/unknown/unavailable, last
recorded outcome/timestamps and latest observable tool names/statuses. It omits
arguments, results, reasoning and full transcripts. OpenCode V2.0.22's plugin
context has no active-session listing method: activity is honestly `unknown` on
that runtime, while recorded outcomes/timestamps and tool names remain observable.
These are observations, not
proof of progress or success: timeout alone does not mean an agent is stuck. Decide
whether to clarify or wait again; do not poll status repeatedly.

## Files and verification

- `opencode.jsonc`: default model, agent, KB skill directory and local plugin.
- `.opencode/agents/`: principal and bounded implementation workers.
- `.opencode/commands/swarm.md`: explicit swarm workflow.
- `.opencode/swarm-mailbox/`: dependency-free JavaScript plugin for the V2 API.
- `tests/swarm_mailbox_test.mjs`, `tests/swarm_activity_test.mjs`: transport,
  membership, waiting and status tests.
- [Accepted contract](knowledge/contracts/agent-swarm.md) and
  [sequence](knowledge/uml/agent-swarm.md): normative development-tooling behavior.

The worker edit-tool rules restrict modules; assignment prompts additionally
define exact file ownership. Shell remains permission-controlled and is not a
filesystem sandbox. Core and Qt builds do not depend on this JavaScript tooling.

`just test-swarm` verifies routing, family isolation, resume/delivery parameters,
initialization, migration, automatic/concurrent enrollment, explicit exclusions,
more than three workers, persistence, cancellation and partial broadcast failure.
Activity tests cover wait deadlines, native idleness, admission/preparation races,
cancellation/cleanup and status authorization/privacy.
A live swarm smoke test must additionally observe native completion, principal wake-up
and idle sibling wake-up with the configured models. Automated transport tests
alone do not prove provider authentication or successful live model execution.

### Verified 2026-09-30

The observations below concern the original four-tool, explicit-registration
protocol. They do not verify the amended `init`/automatic-enrollment protocol live.

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

### Verified 2026-10-02

The amended five-tool plugin was available in the running session. The principal
successfully called `swarm.init` before launching two read-only native background
`test-worker` sessions; no `register` call was made during this live test.

- Both children enrolled automatically via `members` and sent mid-task findings
  to the principal successfully.
- The second child saw the first through progressive peer discovery and sent it
  an actionable routing check. The first had already completed; it resumed,
  observed both children in `members`, and sent the measured result to the principal.
- Native initial completion notifications arrived for both children. No polling,
  manual registration, completion forwarding or acknowledgement loops were used.
- `just swarm-setup`, all 20 `just test-swarm` tests, JavaScript syntax checks and
  `git diff --check` passed. Concurrent enrollment of eight workers is covered by
  automated transport tests, not by an eight-model live test.

This verifies the amended startup protocol, live parent/sibling routing and idle
sibling wake. A separately idle principal wake scenario remains untested live.

### Bounded waiting verified 2026-10-02 (V2.0.22)

The seven-tool plugin exposes `wait` and `status` in the running session. Live
verification caught two assumptions absent from mocks: child plugins need not
share JavaScript storage-object identity, and the plugin event adapter does not
emit the public client's `server.connected` marker. Regression tests now model
both conditions; wake routing uses native events and persisted notifications.

- A native child sent `LIVE_WAIT_NATIVE` during an active message-only wait.
  `wait` returned `reason: message` after 23797 ms, with the same sender and
  inbox ID reported by the child's actual send admission.
- A subsequent wait watching that outstanding child returned `reason: idle`
  after 3494 ms; its native completion notification arrived as well.
- An already-idle child returned immediately (9 ms). A message persisted before
  wait returned immediately (3 ms), without consuming native inbox input.
- After the message notification was observed, a message-only wait with a
  one-second deadline returned timeout at 1000 ms, rather than replaying it.
- `status` returned the recorded successful outcome, timestamps and latest
  completed tool name. Native active-session listing is unavailable in this
  plugin context, so activity was reported as unknown, not fabricated.
- `just swarm-setup`, all 46 `just test-swarm` tests, syntax checks and
  `git diff --check` passed. The 270-second deadline and invalid values are
  verified with deterministic clocks/schema tests, not a 4.5-minute live sleep.

Initial unsuccessful live probes are not counted as passing verification. They
led to the corrected event adapter and no-readiness-marker regression coverage.
