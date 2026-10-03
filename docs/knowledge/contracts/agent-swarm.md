# OpenCode V2 agent swarm

## Runtime selection (accepted 2026-10-03)

The KB-first workflow and principal/implementation-worker ownership apply to
OpenCode and Codex. The mailbox protocol below belongs only to OpenCode V2;
it is not a prerequisite for Codex delegation. Select the current runtime's
available tools before launching workers, and never emulate a missing runtime.

In Codex, use native `collaboration` tools for subtasks of the current request:
`spawn_agent`, `send_message`, `followup_task` and `wait_agent`. No `swarm.init`
or OpenCode session enrollment is required. Use `gpt-6-luna` with `high`
reasoning for bounded implementation assignments; when a model override
requires a limited history fork, include all required KB documents, accepted
contract, file ownership, acceptance criteria and tests in the prompt.
Workers do not delegate. Respect the runtime's actual concurrency limit.
Codex sidebar chat creation is not subagent delegation.

Delegation must be requested by the user or applicable instructions. `/swarm`
requests parallel agent work using the current runtime's native mechanism.
Retain returned agent IDs, review required results and run final checks. When
workers are asked to commit, give disjoint file ownership and serialize Git
index/commit mutations in a shared checkout. Do not stage other agents' edits.
Missing requested models or tools are explicit blockers, not permission to
substitute models or invoke the other runtime silently.

Accepted 2026-09-30: repository-local OpenCode V2 tooling supports background
implementation agents and messages between the principal and its direct children.
This contract concerns development tooling, not the native client or its future
command service. See [sequence](../uml/agent-swarm.md).

Amended 2026-10-02: initialization precedes child launch, direct children enroll
automatically, and there is no fixed project-level worker concurrency maximum.

## Roles and scheduling

- `orchestrator` owns architecture, accepted contracts, KB/UML, planning and
  implementation review. Its configured model is `openai/gpt-6.1-sol#high`.
- `core-worker`, `qt-worker` and `test-worker` use `openai/gpt-6-luna#high`
  and implement bounded assignments. Each personally reads the KB index and
  the exact documents listed in the assignment. Architectural ambiguity returns
  to the principal.
- Accepted 2026-10-02: `model-builder` uses `openai/gpt-6.1-sol#high` and
  `mode: all`: it is visible as a selectable primary agent and callable by the
  orchestrator as a child. It authors model graphs/resources through `nnmodelctl`
  under the existing automation contract, not through source or project-file
  edits. Its description and instructions are English. It cannot launch agents
  or decide new architecture/contracts; ambiguous semantics return to the
  orchestrator (or the user when selected directly). The default primary remains
  `orchestrator`. In OpenCode V2 a primary session's selected model remains
  separate from the agent's configured model; child sessions use the configured
  Sol high model.
- `/swarm` explicitly requests delegation. The principal starts independent
  assignments with the native `subagent` tool in background only after completing
  `swarm.init`. It retains session IDs and processes native completion notifications.
- There is no fixed project-level maximum number of concurrent workers. The three
  worker definitions are roles, not a session limit; multiple instances of a role
  may run. The principal sizes concurrency to independent work, resources and
  actual runtime/provider limits. File ownership must be disjoint. Workers do not
  launch additional agents. The principal runs final checks.

## Model-authoring assignments

`model-builder` personally reads the project skill, KB index, architecture,
automation/resource/model/typed-output/diagnostic contracts and their relevant
UML before commands. Lua resource authoring additionally requires the Lua
contract. It first checks CLI availability and inspects the active project.
The GUI must already expose a private local socket; starting/building the GUI
requires shell approval rather than silent fallback to another project.

An assignment identifies socket, project/destination, intended graph/resources,
allowed mutations and acceptance criteria. Exclusive mutation ownership covers
the entire application behind a socket and its writable project/resources,
not just separate node IDs or files. Do not run two mutating authors against the
same application or writable project; UI edits are also shared-state mutations.
Parallel work is allowed on independent applications and project directories.
Child launches follow the same initialization/enrollment protocol as workers.

All graph/resource changes use the existing CLI operations. Core packages,
source, KB and model/resource files are not edited directly. Shell is ask-by-
default with narrow allowances for `nnmodelctl` and the local Qt-build CLI;
edit tools and further delegation are denied. These are tool permissions,
not a filesystem sandbox or payload-level authorization. The agent must never
discard existing changes without explicit authorization and must account for
resource creation saving all current graph edits. Invalid commands are atomic,
but a sequence of successful commands is not a single transaction; report
partial progress instead of retrying blindly or inventing rollback.

Completion reports include project path, graph/resources authored, actual
diagnostic availability/completeness/problems, successful save status and any
requested screenshot path. Incomplete graphs are allowed but must be labelled;
no backend, numerical execution or training claim is permitted. CLI protocol
and model semantics remain unchanged (see automation.md and typed-outputs.md).

## Mailbox

- The local `nnmodelling.swarm-mailbox` plugin exposes `swarm.register`,
  `swarm.init`, `swarm.members`, `swarm.send`, `swarm.broadcast`, `swarm.wait`
  and `swarm.status` in Code Mode.
- The principal must discover the tools and successfully call `swarm.init({})`
  before launching any child. Initialization is principal-only and idempotent;
  it preserves members and explicit exclusions. No other operation initializes
  implicitly. Calling `register({children: []})` is not an initialization step.
- The initialized registry is persisted in OpenCode plugin storage. Identities
  are session IDs, never model or agent names. A legacy array registry requires
  explicit `init`, which migrates it while preserving its members. Malformed
  persisted state is rejected rather than silently reset.
- A child belongs to its immediate parent. In an initialized swarm, a direct
  child enrolls automatically on its first mailbox operation or when addressed
  by session ID. It can communicate with the principal immediately, without a
  post-launch registration barrier. Sessions must belong to the same project
  and checkout directory and be unarchived. Nested swarms and cross-worktree
  messaging are deferred. Before initialization, a child receives an actionable
  error and returns a blocked report rather than polling or initializing itself.
- `members` lists enrolled members, not all native children. Peer discovery is
  progressive during launch; use known session IDs when needed. Broadcast targets
  a snapshot of enrolled members, excluding the sender.
- `parent` is an alias for the child's immediate parent. Self-send is rejected.
- `register` is optional principal-only explicit membership management, not the
  normal launch workflow. It validates all supplied children before replacing
  membership. Previously enrolled children omitted from the list are persistently
  excluded from automatic enrollment; explicitly registering them again restores
  access. Initialization never clears exclusions. Do not submit a stale complete
  list after launching children; that intentionally removes omitted members.
- Initialization, automatic enrollment and explicit replacement serialize
  read/modify/write operations per principal in the local plugin storage context.
  Concurrent automatic enrollments must not overwrite one another. This is not
  a distributed lock across separate OpenCode services sharing storage.
- Messages are synthetic inbox input, attributed to the sender, with
  `resume: true`. Delivery defaults to `steer`; callers may choose `queue`.
  This requests agent-loop scheduling even for an idle recipient. Sending returns
  after admission, not after the recipient's answer; it does not wait or poll.
- Broadcast excludes the sender and returns a per-recipient success/error report.
  Partial admission is reported, never represented as all-or-nothing delivery.
- Admission is not proof of execution or exactly-once delivery. No automatic
  retry, echo, completion forwarding or acknowledgement is added. Native child
  completion remains OpenCode's responsibility. Agents send only actionable
  updates/questions and never reply solely to acknowledge a message.
- The transport validates bounded nonempty messages and uses the executing tool's
  session ID as sender. It neither stores credentials nor changes permissions.

## Bounded event-driven waiting

Accepted 2026-10-02: the principal may explicitly suspend with `swarm.wait` when
there is no independent useful work. Waiting is not shell sleep or status polling.

- `wait({timeout_seconds, children})` is principal-only and requires initialization.
  `timeout_seconds` is an integer from 1 through 270 (4 minutes 30 seconds), default
  270. Invalid values are rejected, not clamped. The deadline covers preparation
  as well as observation. Only one wait may be active per principal.
- Optional `children` contains native direct-child session IDs to watch. Addressing
  them validates/enrolls them without replacement registration. Omission snapshots
  currently enrolled children; an empty array waits for messages only. Pass the
  outstanding IDs explicitly to avoid repeatedly waking on previously reviewed
  idle children. Newly enrolled children outside the snapshot are not watched for
  idleness, but their incoming messages can wake the principal.
- The result's `reason` is `message`, `idle`, `timeout` or `unavailable`. It includes
  elapsed milliseconds and watched IDs; message results identify sender/inbox ID,
  idle results identify the child, and unavailable results identify the failed
  native observation. Native `session.wait` observes an agent loop becoming idle,
  not proof of successful implementation. Already-idle children return immediately.
- Native `session.inbox.enqueued` events wake the principal for eligible swarm
  synthetic messages. Event iteration starts before reading a persisted
  one-shot notification of the latest admitted message, closing the before-wait
  race across plugin instances. An observed inbox ID is persisted separately to
  avoid replaying that notification. The notification may refer to an input
  already delivered; it is not an unread-inbox claim. Wait marks only its persisted
  wake notification as observed, never consumes durable inbox input. Notification/observed-ID records
  survive plugin unload; native inbox delivery and completion remain authoritative.
  Wait observes without consuming messages, sending acknowledgements,
  duplicating completion reports or changing delivery/resume semantics. Observation
  uses the authenticated native event subscription and validates sender family and
  checkout rather than trusting metadata alone. Native events are live-only: stream
  failure/end yields `unavailable`, never a false successful completion. Notification
  persistence failure is reported separately from successful message admission.
  Unlike the public client, the V2.0.22 plugin event adapter does not emit a
  `server.connected` readiness marker: preparation/native child waits must never
  depend on that marker. The first observed native event triggers one additional
  notification reconciliation, not a periodic status/inbox polling loop.
- Wait races native child wait requests, incoming mailbox events and one deadline
  timer. It does not periodically inspect sessions. Cancellation rejects the tool;
  every exit removes listeners/timers and aborts outstanding observation requests,
  never the child agents. Missing native wait/event support fails explicitly. The installed
  V2.0.22 plugin session context exposes neither inbox listing nor active-session
  listing, although those routes exist on its HTTP API; do not invent plugin methods.
- After timeout, the principal may call `status({children})` once to inspect work and
  decide whether to clarify, resume or wait again. It uses the same child selection
  rules and makes per-child metadata/context reads, plus one native active-session
  snapshot only when the plugin context exposes it. It reports
  running/idle/unknown/unavailable, last recorded outcome and
  timestamps, and tool names/statuses from the latest tool-bearing assistant entry
  (even if a later final report contains only text). Inactivity alone is not
  success. No tool arguments/results, reasoning or full transcripts are returned.
  Individual context failures are reported; absent/failed active-session listing yields unknown
  activity rather than fabricated idleness. This diagnostic call is separate from
  the 270-second wait deadline, and timeout alone is not proof an agent is stuck.
- Principal instructions and the project skill must permit bounded `swarm.wait`
  instead of inventing busywork. Required child results still must be reviewed
  before completion; native notifications remain authoritative and require no polling.

## Verification

Node tests under `tests/` exercise family checks, persisted membership, delivery
and wake flags, initialization, legacy migration, immediate child access,
concurrent enrollment, explicit exclusion/restoration, more than three workers,
partial broadcast failure, message/idle/deadline waiting, cancellation/cleanup,
admission/preparation races, waiting authorization and status privacy. Runtime discovery must confirm
the agent definitions, `/swarm` and the local plugin in the target location.
Live model verification additionally checks background completion, a mid-task
message to the principal, and a message waking an idle sibling. Those behaviors
must be reported as unverified until observed on the installed runtime.
