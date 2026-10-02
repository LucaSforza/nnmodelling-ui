# OpenCode V2 agent swarm

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
- `/swarm` explicitly requests delegation. The principal starts independent
  assignments with the native `subagent` tool in background only after completing
  `swarm.init`. It retains session IDs and processes native completion notifications.
- There is no fixed project-level maximum number of concurrent workers. The three
  worker definitions are roles, not a session limit; multiple instances of a role
  may run. The principal sizes concurrency to independent work, resources and
  actual runtime/provider limits. File ownership must be disjoint. Workers do not
  launch additional agents. The principal runs final checks.

## Mailbox

- The local `nnmodelling.swarm-mailbox` plugin exposes `swarm.register`,
  `swarm.init`, `swarm.members`, `swarm.send` and `swarm.broadcast` in Code Mode.
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

## Verification

Node tests under `tests/` exercise family checks, persisted membership, delivery
and wake flags, initialization, legacy migration, immediate child access,
concurrent enrollment, explicit exclusion/restoration, more than three workers,
and partial broadcast failure. Runtime discovery must confirm
the agent definitions, `/swarm` and the local plugin in the target location.
Live model verification additionally checks background completion, a mid-task
message to the principal, and a message waking an idle sibling. Those behaviors
must be reported as unverified until observed on the installed runtime.
