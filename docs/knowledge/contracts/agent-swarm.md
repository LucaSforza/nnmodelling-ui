# OpenCode V2 agent swarm

Accepted 2026-09-30: repository-local OpenCode V2 tooling supports background
implementation agents and messages between the principal and its direct children.
This contract concerns development tooling, not the native client or its future
command service. See [sequence](../uml/agent-swarm.md).

## Roles and scheduling

- `orchestrator` owns architecture, accepted contracts, KB/UML, planning and
  implementation review. Its configured model is `openai/gpt-6.1-sol#high`.
- `core-worker`, `qt-worker` and `test-worker` use `openai/gpt-6-luna#high`
  and implement bounded assignments. Each personally reads the KB index and
  the exact documents listed in the assignment. Architectural ambiguity returns
  to the principal.
- `/swarm` explicitly requests delegation. The principal starts independent
  assignments with the native `subagent` tool in background, retains session IDs,
  registers them with the mailbox, and processes native completion notifications.
- At most three workers run concurrently by orchestration instruction; this is
  not a server-enforced concurrency limit. File ownership must be disjoint.
  Workers do not launch additional agents. The principal runs final checks.

## Mailbox

- The local `nnmodelling.swarm-mailbox` plugin exposes `swarm.register`,
  `swarm.members`, `swarm.send` and `swarm.broadcast` in Code Mode.
- The principal registers native direct children; identities are session IDs,
  never model or agent names. The registry is persisted in OpenCode plugin storage.
- A child belongs to its immediate parent. Only registered direct children and
  that parent may exchange messages. Sessions must belong to the same project
  and checkout directory. Nested swarms and cross-worktree messaging are deferred.
- `parent` is an alias for the child's immediate parent. Self-send is rejected.
  Registration validates all children before replacing the persisted registry.
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
and wake flags, and partial broadcast failure. Runtime discovery must confirm
the agent definitions, `/swarm` and the local plugin in the target location.
Live model verification additionally checks background completion, a mid-task
message to the principal, and a message waking an idle sibling. Those behaviors
must be reported as unverified until observed on the installed runtime.
