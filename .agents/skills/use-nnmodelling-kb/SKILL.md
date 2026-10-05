---
name: use-nnmodelling-kb
description: Contract-first NNModelling native client workflow for OpenCode and Codex.
---

# Use native NNModelling knowledge base

Before any subagent starts work, that subagent must personally read `docs/knowledge/README.md` and every architecture, contract and UML document relevant to its assigned task. The principal must list those documents in the delegation prompt. Principal and non-subagent implementers follow the same read-first rule. Identify affected areas, then read the relevant architecture, contracts and UML completely. Inspect relevant implementation. Classify each design point as accepted, open, deferred/non-goal, or conflict. Legacy repository is evidence; new KB is normative.

Before changing architecture, ownership, public APIs, model/graph/stereotype semantics, file format, editor behavior, graphics abstraction, command behavior, lifecycle or observable behavior, principal model updates KB, UML and contracts first. Do not implement an open point as though accepted. If implementation exposes a contract problem, stop that direction and return it to principal for KB update before continuing.

Implementation tasks for GPT-6 Luna must cite exact relevant KB documents and UML, accepted contract, files/modules, constraints, acceptance criteria and tests. Luna implements only bounded tasks; it does not set architecture or normative text. Principal reviews implementation against KB, runs/inspects relevant tests, updates plan/status, and checks for stale contracts.

## Choose the current agent runtime

The KB-first rules above apply to OpenCode and Codex. Before delegation, read
`docs/knowledge/contracts/agent-swarm.md` and `docs/knowledge/uml/agent-swarm.md`.
Delegate only when the user or applicable instructions request agent work.
Use the current runtime's available tools; missing OpenCode tools must not
block Codex work. Workers do not spawn agents. Give disjoint file ownership;
serialize Git staging/commits when workers share a checkout. Review every
required result and run relevant final checks.

### Codex

Use native `collaboration.spawn_agent`, `send_message`, `followup_task` and
`wait_agent`; no `swarm.init` or mailbox enrollment is needed. Bounded
implementation workers use `gpt-6-luna` with `high` reasoning. If model override
requires a limited history fork, include all exact KB documents, accepted
contract/plan, assigned files, constraints, acceptance criteria and tests in
the prompt. Respect the available concurrency limit. Use subagents for current
request subtasks; do not create sidebar chats as substitute workers. Report
missing requested model/tool instead of silently substituting another.

For UI tasks, discover actual computer-use surfaces before promising native
control. Browser access does not imply native desktop access. Preserve the
user's requested UI workflow; do not silently substitute `nnmodelctl` or direct
project-file generation for a computer-use trial. Report unavailable surfaces
and distinguish environment limitations from observed application defects.

### OpenCode V2 swarm initialization and communication

The principal must discover the `swarm`
tools in Code Mode and successfully complete `swarm.init({})` BEFORE launching
any subagent. If initialization fails or the tool is unavailable, resolve/report
the blocker before launching workers. `register({children: []})` is not a substitute.

After initialization, direct children in the same project and checkout automatically
enroll on first mailbox use or when addressed by session ID. The principal retains
native background session IDs; it need not register a complete list after launch.
`swarm.members` discovers currently enrolled peers progressively; `swarm.send`
accepts a session ID or `to: "parent"`. Use only actionable questions, blockers and
coordination, not acknowledgement-only replies or completion duplicates. Before
initialization, a worker returns a blocked report rather than polling or trying
to initialize itself. Optional `swarm.register` explicitly replaces membership
and excludes omitted enrolled children; do not use stale launch lists.

There is no fixed project-level concurrency maximum. The three worker roles are
not a limit on session count. Size concurrency to independent bounded assignments,
resources and runtime/provider limits, always with disjoint file ownership. Workers
must not spawn agents. Process native completion notifications without polling;
review all required results and run relevant checks before finishing.

When there is no useful independent work, the principal may call
`swarm.wait({children: outstandingSessionIDs, timeout_seconds: 270})` instead of
inventing busywork, polling or sleeping in the shell. The timeout is optional
(default 270 seconds), must be an integer from 1 to 270, and bounds preparation
and waiting. Wake reasons are incoming `message`, child `idle`, `timeout` or
`unavailable`; idleness is not necessarily successful completion. Omitting children
watches currently enrolled children; `children: []` waits only for messages.
Already-idle children wake immediately, so pass outstanding IDs rather than
previously reviewed children. After timeout, `swarm.status({children: outstandingSessionIDs})`
provides a one-time activity/outcome/tool-name snapshot, not reasoning or transcripts.
Timeout alone is not a stuck-agent diagnosis. Do not repeatedly poll status or
acknowledge wake events; process the actual native message/result and coordinate
only when useful. Wait cancellation stops observation, never the child agents.

The current release includes the accepted local FastAPI backend and container training in `docs/knowledge/contracts/backend.md` and `docs/knowledge/uml/backend.md`, alongside the C11 native client and local `nnmodelctl` in `docs/knowledge/contracts/automation.md`. Preserve exact core package assets and historical UML. Run relevant tests and `git diff --check` before completion; report open questions and missing tooling accurately.
