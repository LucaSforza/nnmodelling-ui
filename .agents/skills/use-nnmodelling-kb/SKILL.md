---
name: use-nnmodelling-kb
description: Contract-first workflow for NNModelling native client architecture and implementation.
---

# Use native NNModelling knowledge base

Before any subagent starts work, that subagent must personally read `docs/knowledge/README.md` and every architecture, contract and UML document relevant to its assigned task. The principal must list those documents in the delegation prompt. Principal and non-subagent implementers follow the same read-first rule. Identify affected areas, then read the relevant architecture, contracts and UML completely. Inspect relevant implementation. Classify each design point as accepted, open, deferred/non-goal, or conflict. Legacy repository is evidence; new KB is normative.

Before changing architecture, ownership, public APIs, model/graph/stereotype semantics, file format, editor behavior, graphics abstraction, command behavior, lifecycle or observable behavior, principal model updates KB, UML and contracts first. Do not implement an open point as though accepted. If implementation exposes a contract problem, stop that direction and return it to principal for KB update before continuing.

Implementation tasks for GPT-6 Luna must cite exact relevant KB documents and UML, accepted contract, files/modules, constraints, acceptance criteria and tests. Luna implements only bounded tasks; it does not set architecture or normative text. Principal reviews implementation against KB, runs/inspects relevant tests, updates plan/status, and checks for stale contracts.

## Swarm initialization and communication

For swarm/parallel work, personally read `docs/knowledge/contracts/agent-swarm.md`
and `docs/knowledge/uml/agent-swarm.md`. The principal must discover the `swarm`
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

Current scope is client UI only. Local `nnmodelctl` is accepted by `docs/knowledge/contracts/automation.md`; backend and training remain future boundaries. Preserve exact core package assets and historical UML. Run relevant tests and `git diff --check` before completion; report open questions and missing tooling accurately.
