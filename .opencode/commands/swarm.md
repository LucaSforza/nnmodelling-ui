---
description: Implement an objective with background Luna workers and inter-agent messages
agent: orchestrator
model: openai/gpt-6.1-sol#high
subagent: false
---

Use the repository's OpenCode V2 swarm to complete this objective: $ARGUMENTS

This is an explicit request for delegation and parallel work. Personally read the
KB skill, KB index, swarm contract and swarm UML, then the documents relevant to
the objective. Accept/update contracts before implementation. Split independent
implementation into bounded assignments with disjoint file ownership. There is no
fixed project-level concurrency maximum; choose the worker count by independent
work, resources and runtime/provider limits, not the number of worker roles.
Discover swarm tools and successfully complete swarm.init({}) BEFORE launching
any worker; resolve/report initialization failure before launching. Then launch
workers in background and retain their native sessionIDs. Children automatically
enroll on mailbox use or when addressed; no post-launch swarm.register is needed.
Use swarm.send for actionable coordination and swarm.members for progressive peer
discovery. Optional register explicitly replaces membership and excludes omitted
enrolled children; do not submit stale launch lists.
Process native completion notifications, review all results, and run relevant
justfile checks plus git diff --check before reporting completion.
