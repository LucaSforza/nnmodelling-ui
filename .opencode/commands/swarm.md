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
implementation into bounded assignments with disjoint file ownership and at most
three concurrent Luna workers. Launch workers in background, register their native
sessionIDs with swarm.register, and use swarm.send for actionable coordination.
Process native completion notifications, review all results, and run relevant
justfile checks plus git diff --check before reporting completion.
