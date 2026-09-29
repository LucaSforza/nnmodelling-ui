---
name: use-nnmodelling-kb
description: Contract-first workflow for NNModelling native client architecture and implementation.
---

# Use native NNModelling knowledge base

Before any subagent starts work, that subagent must personally read `docs/knowledge/README.md` and every architecture, contract and UML document relevant to its assigned task. The principal must list those documents in the delegation prompt. Principal and non-subagent implementers follow the same read-first rule. Identify affected areas, then read the relevant architecture, contracts and UML completely. Inspect relevant implementation. Classify each design point as accepted, open, deferred/non-goal, or conflict. Legacy repository is evidence; new KB is normative.

Before changing architecture, ownership, public APIs, model/graph/stereotype semantics, file format, editor behavior, graphics abstraction, command behavior, lifecycle or observable behavior, principal model updates KB, UML and contracts first. Do not implement an open point as though accepted. If implementation exposes a contract problem, stop that direction and return it to principal for KB update before continuing.

Implementation tasks for GPT-6 Luna must cite exact relevant KB documents and UML, accepted contract, files/modules, constraints, acceptance criteria and tests. Luna implements only bounded tasks; it does not set architecture or normative text. Principal reviews implementation against KB, runs/inspects relevant tests, updates plan/status, and checks for stale contracts.

Current scope is client UI only. Backend and `nnmodelctl` diagrams are future boundaries, not implementation instructions. Preserve exact core package assets and historical UML. Run relevant tests and `git diff --check` before completion; report open questions and missing tooling accurately.
