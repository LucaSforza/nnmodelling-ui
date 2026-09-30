# NNModelling native agent guidance

For all architecture, implementation, refactoring, review, planning and testing,
read and follow `.agents/skills/use-nnmodelling-kb/SKILL.md` first. The
`docs/knowledge/` tree is authoritative for this native client. Keep it
coherent with implementation; update contracts/UML before semantic changes.

The principal reasoning model owns architecture, planning, contracts, KB,
UML and implementation review. GPT-6 Luna subagents are implementation-only:
they may write narrowly specified C core, Qt frontend, build and test changes after the principal
provides accepted contract and plan. They must return architectural ambiguity
to the principal rather than choose a new contract.

OpenCode V2 local swarm setup is in `docs/opencode2.md`; its normative tooling
contract and sequence are `docs/knowledge/contracts/agent-swarm.md` and
`docs/knowledge/uml/agent-swarm.md`. `/swarm` explicitly requests delegation.
Use native background child sessions and registered `swarm` mailbox tools for
actionable coordination, with at most three concurrent workers and disjoint
file ownership. Native completion notifications require no polling.

Use `fff` MCP tools for every file search. Preserve user changes. Do not expose
secrets. Keep implementation small and idiomatic C. Use `justfile` for build
recipes, put test sources under `tests/`, keep model/application code C11, and restrict C++/Qt to gui/qt. Run relevant tests and `git diff --check`; report missing
Qt/tooling instead of claiming success.
The current project is UI/client only; backend and training remain deferred.
Local command service is accepted by `docs/knowledge/contracts/automation.md`.
Legacy files under `stereotype-packages/` and
`analysis/uml/nn.vpp` are preserved references, not code to port mechanically.
