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

Use `fff` MCP tools for every file search. Preserve user changes. Do not expose
secrets. Keep implementation small and idiomatic C. Use `justfile` for build
recipes, put test sources under `tests/`, keep model/application code C11, and restrict C++/Qt to gui/qt. Run relevant tests and `git diff --check`; report missing
Qt/tooling instead of claiming success.
The current project is UI/client only; no backend or command service code is
authorized in bootstrap. Legacy files under `stereotype-packages/` and
`analysis/uml/nn.vpp` are preserved references, not code to port mechanically.
