# NNModelling native knowledge base

This tree is normative for the native client. Legacy NNModelling is evidence;
its browser and backend mechanisms do not define this implementation. An `OPEN`
item is a question, not an implementation license. Accepted scope now includes local FastAPI backend and container training per
[backend contract](contracts/backend.md), superseding earlier deferrals. Local
command automation and C11 native client remain accepted.

Read [architecture](architecture/overview.md), then relevant contracts and UML
before changing code. Update contract and UML before changing accepted
semantics or ownership. [UI rewrite plan](../plans/ui-rewrite.md) tracks work;
[bootstrap plan](../plans/bootstrap.md) records completed groundwork.

| Area | Documents |
| --- | --- |
| Backend and Python SDK | [Local API, containers, adapters and wheel](contracts/backend.md), [execution UML](uml/backend.md) |
| Architecture | [Overview and module ownership](architecture/overview.md) |
| Source organization | [Module layout and private helpers](contracts/source-layout.md), [architecture diagram](architecture/overview.md) |
| Contracts | [UI release](contracts/ui-release.md), [model and packages](contracts/model.md), [graphics and editor](contracts/editor.md), [Lua runtime](contracts/lua.md), [future interfaces](contracts/future-interfaces.md) |
| Decisions | [Native client and Lua](decisions/native-client.md) |
| UML | [Legacy reference](uml/legacy.md), [metamodel and typed outputs](uml/metamodel.md), [project/resources](uml/project.md), [editor/graphics and analysis ownership](uml/editor.md), [automation](uml/automation.md), [sequences](uml/sequences.md) |
| Reference | [Legacy mapping](reference/legacy-mapping.md) |
| Verification | [Testing strategy](testing/strategy.md), [release QA](testing/qa.md), [computer-use LLM trial](testing/ui-llm-authoring-2026-10-03.md), [training dashboard and examples](testing/training-dashboard-progress.md), [full LLM training and quality](testing/full-llm-training.md) |
| Development tooling | [OpenCode/Codex agent contract](contracts/agent-swarm.md), [agent sequence](uml/agent-swarm.md) |
| Resources, VAE and LLM commands | [Authoring and subflows](contracts/resource-authoring.md), [local CLI protocol](contracts/automation.md), [command use cases](uml/automation.md) |
| Expanded 3D explorer | [Composition, occurrences and camera](contracts/visualization-3d.md), [ownership UML](uml/editor.md) |
| Typed outputs and terminals | [Output/loss contract](contracts/typed-outputs.md), [boundary and output UML](uml/metamodel.md), [implementation plan](../plans/typed-outputs.md) |
| Errors and strings | [Diagnostics and common helpers](contracts/diagnostics.md), [report ownership](uml/editor.md), [UI sequence](uml/sequences.md#analysis-cache-and-problem-navigation) |

Current explicit user requirements take precedence; update accepted contracts
and UML before implementing a changed requirement. Otherwise accepted current
contracts and UML take precedence over legacy documentation and implementation.
Conflicts require a documented decision. Current release builds full native
project and graph editor, including project-owned datasets and stereotype
dependencies. Local backend and training are accepted per contracts/backend.md. Resource authoring
and local `nnmodelctl` are accepted in [resource authoring](contracts/resource-authoring.md)
and [automation](contracts/automation.md).

UML consolidation (2026-10-02): current graphics/report ownership is in editor.md,
typed outputs/results in metamodel.md, and mutation/analysis/navigation sequences
in the [sequence catalog](uml/sequences/README.md), with sequences.md retained as
a compatible index. Accepted 2026-10-05: sequence files group actual lifecycle,
editor/analysis, resource/IPC, backend/runtime, 3D and development-agent flows
by subsystem boundary and name their source owners. Redundant graphics.md,
diagnostics.md and typed-outputs.md UML
were merged rather than kept as parallel versions. Legacy UML stays explicitly
historical; automation and development swarm describe different protocols and
are not duplicates. Extend these current diagrams rather than creating a new
document for each incremental feature; the sequence catalog is the shared home
for runtime interactions.
