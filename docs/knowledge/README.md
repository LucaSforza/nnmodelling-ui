# NNModelling native knowledge base

This tree is normative for the native client. Legacy NNModelling is evidence;
its browser and backend mechanisms do not define this implementation. An `OPEN`
item is a question, not an implementation license. Current scope is UI/client
only. Local command automation is accepted for this milestone; backend connection
and numerical training/execution remain deferred boundaries.

Read [architecture](architecture/overview.md), then relevant contracts and UML
before changing code. Update contract and UML before changing accepted
semantics or ownership. [UI rewrite plan](../plans/ui-rewrite.md) tracks work;
[bootstrap plan](../plans/bootstrap.md) records completed groundwork.

| Area | Documents |
| --- | --- |
| Architecture | [Overview and module ownership](architecture/overview.md) |
| Contracts | [UI release](contracts/ui-release.md), [model and packages](contracts/model.md), [graphics and editor](contracts/editor.md), [Lua runtime](contracts/lua.md), [future interfaces](contracts/future-interfaces.md) |
| Decisions | [Native client and Lua](decisions/native-client.md) |
| UML | [Legacy reference](uml/legacy.md), [metamodel](uml/metamodel.md), [project/resources](uml/project.md), [graphics](uml/graphics.md), [editor](uml/editor.md), [automation](uml/automation.md), [sequences](uml/sequences.md) |
| Reference | [Legacy mapping](reference/legacy-mapping.md) |
| Verification | [Testing strategy](testing/strategy.md), [release QA](testing/qa.md) |
| Development tooling | [OpenCode V2 swarm contract](contracts/agent-swarm.md), [swarm sequence](uml/agent-swarm.md) |
| Resources, VAE and LLM commands | [Authoring and subflows](contracts/resource-authoring.md), [local CLI protocol](contracts/automation.md), [command use cases](uml/automation.md) |

Current explicit user requirements take precedence; update accepted contracts
and UML before implementing a changed requirement. Otherwise accepted current
contracts and UML take precedence over legacy documentation and implementation.
Conflicts require a documented decision. Current release builds full native
project and graph editor, including project-owned datasets and stereotype
dependencies. Training and backend execution remain deferred. Resource authoring
and local `nnmodelctl` are accepted in [resource authoring](contracts/resource-authoring.md)
and [automation](contracts/automation.md).
