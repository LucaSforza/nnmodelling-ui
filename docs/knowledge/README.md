# NNModelling native knowledge base

This tree is normative for the native client. Legacy NNModelling is evidence;
its browser and backend mechanisms do not define this implementation. An `OPEN`
item is a question, not an implementation license. Current scope is UI/client
only; command automation and backend connection are designed boundaries but
not bootstrap deliverables.

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

Current explicit user requirements take precedence; update accepted contracts
and UML before implementing a changed requirement. Otherwise accepted current
contracts and UML take precedence over legacy documentation and implementation.
Conflicts require a documented decision. Current release builds full native
project and graph editor, including project-owned datasets and stereotype
dependencies. Training, backend execution and `nnmodelctl` remain deferred.
