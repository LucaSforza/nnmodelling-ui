# Legacy-to-native mapping

Reference commit: `2828ac31bc0e2de04f258afd88461db425de040a` of
[LucaSforza/NNModelling](https://github.com/LucaSforza/NNModelling).
Classification concerns this client rewrite, not deletion from legacy.
This table records the original rewrite classification. Accepted 2026-10-05
[backend contract](../contracts/backend.md) supersedes its backend deferrals:
new local FastAPI/container execution uses preserved Python entrypoints without
porting the historical backend implementation mechanically.

| Legacy concept | Class | Reason and native direction |
| --- | --- | --- |
| `stereotype-packages/core` manifests, definitions, Lua and Python resources | PRESERVE | Exact bytes and package identities copied. Python retained as package data for future backend export, never run by client. |
| Package IDs/versions, dependencies, model-owned scope | PRESERVE SEMANTICS, REIMPLEMENT MECHANISM | Catalog activation moves from Cordis/browser to native client, exact identity remains. |
| Package-owned Lua type inference, structured semantic errors | PRESERVE SEMANTICS, REIMPLEMENT MECHANISM | Embedded Lua replaces browser runtime; C host API must match accepted tensor semantics. |
| `DiagramCore` single graph authority and graph mutations | ADAPT | Authority moves to application-owned typed model; no UI/backend duplicate. |
| Join input order by `targetHandle`, same-scope edges, DAG validation | PRESERVE | Domain rules independent of browser. |
| Collapsed subflow children remain semantic; route points are presentation | PRESERVE | Display state does not change graph meaning. |
| Model manifest v2 custom package/dataset references | ADAPT | Preserve exact, project-relative identity and transactional loading; dataset/backend details deferred. |
| Svelte components, Svelte Flow, browser DOM and IndexedDB | REPLACE | Qt Widgets editor, C state, native project persistence. |
| Cordis Fiber lifecycle and browser `TypeSystemHost` | REPLACE | Native catalog and Lua state ownership. |
| Browser RPC, WebSocket MCP transport | REPLACE | Future local command adapter; no MCP server in client. |
| MCP modeling and screenshot use cases | ADAPT | Future application operations/semantic UI IDs; training use cases depend on backend and are deferred. |
| Visual Paradigm `nn.vpp` and generated PlantUML | REFERENCE ONLY | Preserved; transcription and reconciled model live in KB. |
| NNTree, Hydra, PyTorch training/worker, FastAPI, Valkey | OUT OF SCOPE | Existing backend domain, not native UI implementation. |
| Package/backend wire format and authentication for future client | OPEN | Backend connection design needs explicit future contract. |

Legacy source separates domain semantics from mechanisms: the active graph and
type rules are client domain/application state; Svelte layout is UI behavior;
MCP/WebSocket is transport; Cordis/DOM are browser mechanisms; conversion,
training and worker jobs are backend behavior. Historical NNTree artifacts are
reference only and do not become editable native model state.

Evidence: legacy `docs/knowledge/architecture/overview.md`,
`docs/knowledge/contracts/package-type-system.md`,
`docs/knowledge/decisions/model-scoped-stereotype-packages.md`,
`docs/knowledge/decisions/editable-edge-routing.md`,
`docs/knowledge/uml/mcp-use-case-parity.md`, and `front-end/src/core/validation.ts`.
