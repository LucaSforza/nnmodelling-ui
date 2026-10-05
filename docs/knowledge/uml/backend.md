# Backend execution and snapshot ownership

Accepted 2026-10-05. Normative protocol: [backend](../contracts/backend.md).

```mermaid
flowchart TD
  UI[Qt backend panel] -->|async HTTP v1| API[FastAPI local service]
  UI -->|save/open| C[C application and project authority]
  API --> Store[Persistent jobs and immutable snapshots]
  API --> Runner[Container queue and lifecycle]
  Runner --> Worker[Isolated Python training worker]
  Store -->|read-only snapshot| Worker
  Worker --> Graph[Generic PyTorch DAG module]
  Worker --> Adapter[DatasetAdapter InputT OutputT]
  Worker --> Artifacts[Metrics safetensors standalone wheel]
  Artifacts --> Store
  Store -->|snapshot/downloads| UI
  Wheel[Exported Model] -->|tokenize| Adapter
  Wheel -->|prediction closure| Graph
  Graph -->|untokenize| Adapter
```

```mermaid
sequenceDiagram
  actor User
  participant UI as Qt
  participant C as C Application
  participant API as FastAPI
  participant Store as Job store
  participant Container as Training container
  User->>UI: Submit current model
  UI->>C: Save current edits
  C-->>UI: Exact saved model and resource paths
  UI->>API: POST /v1/jobs project/files/training
  API->>Store: Freeze validated bundle and core packages
  API-->>UI: Queued job id
  API->>Container: Launch with snapshot and output mounts
  Container->>Store: Epoch train/validation losses
  Container->>Store: Test loss, weights, wheel
  UI->>API: GET job (bounded polling)
  API-->>UI: Persisted state and metrics
  User->>UI: Restore old job
  UI->>API: GET snapshot
  API-->>UI: Exact original model and files
  UI->>C: Open newly restored project (confirm unsaved edits)
```

DatasetAdapter is abstract generic over distinct input/output types. GraphModule
owns registered stereotype module instances and immutable graph schedule; service
owns no active UI graph. Exported Model owns adapter, graph and safetensors state.
