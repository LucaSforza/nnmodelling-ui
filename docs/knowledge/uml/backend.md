# Backend execution and snapshot ownership

Accepted 2026-10-05. Normative protocol: [backend](../contracts/backend.md).
The diagrams below show component ownership; request and runtime sequences are
kept in the [backend sequence](sequences/backend-training.md).

```mermaid
  flowchart TD
    UI[Qt backend panel] -->|async HTTP v1| API[FastAPI local service]
    UI -->|explicit config and process lifecycle| Launcher[Local backend launcher]
    Launcher -->|spawn with validated env| API
    UI -->|save/open| C[C application and project authority]
    API --> Store[Persistent jobs and immutable snapshots]
    API --> Runner[Local queue and Docker or SSH Slurm lifecycle]
    Runner --> Scheduler[SSH Slurm scheduler]
    Scheduler --> SIF[Contained Singularity worker]
    Runner --> Docker[Local Docker runtime]
    Docker --> Worker[Isolated Python training worker]
    SIF --> Worker
    Store -->|read-only snapshot| Runner
    Runner -->|mount snapshot and output| Worker
    Worker --> Graph[Generic PyTorch DAG module]
    Worker --> Adapter[DatasetAdapter InputT OutputT]
    Worker --> Artifacts[Metrics safetensors standalone wheel]
    Artifacts --> Store
    Store --> API
    API -->|snapshot/downloads| UI
    Wheel[Exported Model] -->|tokenize| Adapter
    Wheel -->|prediction closure| Graph
    Graph -->|untokenize| Adapter
```

`GraphModule` owns registered runtime modules and derives its topological
execution order from frozen graph metadata. The worker owns optimizer, random seeds and metrics. `JobStore` owns
persisted job state, immutable request bytes and published output files. The
Qt panel owns transient controls and asynchronous replies; the main window owns
the optional local backend child process, and per-user Qt settings own its
non-secret launch profile. C still owns the active native project. Exported
`Model` owns a private dataset adapter, graph runtime and weights, without
training payloads.

Wheel export separates readable distribution identity (`nnm_<normalized-project-id>`)
from the existing job-specific Python import package (`nnmodel_<normalized-job-id>`).
The API serves the stored artifact filename; old wheels remain unchanged.

Slurm execution keeps Store/API local. UUID job directories receive frozen snapshots
and worker source; per-allocation scheduler IDs are persisted for cancellation.
Warning signals checkpoint model, Adam, RNG, cursor and metric-window state as
safetensors plus JSON in remote output. Runner retrieves and validates handoff,
then submits another allocation against same logical job and remote directory.
Only final weights and wheel are served as artifacts after test evaluation.
Metrics and outputs return through bounded SSH transfers and existing local result
validation.
