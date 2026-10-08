# Backend jobs and training dashboard

The service freezes an API request into a job directory before execution. The
container reads the snapshot and writes only the mounted output; the service
reads and validates those files before marking a job complete. Sources include
`src/gui/qt/BackendDialog.cpp`, `backend/app.py`, `backend/validation.py`,
`backend/store.py`, `backend/runner.py`, `backend/worker.py` and
`python/nnmodelling-runtime/`.

## Save and submit an immutable snapshot

```mermaid
  sequenceDiagram
    actor User
    participant UI as Qt Training dashboard
    participant C as NNApplication and NNProject
    participant API as FastAPI
    participant Validate as Metadata validator
    participant Store as JobStore
    participant Runner as JobRunner queue
    User->>UI: submit settings and publish cadence
    UI->>C: save current project
    alt save fails
      C-->>UI: visible save error, no HTTP submission
    else save succeeds
      C-->>UI: exact saved model and project directory
      UI->>UI: collect manifest resources and inference assets
      UI->>API: POST project files and training configuration
      API->>API: check auth, payload bounds and training fields
      API->>Validate: validate schema, identities, graph and package metadata
      Note over Validate: Uploaded Python is not executed during validation
      alt metadata and payload accepted
        Validate-->>API: validated project and resolved core packages
        API->>Store: persist model, files, core bytes and config
        Store->>Store: write immutable snapshot and queued job record
        Store-->>API: queued job
        API->>Runner: enqueue job ID
        API-->>UI: queued job ID and initial metrics
      else invalid payload or storage failure
        API-->>UI: HTTP error, no queued training job
      end
    end
```

The snapshot includes selected dataset resources and submitted data. It is
read-only to the worker. Later UI edits cannot change a queued or running job.
The resolved core directory is frozen under `snapshot/core/<package-folder>`.

## Configure and launch the local service

```mermaid
sequenceDiagram
    actor User
    participant Dashboard as Qt Training dashboard
    participant Config as Separate modeless backend configuration window
    participant Settings as Per-user Qt settings
    participant Process as Local service child process
    participant API as Local FastAPI
    User->>Dashboard: open backend configuration
    Dashboard->>Config: show separate movable window
    Note over Dashboard,Config: Dashboard remains usable with endpoint, jobs and curves visible
    User->>Config: choose Docker or Slurm and enter service paths
    Config->>Settings: save non-secret launch profile
    User->>Config: start or restart managed backend
    Config->>API: inspect jobs if existing managed process is running
    alt queued/running jobs exist
        Config-->>User: refuse restart so jobs are not interrupted
    else safe to launch
        Config->>Config: reserve available loopback port
        Config->>Process: spawn uv/uvicorn on loopback with validated environment
        Process->>API: start FastAPI and select executor
        Config->>Dashboard: set Endpoint to managed loopback address
        Config->>API: GET /health
        API-->>Config: executor/runtime availability
        Config-->>User: show readiness or startup error
    end
```

The child is a separate local Python process, not embedded in Qt. Closing the
Training dialog leaves it running. Application shutdown warns that active jobs
will be interrupted; user must explicitly confirm stopping it. An external
backend endpoint is never managed by this launcher.

## Queue, container and worker lifecycle

```mermaid
  sequenceDiagram
    participant Runner as JobRunner
    participant Store as JobStore
    participant Runtime as Docker-compatible runtime
    participant Worker as backend.worker
    participant Adapter as DatasetAdapter
    participant Graph as GraphModule
    participant Output as Mounted output directory
    Runner->>Store: transition queued to running
    Runner->>Runtime: run isolated container with no network
    Note over Runtime: snapshot read-only, output read-write, resource limits enabled
    Runtime->>Worker: python -m backend.worker with snapshot/output paths
    Worker->>Worker: seed Python, NumPy and torch
    Worker->>Adapter: load dataset adapter
    Worker->>Graph: build graph from exact resources and frozen core
    Worker->>Output: publish initial empty metrics
    loop Each epoch and train batch
      Worker->>Adapter: load train batch
      Adapter-->>Worker: inputs and targets
      Worker->>Worker: zero gradients
      Worker->>Graph: forward including objective
      Graph-->>Worker: prediction and scalar loss
      Worker->>Worker: backward and Adam update, increment global step
      opt publication cadence or epoch boundary
        Worker->>Worker: preserve RNG streams and switch to eval/no_grad
        Worker->>Adapter: load full validation split
        Worker->>Graph: evaluate under no_grad and eval mode
        Graph-->>Worker: validation loss
        Worker->>Worker: restore random streams and prior model mode
        Worker->>Output: atomically write step point and epoch summaries
      end
    end
    Worker->>Adapter: load test split once after final epoch
    Worker->>Graph: evaluate final test loss
    Graph-->>Worker: finite test loss
    Worker->>Output: write final metrics
    Worker->>Worker: save cloned CPU state as safetensors
    Worker->>Worker: build standalone wheel with those weights
    Worker-->>Runtime: exit status
    Runtime-->>Runner: container result
    Runner->>Store: validate metrics, weights and exactly one wheel
    alt output is complete and finite
      Runner->>Store: commit completed only if cancellation did not win
    else worker failed or output is incomplete
      Runner->>Store: commit failed status and visible error
    end
```

Each step point pairs the sample-weighted training mean since the prior point
with full-split validation loss. The global optimizer step increases only after
a successful update. Epoch end publishes any remaining window once; if that
step already has a point, it reuses that validation value. Epoch summaries keep
the epoch-wide sample-weighted mean. Test loss runs once after the final epoch.

## Poll curves, cancel and recover

```mermaid
  sequenceDiagram
    actor User
    participant UI as Qt dashboard
    participant API as FastAPI jobs API
    participant Store as JobStore
    participant Runner as JobRunner
    participant Runtime as Container runtime
    loop Dashboard timer
      UI->>API: GET /v1/jobs
      API->>Store: list jobs for current owner
      Store-->>API: persisted records plus validated live metrics
      API-->>UI: history and statuses
      UI->>API: GET selected job
      API->>Store: read metrics.json and validate finite points
      Store-->>API: epochs, step points and test loss
      API-->>UI: selected job metrics and error
      UI->>UI: update labeled curves, scale and log without resetting scroll
    end
    alt queued job cancelled
      User->>UI: cancel selected job
      UI->>API: POST /v1/jobs/id/cancel
      API->>Store: transition queued to cancelled
    else running job cancelled
      User->>UI: cancel selected job
      UI->>API: POST /v1/jobs/id/cancel
      API->>Store: set cancellation request
      API->>Runner: stop named container
      Runner->>Runtime: stop container
      Runtime-->>Runner: stop result
      Runner->>Store: transition to cancelled or report stop failure
    end
    opt service starts after interruption
      Runner->>Store: inspect prior job records
      Store-->>Runner: interrupted running IDs
      Runner->>Runtime: stop interrupted named containers
      Runtime-->>Runner: stop results, including visible failures
      Runner->>Store: recover records, mark interrupted running jobs failed
      Store-->>Runner: surviving queued IDs
      Runner->>Runner: requeue surviving queued jobs
    end
```

Polling is asynchronous in Qt. Job records are per local owner; metrics readers
accept legacy epoch-only records and new step records. Metrics over the contract
limit or invalid/nonfinite points fail visibly. Container launch or worker
errors are recorded instead of represented as successful completion.

## Download artifacts or restore a snapshot

```mermaid
  sequenceDiagram
    actor User
    participant UI as Qt dashboard
    participant API as FastAPI
    participant Store as JobStore
    participant C as NNApplication
    User->>UI: download weights or wheel
    UI->>API: GET completed job artifact
    API->>Store: require completed job and safe output file
    Store-->>API: weights.safetensors or one wheel
    API-->>UI: streamed artifact
    UI->>UI: bounded temporary download then atomic save dialog
    User->>UI: restore snapshot to a new project
    UI->>API: GET /v1/jobs/id/snapshot
    API->>Store: return original model and uploaded files
    Store-->>API: exact frozen snapshot
    API-->>UI: project JSON and base64 files
    UI->>UI: validate paths and write into selected new directory
    UI->>C: open restored model through normal project loader
    alt destination and project are valid
      C-->>UI: restored project active
    else unsafe path, existing destination or open failure
      UI-->>User: visible error and active project preserved
    end
```

Restore does not overwrite the original project or an existing destination.
Downloaded weights and wheels require completed jobs. Safetensors is the
interchange checkpoint; pickle checkpoints are not emitted.

## Local API with remote Slurm containers (accepted 2026-10-08)

```mermaid
sequenceDiagram
    participant API as Local FastAPI
    participant Store as Local JobStore
    participant Runner as SlurmJobRunner
    participant SSH as Cluster SSH
    participant Slurm as Slurm scheduler
    participant Batch as Slurm batch wrapper
    participant SIF as Singularity worker
    API->>Store: Freeze request
    API->>Runner: Queue local job ID
    Runner->>SSH: Stage snapshot and worker source in new UUID directory
    Runner->>SSH: sbatch contained Singularity command
    SSH-->>Runner: Scheduler job ID
    Runner->>Store: Persist scheduler identity and configuration
    Slurm->>SIF: Execute same backend.worker
    loop Until scheduler termination
        Runner->>SSH: Read scheduler state and complete metrics
        SSH-->>Runner: State and metrics bytes
        Runner->>Store: Atomically publish local metrics
    end
    alt warning signal before per-allocation walltime
        Slurm->>Batch: SIGUSR1 warning
        Batch->>Batch: Create attempt-scoped checkpoint request file
        Batch->>SIF: Expose request path to worker
        SIF->>SIF: Check request after optimizer step
        SIF->>SIF: Save model, Adam, RNG, cursor and metric windows
        SIF-->>Slurm: Exit with checkpoint handoff status
        Runner->>SSH: Retrieve and validate checkpoint and logs
        Runner->>Store: Preserve same logical job and latest scheduler ID
        Runner->>SSH: Submit next allocation in same remote job directory
        SSH-->>Runner: New scheduler job ID
        Runner->>Store: Record new scheduler attempt
        Slurm->>SIF: Resume from private safetensors/JSON checkpoint
    else final allocation completed
        Runner->>SSH: Retrieve safe output files
        Runner->>Store: Validate final metrics weights wheel and commit completion
    else no valid checkpoint or explicit interruption
        Runner->>Store: Record visible failed or cancelled job
    end
    opt Cancel shutdown restart or transport failure
        Runner->>SSH: scancel recorded owned scheduler job
        Runner->>Store: Record cancellation or visible interruption/error
    end
```

Each local job keeps one ID across allocation attempts. Its Slurm ID changes and
the active ID alone is targeted by cancellation. Checkpoints are written in the
remote job output and transferred to local private output after each handoff;
they use safetensors plus JSON, never pickle, and remain unavailable from the
artifact API. The training adapter must replay an epoch deterministically from
the captured Python, NumPy and torch RNG states so the worker can skip batches
already covered by a checkpoint. A signal is honored at the next optimizer-step
boundary; a hard walltime kill before that boundary fails visibly without
automatic continuation. Final weights and wheel are created only after test
evaluation on the last allocation.
