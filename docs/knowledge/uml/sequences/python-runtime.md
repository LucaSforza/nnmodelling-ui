# Python dataset and numerical graph runtime

The C client persists and validates schema-v2 model structure but does not run
Python. Runtime behavior is isolated to the container worker and exported
wheel. Sources: `python/nnmodelling-runtime/src/nnmodelling_runtime/`,
`python/nnmodelling-runtime/src/stereotype_runtime/pytorch.py`,
`backend/worker.py` and `backend/Dockerfile`.

## Adapter and graph construction

```mermaid
  sequenceDiagram
    participant Worker as Isolated worker
    participant Project as Frozen project snapshot
    participant Adapter as DatasetAdapter
    participant Graph as GraphModule
    participant Catalog as Resource catalog
    participant Builder as Python package build
    Worker->>Worker: seed Python, NumPy and torch from frozen training config
    Worker->>Project: copy model and submitted files to temporary project root
    Worker->>Adapter: load_dataset(project root)
    Adapter->>Project: resolve exact active dataset and manifest
    Adapter->>Catalog: import declared Dataset entrypoint
    Catalog-->>Adapter: DatasetAdapter subclass
    Adapter-->>Worker: instantiate with dataset directory
    Worker->>Graph: GraphModule(project root, frozen core directory)
    Graph->>Catalog: resolve node package and dependency closure
    Catalog-->>Graph: exact manifests, definitions and Python entrypoints
  loop Root nodes in saved model order
      Graph->>Builder: build(parameters, BuildContext, services)
      Note over Builder: BuildContext.inputs is incoming edge count plus declared objective external inputs
      opt Builder requests a subflow body
        Builder->>Graph: services.build_subflow()
        Graph->>Builder: recursively build scoped nodes in model order
        Builder-->>Graph: fresh child modules and independent parameters
        Graph-->>Builder: runner owning registered child modules
      end
      Builder-->>Graph: registered torch.nn.Module
  end
    Graph-->>Worker: executable graph module
```

Package defaults are merged with node overrides and schema-validated before
each build. `BuildContext.inputs` reports actual graph arity: incoming edges
plus that package instance's declared objective external inputs. It preserves
definition input/output metadata. A stereotype reference built without a graph
instance has no known graph arity, so the runtime does not invent one. Modules
are registered under node identity. Repeat and horizontal-repeat instances own
freshly built child modules and parameters, using the resource builders for each
requested body rather than copying one initialized template. Construction and invocation limits
bound nested scopes.

## Batch execution and objective targets

```mermaid
  sequenceDiagram
    participant Worker
    participant Adapter as DatasetAdapter
    participant Graph as GraphModule
    participant Module as Package torch.nn.Module
    participant Loss as Loss Output collector
    loop Each training epoch and train batch
      Worker->>Adapter: load train split with batch size
      Adapter-->>Worker: Batch(inputs, targets)
      Worker->>Worker: zero gradients
      Worker->>Graph: forward(inputs, targets, include_loss=true)
      Graph->>Graph: bind root Input nodes by declared dataset slot
      loop Topological scope execution
        Graph->>Graph: sort incoming handles and append objective.externalInputs targets
        Graph->>Module: forward positional tensors
        Module-->>Graph: tensor or declared handle-keyed tensor map
        Graph->>Graph: store values by source node and handle
      end
      Graph->>Loss: collect the connected scalar objective result
      Loss-->>Graph: loss tensor without a Python module call
      Graph-->>Worker: prediction and loss
      Worker->>Worker: backward loss, Adam step, increment global step
      opt Publication cadence or epoch tail without an existing point
        Worker->>Worker: preserve RNG streams and enter eval/no_grad
        Worker->>Adapter: load full validation split
        Adapter-->>Worker: validation batches
        Worker->>Graph: forward validation with targets
        Graph-->>Worker: scalar validation objectives
        Worker->>Worker: restore RNG streams and prior training mode
        Worker->>Worker: publish weighted training window and full validation mean
      end
    end
    Worker->>Adapter: load test split after final epoch
    Adapter-->>Worker: test batches
    Worker->>Graph: forward in eval/no-grad mode
    Graph-->>Worker: finite test loss
```

Graph execution is a generic DAG keyed by scope and source handle. Join inputs
are sorted by numeric `in-N` suffix. Multiple output handles remain distinct;
unknown keys, missing required outputs and non-tensors fail visibly. Training
includes loss dependencies and maps targets only through each definition's
`objective.externalInputs`. Prediction inference evaluates the recursive
prediction dependency closure and omits unrelated loss branches; it never
fabricates targets. Scalar loss values are per-batch means, aggregated by sample
count. The accepted execution limits are 32 nested scopes and 256 subflow
builds/invocations.

The graph metadata is immutable for this instance. Dependency closure and
topological order are currently derived during each scope evaluation; cached or
lowered execution remains the performance TODO in the backend contract.

## Standalone wheel export and inference

```mermaid
  sequenceDiagram
    participant Worker
    participant Runtime as build_wheel
    participant Wheel as Private model package
    participant User as Inference caller
    participant API as Local artifact endpoint
    participant UV as Independent uv environment
    participant Adapter as Bundled DatasetAdapter
    participant Graph as Bundled GraphModule
    Worker->>Worker: clone state_dict tensors to CPU
    Worker->>Worker: save weights.safetensors
    Worker->>Runtime: build_wheel(project, core, weights, output, job ID)
    Runtime->>Wheel: copy exact graph and package closure
    Runtime->>Wheel: copy adapter Python, metadata and declared inference assets
    Runtime->>Wheel: vendor private SDK ABI and weights
    Note over Runtime,Wheel: Training split payloads and undeclared data stay out of wheel
    User->>API: GET completed job wheel
    API-->>User: wheel bytes
    User->>User: verify recorded SHA256 before installation
    User->>UV: sync frozen local wheel and CPU dependencies
    UV->>Wheel: install package with private runtime and bundled weights
    User->>Wheel: Model(weights_path optional)
    Wheel->>Adapter: construct from packaged project snapshot
    Wheel->>Graph: construct generic graph and load selected safetensors
    User->>Wheel: infer(raw input)
    Wheel->>Adapter: tokenize input
    Adapter-->>Wheel: one tensor for one Input, otherwise named tensor map
    Wheel->>Graph: prediction forward under inference mode
    Graph-->>Wheel: root prediction tensor
    Wheel->>Adapter: untokenize prediction
    Adapter-->>User: typed output value
```

The default model loads bundled `weights.safetensors`; an explicit compatible
path overrides it. Wheel resources are namespaced away from generated package
code and frozen core. Declared dataset `inferenceAssets` use project-relative
paths and are copied byte-for-byte. External resource dependencies are retained
in wheel metadata, while the SDK is privately vendored; the wheel is independent
of the service and source checkout.

The executable consumer at `examples/implementation/llm/` downloads the completed
job's wheel through the service's artifact endpoint and verifies its SHA256
before uv installs it with locked dependencies. The inference caller above is
then this standalone application: raw strings enter the public `Model` methods,
using bundled weights by default. The backend and editable projects under
`examples/models/` are not accessed during inference. Its optional greedy loop
feeds the final decoded character back through that same public boundary, with
the latest 128 characters as context.
