# RNN — sine-wave prediction

An editable, untrained NNModelling design. The project retains its requested
name, “RNN — previsione di una sinusoide”. Open this directory in the GUI, or use:

```sh
nnmodelctl --socket /tmp/opencode/nnmodelling.sock project.open \
  '{"path":"/absolute/path/to/examples/rnn-sine"}'
```

## Metadata and graph

Project-owned dataset `sine.windows@1.0.0`, **metadata only**:

- input `sequence`: float32 `[B,32,1]`, 32 consecutive samples;
- target `target`: float32 `[B,1]`, the immediately following value.

No samples, splits or checkpoints are included. Future window extraction must
avoid leakage between training and validation: split series or time intervals
before constructing overlapping windows.

```text
Root: sequence -> Subflow RNN -> Linear 32->1 -> Output [B,1]
                                      |
                                      +-> MSE Loss -> Loss Output []

Subflow RNN: inherited Input [B,32,1]
              -> tanh RNN, last state [B,32]
              -> internal Output (boundaryHandle="out")
```

The subflow uses `core.subflow-proxy@0.1.0`: its rule delegates to
`services.infer_subflow` rather than declaring a fabricated shape. The internal
Input inherits the owner's tensor, without cross-scope edges. Terminal
`rnn-boundary-out` maps the `out` output of owner `rnn`. There are no graph
cycles and no Repeat nodes.

Local stereotype `sine.rnn-last-state@1.0.0` exposes sequence length, input
size, hidden size, bias, dtype, tanh and zero initial state. Lua validates rank,
dtype and dimensions and returns `[B,hidden_size]`. This design uses sequence
length 32, input size 1 and hidden size 32. The 32 steps are temporal steps,
**not training epochs**.

## PyTorch resources and limitations

- `packages/sine.rnn-last-state-1.0.0/pytorch.py`: package entrypoint
  `build(parameters, context, services)`, following the core package convention.
  Builds only the internal module, without Linear. It does not require the
  external `stereotype_runtime` import used by core resources.
- The local manifest registers `entrypoints.pytorch` with `language="python"`
  and `file="pytorch.py"`, following the core manifest convention.
- Root `pytorch.py`: manual reference composition, **not** a client entrypoint.
  `build_predictor()` explicitly loads the local resource and adds exactly one
  `torch.nn.Linear(32,1)` head.
- `build_loss()` constructs mean-reduced MSE: float32 predictions and targets
  `[B,1]`, scalar result. No second prediction head.

The internal module contains a single `torch.nn.RNN`, unidirectional and
single-layer, with `batch_first=True`, `nonlinearity="tanh"` and zero dropout.
The same weights are reused across all 32 steps. Each call creates zero `h0`
on the input device and returns `h_n[0]`. State is not carried between windows.
The predictor must remain float32; a future caller must place the module and
input on the same device. The manual composition reflects the current design
values; it does not automatically read subsequent edits to `model.json`.

The current client performs only Lua shape/type analysis. There is no graph
compiler, PyTorch backend or active documented numerical runtime here. Python
resources are not executed or numerically validated by the client. Core MSE
checks floating dtype and produces scalar shape metadata; it does not prove a
numerical comparison with the target. The target metadata matches the core
MSE reference `batch.targets.target`; `[B,1]` is already compatible with its
`flatten_batch` reference.

Validation-set RMSE would be a future metric, distinct from the MSE loss.
No dataset or training pipeline has been implemented. No project Python
resources, training loops or backend services were executed during authoring;
PyTorch source availability does not establish numerical execution or training.
