# NNModelling native editor

Qt 6 Widgets frontend over an independent C11 NNModelling core. The C library
owns graph/model state, coordinates, validation, package metadata, Lua shape
analysis, application operations and schema-v2 project persistence. Only
`src/gui/qt/` is C++; Qt items hold stable model IDs and transient graphical state.
The local FastAPI backend runs training in isolated CPU containers; there is no
compiler/IR implementation in this checkout.

📖 **[Documentazione online](https://lucasforza.github.io/nnmodelling-ui/)**

The illustrated user manual opens in [English](docs/user/index.html) by default
and is also available in [Italian](docs/user/introduction.html), with a language
selector on every page. It
covers client panels, menus and fields, backend setup/API commands, and a Tiny
Decoder LLM training tutorial. It works offline;
[Markdown sources](docs/user/README.md) are included.

The [knowledge base](docs/knowledge/README.md) defines ownership and semantics.
The dependency direction is Qt GUI → pure C API → C core, never the reverse.
Preserved stereotype packages and historical UML remain reference assets.

## Source organization

Each module has its own directory under `src/`: `model`, `catalog`, `project`,
`inference`, `application`, `automation`, `nnmodelctl`, `utils` and `gui/qt`.
Implementation files are split by responsibility (e.g. project loading/saving,
datasets/resource transactions, Lua runtime/tensors/reports, window panels/forms).
Include public headers as `application/application.h`, `model/model.h`, etc.
`utils/utils.h` contains global helpers; module-qualified private helper headers
and `*_internal.h` stay inside their owning module. Tests remain under `tests/`.
See the [source-layout contract](docs/knowledge/contracts/source-layout.md).

## Build and test

Core: C11 compiler, CMake and `just`. GUI additionally requires a C++17 compiler
and Qt 6 Widgets development files; GUI tests require Qt 6 Test and Python 3
for the live CLI/UI integration test. Lua and yyjson
are vendored C sources; builds download nothing.

```sh
just test      # C-only configuration; no Qt or C++ compiler required
just build
just test-ui   # separate offscreen Qt interaction tests
just test-sanitize # C-only ASan/UBSan/leak gate (Clang by default)
just run examples/models/mnist-mlp
```

Executable: `build/qt/nnmodelling-ui [project-directory]`. Without an argument,
choose Open, New project or New MNIST MLP. Save uses existing atomic C
persistence. Dirty project closure offers save, discard or cancel.

**New mini LLM** creates and opens an editable copy of Tiny Decoder LLM, with
token dataset, decoder blocks and explicit attention graph. Model starts untrained.
Saved JSON uses two-space indentation. Top/bottom stereotype parameters render
as card rows and value badges. **New stereotype** and **New dataset** open visual
forms; their Create action saves the project and immediately activates resources.
**Dataset** opens the project dataset manager: create, edit metadata/tensor slots,
or select the active dataset. Editing preserves exact identity, adapter code and
data files; saving also saves current graph edits. **New dataset** remains a shortcut.

Direct library build without GUI or tests:

```sh
cmake -S . -B build/core-only -DNN_BUILD_GUI=OFF -DBUILD_TESTING=OFF
cmake --build build/core-only
```

If a sandbox disallows the configured ccache directory, use
`CCACHE_DISABLE=1 just test` (and similarly for the GUI build).

## Development agents

For development with OpenCode V2 background agents and inter-agent messaging,
see [the OpenCode V2 swarm setup](docs/opencode2.md).

## Editor

Search the Components palette and double-click a package to create a node.
Drag output ports to input ports to connect; C validates the request. Drag nodes
to move them, select with Ctrl/rubber band, and press Delete to remove selection.
Wheel zooms; middle-button or Space-drag pans. Fit frames the current scope.
The inspector edits names and package-defined parameters; resources and analysis
show project metadata and C/Lua diagnostics.

Subflows have a distinct container card. Double-click to enter; use scope
navigation to return. Child coordinates and edges remain C-owned and scoped.
Nonempty subflows must be emptied before deletion. The editor preserves existing
same-scope connections and package proxy nodes; it does not invent cross-scope
numerical execution or compilation semantics. C/Lua recursively analyzes Proxy
scopes with inherited boundary inputs, including children hidden from the canvas.

### Typed outputs

Stereotypes may override their default outgoing handle with `outputs`:

```json
"outputs": [
  { "id": "prediction", "type": "output" },
  { "id": "objective", "type": "loss" }
]
```

At most one handle per type is allowed. Defaults are `out`/`output`, or
`loss`/`loss` for a `kind: "loss"` calculator; explicit declarations replace
the default. Intermediate nodes accept either type as ordinary tensor inputs.
Only terminals restrict connections: Output accepts normal output, Loss Output
accepts loss. Each collector takes one edge; insert an explicit join to combine
loss contributions. Outgoing ports/edges are black or red by their source type.

Input, Output and Loss Output appear as filled black, brown and red circles.
Every complete root has one Output and one Loss Output. Subflow creation spawns
one internal Input and one mapped terminal for each declared output; designers add
nodes and connections. Nested terminal `data.boundaryHandle` names the external
handle. Single-output Lua uses `output=tensor`; two-output Lua uses a keyed
`outputs={prediction=tensor1,objective=tensor2}` result. Inspect all outputs in
the inspector or `analysis.diagnostics`. See the
[typed outputs contract](docs/knowledge/contracts/typed-outputs.md).

## LLM/local command interface (Linux)

Start the UI once with a private socket; no backend, MCP or network server:

```sh
./build/qt/nnmodelling-ui --socket /tmp/opencode/nnmodelling.sock
./build/qt/nnmodelctl --socket /tmp/opencode/nnmodelling.sock project.create \
  '{"parent":"/tmp/opencode","id":"my-vae","name":"My VAE","template":"mnist-vae"}'
./build/qt/nnmodelctl --socket /tmp/opencode/nnmodelling.sock ui.scope '{"id":"encoder"}'
./build/qt/nnmodelctl --socket /tmp/opencode/nnmodelling.sock project.snapshot
```

`nnmodelctl --help` lists commands. JSON arguments may come from stdin using `-`.
Both visual dialogs and `stereotype.create`/`dataset.create` use the same C
transaction. See [protocol and payloads](docs/knowledge/contracts/automation.md)
and [LLM use cases](docs/knowledge/uml/automation.md). Socket permissions are
0600; existing paths are rejected rather than overwritten. `just test-cli`
tests real command/resource authoring, VAE navigation and UI screenshot capture.

GPLv3-or-later or separately negotiated commercial license; see [LICENSE](LICENSE).
Vendored components retain their original notices under `third_party/`.
