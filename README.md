# NNModelling native editor

Qt 6 Widgets frontend over an independent C11 NNModelling core. The C library
owns graph/model state, coordinates, validation, package metadata, Lua shape
analysis, application operations and schema-v2 project persistence. Only
`gui/qt/` is C++; Qt items hold stable model IDs and transient graphical state.
There is no training backend or compiler/IR implementation in this checkout.

The [knowledge base](docs/knowledge/README.md) defines ownership and semantics.
The dependency direction is Qt GUI → pure C API → C core, never the reverse.
Preserved stereotype packages and historical UML remain reference assets.

## Build and test

Core: C11 compiler, CMake and `just`. GUI additionally requires a C++17 compiler
and Qt 6 Widgets development files; GUI tests require Qt 6 Test. Lua and yyjson
are vendored C sources; builds download nothing.

```sh
just test      # C-only configuration; no Qt or C++ compiler required
just build
just test-ui   # separate offscreen Qt interaction tests
just run examples/mnist-mlp
```

Executable: `build/qt/nnmodelling-ui [project-directory]`. Without an argument,
choose Open, New project or New MNIST MLP. Save uses existing atomic C
persistence. Dirty project closure offers save, discard or cancel.

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
execution or compilation semantics.

GPLv3-or-later or separately negotiated commercial license; see [LICENSE](LICENSE).
Vendored components retain their original notices under `third_party/`.
