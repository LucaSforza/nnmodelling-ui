# Standalone Tiny Shakespeare consumer

This small application consumes the completed 20-epoch model wheel. Its UV
project is separate from the repository workspace, and its only model import
is the wheel's public `Model` class. The wheel contains its own runtime and
weights; generation does not call the backend or read the editable model.

From this directory, download the exact job artifact and verify its published
SHA256 before installing it:

```sh
uv run --no-project python download_wheel.py
uv sync --locked
uv run --locked python main.py --tokens 200
```

The downloader defaults to `http://127.0.0.1:8765`; use `--base-url` if the
local service listens elsewhere. It keeps the wheel under the ignored
`wheels/` directory and rejects a mismatched hash. To use a compatible external
weights file, pass `--weights /path/to/weights.safetensors`. The default uses
the weights bundled in the wheel.

Try a different prompt or generation length with:

```sh
uv run --locked python main.py $'ROMEO:\n' --tokens 120
```

The consumer imports the trained model wheel directly and repeatedly calls
`Model.inference(text)` / `Model.infer(text)`, keeping only the last 128
characters as context. It works whether or not the public SDK is installed;
the isolated verification copied only this application and wheel to `/tmp`,
confirmed `find_spec("nnmodelling_runtime")` returned `None` and the checkout
was absent from `sys.path`, then ran inference offline. PyTorch is limited to
two CPU threads, matching the completed training run.

An 8-character greedy smoke run produced `ROMEO:\nThe shal`. This confirms
the installed wheel performs local next-character inference; the small model's
greedy text remains repetitive and is not a demonstration of coherent dialogue.
