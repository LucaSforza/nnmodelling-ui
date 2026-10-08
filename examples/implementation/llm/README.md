# Standalone Tiny Shakespeare consumer

This standalone application uses the wheel exported by Slurm job
`1e8594bb-3d8c-43e7-ae96-56bc94c461d9`. That job trained the bundled Tiny
Decoder LLM for 3 epochs over the bounded example corpus (64 training, 16
validation, and 16 test sequences), with batch size 16, learning rate 0.001,
and seed 0. Its final training, validation, and test losses were 3.4623,
3.3878, and 3.3453. This wheel represents the bounded example run; the
separate historical full-corpus 20-epoch result remains documented elsewhere.
The pinned wheel is `nnm_tiny_decoder_llm-0.1.0-py3-none-any.whl` with
SHA256 `154379d2eb73fb9d3d8dc1990b11fd1d0d67bf2d7c9835452501aef1e83ab601`.

The UV project is independent from the repository workspace. The downloaded
wheel bundles the graph runtime and trained weights. After download, inference
does not call the backend or read the editable model project.

Download the exact job artifact and verify its published SHA256 before
installing it:

```sh
uv run --no-project python download_wheel.py
uv sync --locked
uv run --locked python main.py --tokens 200
```

The downloader defaults to `http://127.0.0.1:8765`; use `--base-url` if the
local service listens elsewhere. It keeps the wheel under the ignored
`wheels/` directory and rejects a mismatched hash. To use the compatible
safetensors file from the same job, pass `--weights /path/to/weights.safetensors`.
The default uses weights bundled in the wheel.

Try a different prompt or generation length with:

```sh
uv run --locked python main.py $'ROMEO:\n' --tokens 120
```

The consumer calls the wheel's public `Model.inference(text)` and
`Model.infer(text)` methods and repeatedly generates one greedy character,
keeping only the last 128 characters as context. PyTorch uses two CPU threads.
The isolated 8-character smoke run printed `ROMEO:\n t t t t`; the output is
repetitive, as expected for this small run.
