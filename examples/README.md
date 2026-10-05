# Executable examples

These five projects under `models/` are small, editable NNModelling designs with runnable
Python dataset adapters and bounded local training payloads. The native client
opens and validates each project; the backend runs Python only in its isolated
CPU worker container.

From the repository root, prepare the public example data and install the
workspace's example dependencies with:

```sh
uv sync --all-packages --group examples
uv run --group examples python tools/prepare_example_data.py
```

The preparation script downloads official MNIST IDX files and the pinned
Karpathy Tiny Shakespeare source into `/tmp/nnmodelling-example-cache`. It
verifies source checksums and rewrites only the bounded payloads in the example
projects. Sine data is generated deterministically. Training never downloads
data. For setup, running the local service, and submitting jobs, see the
repository's `nnmodelling-backend` skill.

| Project | Dataset | Bounded splits | Raw inference input |
| --- | --- | --- | --- |
| [local-training](models/local-training/) | Tiny linear regression | 4 / 2 / 2 explicit rows | scalar |
| [mnist-mlp](models/mnist-mlp/) | Official MNIST | 64 / 16 / 16 images | PNG path, PIL image, or NumPy array |
| [mnist-vae](models/mnist-vae/) | Official MNIST | 64 / 16 / 16 images | PNG path, PIL image, or NumPy array |
| [rnn-sine](models/rnn-sine/) | Deterministic sine series | 64 / 16 / 16 windows | 32-value sequence |
| [tiny-decoder-llm](models/tiny-decoder-llm/) | Tiny Shakespeare | 64 / 16 / 16 windows | up to 128 characters |

Every dataset resource declares its `DatasetAdapter` entrypoint and standalone
uv metadata. Data loading is lazy; inference uses only the adapter and declared
assets. Training payloads are excluded from exported model wheels.
