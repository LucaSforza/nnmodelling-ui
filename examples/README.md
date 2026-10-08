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

## Slurm jobs with a local backend

Configure the local service with an existing SSH alias, an absolute remote job
root and a prebuilt Singularity image containing the worker dependencies.
Create that remote directory with private permissions before starting; health
checks that it exists and is writable:

```sh
export NNMODELLING_SLURM_HOST=cluster
export NNMODELLING_SLURM_ROOT=/absolute/cluster/path/nnmodelling-jobs
export NNMODELLING_SLURM_IMAGE=/absolute/cluster/path/worker.sif
just backend-slurm
```

API remains at `http://127.0.0.1:8765`; the normal Qt Training dashboard uses
that same endpoint. `POST /v1/jobs` stages immutable snapshots on the cluster,
then Slurm runs Singularity. Published losses and completed wheels return to
the local service. Docker remains the default for `just backend-run`.

In another terminal, submit all bundled models except DeepSeek and wait for
completed artifacts:

```sh
just train-cluster-examples
```

This uses three epochs, batch size 16, learning rate 0.001 and seed 0. Override
settings with `python tools/train_cluster_examples.py --help`. Reports and
artifacts are ignored under `.computer-use/cluster-training`; repeat the same
command to resume observation without resubmitting recorded jobs. Failures
remain visible. Standalone consumers live in `implementation/llm/` and
`implementation/vae/`; each verifies its frozen wheel hash before installation.
