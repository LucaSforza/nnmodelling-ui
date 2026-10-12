# Standalone MNIST VAE consumer

This isolated uv project consumes the trained VAE wheel through its public
`Model` API. The wheel bundles its graph runtime and default safetensors, so
inference does not need the backend or editable model project.

The consumer is pinned to the corrected 100-epoch MNIST VAE job
`173dec99-e10a-4579-9e9d-41fbe7f9fca3`. Its artifact identity and SHA256 are
recorded in `artifact.json`; the wheel distribution is `nnm_mnist_vae` and its
job-specific Python module is
`nnmodel_job_173dec99_e10a_4579_9e9d_41fbe7f9fca3`. The run used 64 training,
16 validation and 16 test images, seed 0, batch size 16 and learning rate
0.001. With pixel-summed reconstruction error plus mean KL, final training,
validation and test losses were 40.3861, 48.6109 and 42.6717.

From this directory:

```sh
uv run --no-project python download_wheel.py
uv sync --locked
uv run --locked python main.py
```

The downloader fetches only `/v1/jobs/{job_id}/wheel`, verifies the frozen
SHA256, and atomically replaces the destination wheel after verification. It
defaults to `http://127.0.0.1:8765`; pass `--base-url` for another local
backend address. Downloaded wheels and environments are ignored by git.

The consumer accepts an optional 28x28 grayscale PNG and compatible
safetensors file. With no PNG it supplies a raw NumPy image to
`Model.inference()` and `Model.infer()`. It checks that `encode()` returns a
finite deterministic `(1, 32)` latent, decodes that latent, then decodes one or
more seeded standard-normal prior latents. Reconstruction and prior images are
written to the ignored `output/` directory.

Use `--input image.png`, `--weights /path/to/weights.safetensors`, `--seed 7`,
`--count 4`, or `--output-dir /path/to/output` as needed. Pillow and CPU PyTorch
are declared by this consumer; the exported wheel supplies its own private
runtime and does not require the repository SDK. The public operations were
also verified with the job's separate `weights.safetensors` file.

To encode bundled held-out MNIST examples of 3 and 7, linearly interpolate
between their posterior-mean latents, and decode five intermediate points:

```sh
uv run --locked python main.py --interpolate
```

This writes seven 28x28 images and a readable `output/interpolation.png`
contact sheet labeled from digit 3 through digit 7. The endpoints come from
the official MNIST test split. Interpolation uses the wheel's public `encode`
and `decode` operations.

The earlier isolated consumer proof for the previous wheel is retained at
`/tmp/nnm-vae-proof-1856e337.XxhY54`. The corrected wheel was installed with
`uv sync --locked`; `main.py --interpolate` generated seven images and a labeled
contact sheet using the bundled public wheel operations. The standalone
interpolation consumer does not import the public SDK.
