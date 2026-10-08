# Standalone MNIST VAE consumer

This isolated uv project consumes the trained VAE wheel through its public
`Model` API. The wheel bundles its graph runtime and default safetensors, so
inference does not need the backend or editable model project.

The consumer is pinned to the completed three-epoch MNIST VAE job
`1856e337-b103-4f4e-a29b-4a8c7e6571e3`. Its artifact identity and SHA256 are
recorded in `artifact.json`; the wheel distribution is `nnm_mnist_vae` and its
job-specific Python module is `nnmodel_job_1856e337_b103_4f4e_a29b_4a8c7e6571e3`.
This bounded run used 64 training images, 16 validation images and 16 test
images, with seed 0, batch size 16 and learning rate 0.001. Training,
validation and test losses were 0.2108, 0.2417 and 0.2366 respectively.

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

The proof copied only the consumer application, lock and metadata, downloaded
wheel, explicit weights and one 28x28 MNIST test PNG into
`/tmp/nnm-vae-proof-1856e337.XxhY54`. In that isolated copy,
`uv sync --locked --offline` succeeded with Python 3.13.12 and CPU PyTorch
2.14.1. Both CLI modes completed offline; `Model.inference()` and `Model.infer()`
matched, `encode()` returned a deterministic finite `(1, 32)` latent,
`decode(encode(image))` matched inference, and a seeded prior decode produced a
finite 28x28 image. Bundled and explicit weights produced identical outputs.
The installed model imported from the isolated environment, the public
`nnmodelling_runtime` SDK was absent, and the source checkout was absent from
`sys.path`.
