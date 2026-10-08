# Sapienza Slurm training verification — 2026-10-08

FastAPI remains on this computer at `http://127.0.0.1:8765`. Only training
workers ran on Sapienza via SSH alias `cluster`, partition `students` and
Singularity 3.9.8. The final five jobs all finished `COMPLETED`, exit `0:0`,
on `node111`, with two CPUs and 4G allocation; recorded runtime was 10–12 seconds.

Worker SIF is the immutable copy
`/home/sforza_2050030/nnmodelling-worker-20261008.sif`, SHA256
`be5d5a885d3525c61a8ef08fe1ec249d726d91c68a37540b24b0a7a157f88d2e`.
It contains Python 3.11.11, torch 2.6.0+cu124 (CPU execution), NumPy 2.2.2,
safetensors 0.8.0 and Pillow 11.0.0. Source and SDK were frozen separately per job.

Containers use clean environment, containment, no home, read-only snapshot and
source, writable output, and `--net --network none`. Actual scheduler probe
`1030700` verified an empty external route table and exited successfully.
The loopback-only network mode is documented by
[SingularityCE 3.9](https://docs.sylabs.io/guides/3.9/user-guide/networking.html).

Each run used the unchanged checked-in bounded dataset, Adam, three epochs,
batch size 16, learning rate 0.001, seed 0 and publication cadence 10.
DeepSeek was excluded and its untracked project was preserved. These are
bounded executable-example runs, not full-dataset convergence claims.

| Model | Slurm ID | Local API job ID | Train | Validation | Test | Updates |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| local-training | 1030701 | `89d4158b-fa01-4fa1-a4d1-a5a62180c309` | 17.066893 | 90.889069 | 182.977081 | 3 |
| mnist-mlp | 1030702 | `5e357013-690a-4bd3-aed5-4d8d919fde07` | 2.121266 | 2.144985 | 2.096830 | 12 |
| mnist-vae | 1030703 | `1856e337-b103-4f4e-a29b-4a8c7e6571e3` | 0.210845 | 0.241681 | 0.236612 | 12 |
| rnn-sine | 1030704 | `78b2824d-0dcc-4d89-a532-298aedcea914` | 0.341369 | 0.306203 | 0.309310 | 12 |
| tiny-decoder-llm | 1030705 | `1e8594bb-3d8c-43e7-ae96-56bc94c461d9` | 3.462321 | 3.387799 | 3.345268 | 12 |

Every completed API job contains validated metrics, safe safetensors and one
exported wheel. HTTP downloads were SHA256 checked before consumer installation.

| Model | Wheel SHA256 |
| --- | --- |
| local-training | `4eccbda7c6f3fc128979af95216c2f7062b01c8f18e52bea3e94d91637180576` |
| mnist-mlp | `4f9942b28cfbb4069efc316c896667cbde68f307d95cd91e5565046d1a74004f` |
| mnist-vae | `5f54b83442c55371a9e2052964c241a60f289aaee07de86f586e5cae6bcfc840` |
| rnn-sine | `b0aab5222f2a93360b6519b43f5f118bb6567e618713e4053bffa805f10121ba` |
| tiny-decoder-llm | `154379d2eb73fb9d3d8dc1990b11fd1d0d67bf2d7c9835452501aef1e83ab601` |

The LLM consumer is pinned to its final job. The VAE consumer was subsequently
refreshed after correcting its loss reduction, as recorded below. Both use
standalone uv projects and lockfiles. Actual offline proofs used Python 3.13.12 and
PyTorch 2.14.1+cpu, without the public SDK or repository checkout on `sys.path`.
Both public inference aliases and bundled/explicit weights gave identical outputs.
VAE additionally produced a deterministic finite `(1,32)` encoding, an identical
`decode(encode(image))` reconstruction, and a finite `(28,28)` seeded prior image.
The LLM eight-character smoke printed `ROMEO:\n t t t t`; this proves execution,
and does not demonstrate strong language quality.

The principal also installed all five final wheels together in an isolated
environment and verified raw-input inference, aliases and alternate weights.
Local proof directories:

* All five: `/tmp/nnmodelling-all-wheels-u368d5ec`.
* LLM: `/tmp/nnmodelling-llm-proof-jf9hilmq`.
* VAE: `/tmp/nnm-vae-proof-1856e337.XxhY54`.

## Corrected VAE retraining and latent interpolation

After the original smoke result revealed near-constant gray reconstructions, the
VAE objective was corrected to multiply pixel-mean MSE by 784 before adding
batch-mean per-sample KL. The same bounded MNIST data (64 train, 16 validation,
16 test) was trained for 100 epochs, batch size 16, learning rate 0.001 and seed
0. Slurm job `1030714` completed in 15 seconds with exit `0:0`; local API job
`173dec99-e10a-4579-9e9d-41fbe7f9fca3` reports final train/validation/test
losses `40.3861`, `48.6109` and `42.6717`. Its wheel SHA256 is
`ace092f8ae27d106d3192746d9f8824b6a3f239f3f4b4c9a628d7ba808941409`.

The standalone consumer now encodes official MNIST test images labeled 3 and 7,
linearly interpolates five points between their posterior means and decodes all
seven vectors through public wheel operations. The command
`uv run --locked python main.py --interpolate` completed and wrote seven images
plus `output/interpolation.png`; the contact sheet visibly transitions from 3
to 7. `uv sync --locked` verified the updated wheel hash in `uv.lock`.

Local/ignored evidence lives in `.computer-use/cluster-training/report.json`,
`.computer-use/cluster-training/wheel-proof.json` and the per-model artifact
directories. The earlier preliminary five-job run is preserved separately in
`.computer-use/cluster-training-initial/`; final evidence above uses the network-
isolated release path. Scheduler IDs identify actual compute work; mocked tests
are recorded separately.

## Reproduce

Current machine configuration is retained, without credentials, in
`.computer-use/cluster-backend.env`. To restart the local backend:

```sh
source .computer-use/cluster-backend.env
just backend-slurm
```

In another terminal:

```sh
just train-cluster-examples
just test-llm-consumer
just test-vae-consumer
```

The training command resumes recorded jobs. For a fresh run, use
`python tools/train_cluster_examples.py --wait --output-dir /new/report/directory`.
On another machine, configure its SSH alias, writable remote job directory
and prebuilt SIF as described in `examples/README.md`.

Validation: 115 integrated Python tests, 13 core tests and 7 Qt/CLI tests passed.
Official backend/example recipes also passed (26 and 51 tests). Standalone
consumer recipes, Mermaid validation and `git diff --check` passed.
