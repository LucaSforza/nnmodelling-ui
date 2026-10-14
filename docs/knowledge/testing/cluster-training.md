# Slurm training verification — 2026-10-08

FastAPI remains on this computer at `http://127.0.0.1:8765`. Only training
workers ran on an explicitly configured SSH/Slurm cluster with
Singularity 3.9.8. The final five jobs all finished `COMPLETED`, exit `0:0`,
on an allocated worker node, with two CPUs and 4G allocation; recorded runtime was 10–12 seconds.

Worker SIF is the immutable copy
a prebuilt remote worker image (account-specific path omitted), SHA256
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
These are
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

Configure the per-user JSON profile described in
[the user guide](../../user/en/server.md), including executor, SSH host, writable
remote directory, prebuilt SIF and partition. No account or cluster is supplied
by the repository. Remote reproduction requires your own reachable cluster:

```sh
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

## Metavariables, prepared SIF and portable imports — 2026-10-14

The principal drove the actual Qt application through the project noVNC bridge;
native computer-use surfaces were unavailable. Three GPT-6 Luna workers wrote
the bounded C, Qt and backend/runtime changes after contracts and UML were
updated. The checked-in tiny decoder was copied into the ignored trial directory;
the original example was preserved.

The Operations-style manager created integer `embedding-features=64` and
`num-heads=4`. Inspector edits bound embedding/positional dimensions and all
Q/K/V `in_features` to `$embedding-features`, Horizontal Repeat `times` to
`$num-heads`, and all Q/K/V `out_features` to
`$(params.in_features / meta["num-heads"])`. Scale uses
`$(1 / math.sqrt(meta["embedding-features"] / meta["num-heads"]))`.
Changing `num-heads` to 8 through the manager changed each projection's inferred
last dimension from 16 to 8; concatenation retained 64 features and the GUI
reported no model problems. The authored schema-3 model was saved through Qt.

Prepare and install worker built the locked CPU environment locally, transferred
and SHA256-verified its Docker archive, and submitted actual Slurm conversion
job `1033259`. It finished `COMPLETED`, exit `0:0`, in 2:58. Dependency/startup
probes ran with the training source mounted over `/app`; the Python environment
remained available at `/opt/nnmodelling/venv`. The helper published immutable
`worker-151c817a64aaf577c8bafd470f496efba29f42a637176caed63ba0057ffa24f0.sif`
and Qt updated the isolated trial profile. Check worker also succeeded through
the configuration dialog. No account path or credential is committed here.

Qt then started the local service, confirmed Slurm health, and submitted three
CPU epochs with batch size 16, learning rate 0.001, seed 0 and publication
cadence 4. Scheduler job `1033260` finished `COMPLETED`, exit `0:0`, in 30 seconds;
local job ID is `b09b4182-c80a-408c-9398-7f3a584da515`. The dashboard displayed
completed state, finite paired curves and final test loss `3.0620439052581787`.

| Epoch | Training loss | Validation loss |
| --- | ---: | ---: |
| 1 | 3.8181790113449097 | 3.459254264831543 |
| 2 | 3.373031497001648 | 3.2701480388641357 |
| 3 | 3.1861157417297363 | 3.1101107597351074 |

There were 12 optimizer updates and 24,576 processed tokens. This is a bounded
deployment/inference smoke, not a language-quality or full-corpus convergence
claim. The source snapshot retained declarations and bindings. The separate
execution snapshot and the wheel's model were schema 2 with concrete values:
eight heads, 64 input features, eight Q/K/V output features and the resolved
scale factor. Original identity/resources remained intact.

Wheel, portable ZIP and safetensors were downloaded using the dashboard buttons.
Each download matched its validated local job artifact byte for byte.

| Artifact | SHA256 |
| --- | --- |
| `nnm_tiny_decoder_llm-0.1.0-py3-none-any.whl` | `399a3c3d284917434788105e132c173a19cb15a554b382edb1032f68108ae661` |
| `python-package.zip` | `5235e5bc5a41852ee06251064767ac493a654bc0a2cd1de98b4b4e9fe4557e97` |
| `weights.safetensors` | `031ac043d709f6e0a4ac6e8260702879373d2f5f1b1069727627c42abf23eec9` |

Three fresh Python 3.14.8 processes tested the extracted wheel package, extracted
portable ZIP and an offline `uv pip install --no-deps --target` installation.
Each used the literal public import `from nnm_tiny_decoder_llm import Model`,
verified its expected package location, excluded repository source and editable
`.pth` files, and confirmed the public NNModelling SDK was not importable.
Third-party dependencies were curated from the installed environment, including
PyTorch 2.14.1+cpu. Resetting the RNG before comparisons proved `infer` and
`inference` agree and downloaded explicit weights match bundled weights.
Eight generated characters produced `ROMEO:\nhTHUey e` in every process.
This model exposes text inference and declares no public tensor operation;
consumer tensor-finiteness testing therefore does not apply.

Local evidence is under `.computer-use/metavariable-slurm-trial/`: authored
`tiny-decoder-llm/model.json`, profile, job snapshots/metrics, downloaded
`artifacts/`, `formula-heads-8.png`, `worker-installed.png`,
`training-and-downloads.png`, and `consumer-proof/actual-smoke.txt` with its
reproduction README. One initial Qt VNC session crashed in the installed
`libqvnc.so` redraw path while dismissing a combo popup; its unsaved trial edits
were discarded. A fresh session using keyboard type selection completed the
entire authoring, preparation, training and download flow. This records a bridge
environment limitation, not a claim of native Wayland verification. The saved
project was closed through Qt; bridge and owned local backend were stopped.

Final automated gates passed: core 16/16, Qt/CLI 8/8, backend 113 pytest cases,
and runtime/operations/LLM/VAE consumer 58 pytest cases. Socket/cache-dependent
checks ran outside the restricted sandbox. Mermaid validated 44 diagrams in
41 files; the offline user guide verified 13 pages. `git diff --check` passed.
