# Full Tiny Shakespeare training

Started after implementation commit `ced6dc5`, following the accepted user
choice of 20 epochs. Job: `a64143c8-07c2-40e7-8f89-37c33d64970b`, accepted
2026-10-05T19:45:03Z by the local FastAPI service. CPU container image:
`961783c493737136ddf4459f8fa81fe812b0c4d4bef826062273dee18a0fdd77`.

## Frozen input and configuration

Source: [Karpathy Tiny Shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt).
SHA256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
Full source: 1,115,394 characters; sorted vocabulary: 65 characters.
Submitted model SHA256:
`d3430cf5d49dadd47115adfbe5d4772d6a3bf390b7f3cdb2faed7c0d77f2de31`.
Model remains width 64, two blocks, four heads, head width 16, feed-forward 256,
context 128. Training starts from the frozen seed, without loading the small
three-epoch proof's weights.
The editable source has since moved to `examples/models/tiny-decoder-llm/`;
the hash above identifies the immutable submitted bytes, not later editor layout.

| Split | Character interval, end exclusive | Windows | Omitted final characters |
| --- | --- | ---: | ---: |
| Train | [0, 892315) | 6971 | 26 |
| Validation | [892315, 1003854) | 871 | 50 |
| Test | [1003854, 1115394) | 871 | 51 |

Every window contains 129 characters: 128 inputs and their shifted next-token
targets. Stride 128; no window crosses a split boundary. Full text remains in
the prepared payload; only incomplete tails are omitted from batches.

```json
{
  "epochs": 20,
  "batch_size": 64,
  "learning_rate": 0.001,
  "seed": 0,
  "publish_every_steps": 100
}
```

Adam optimizer; two CPU threads and the existing two-CPU/4-GiB container limit.
109 updates per epoch, 2180 updates total. Publication runs full validation at
global step multiples of 100 and each unique epoch tail: 41 paired points.
Epoch summaries retain full-epoch weighted training means; test evaluates once.

## Reproduction and local artifacts

```sh
uv run --group examples python tools/prepare_full_llm.py \
  --output-project .computer-use/full-llm-project-new
```

The command verifies the cached/downloaded corpus before copying the unchanged
example to a new directory. It refuses existing output paths, excludes
environments/caches, and writes dataset payload, vocabulary and `training.json`.
Open the prepared project in Qt, click Training, connect to loopback port 8765,
enter the frozen configuration above, then save and submit. The generic service
stores a new immutable snapshot; later source edits cannot affect this job.

This run's prepared project is local/ignored at `.computer-use/full-llm-project`.
The job snapshot contains 22 submitted resource files. Full corpus, weights,
wheel and worker log stay in the service's local job store, outside Git.

## Actual outcome

Completed successfully: 20 epoch summaries, 2180 optimizer updates, exactly
41 unique paired publications. The principal verified the exact sequence of
global multiples of 100 plus epoch-tail steps through the HTTP response.
Final epoch mean training loss: **1.725061**; validation loss: **1.840051**;
final test loss: **1.947120**; test character perplexity: **7.008475**.

| Epoch | Training loss | Validation loss |
| ---: | ---: | ---: |
| 1 | 2.912347 | 2.598723 |
| 2 | 2.527227 | 2.519589 |
| 3 | 2.451498 | 2.442963 |
| 4 | 2.363266 | 2.361374 |
| 5 | 2.280442 | 2.283588 |
| 6 | 2.207480 | 2.214982 |
| 7 | 2.142409 | 2.156906 |
| 8 | 2.084065 | 2.108579 |
| 9 | 2.031587 | 2.064273 |
| 10 | 1.984336 | 2.024906 |
| 11 | 1.942384 | 1.992218 |
| 12 | 1.906170 | 1.967411 |
| 13 | 1.873999 | 1.940961 |
| 14 | 1.844278 | 1.920789 |
| 15 | 1.818223 | 1.904031 |
| 16 | 1.795705 | 1.889235 |
| 17 | 1.775636 | 1.875108 |
| 18 | 1.757147 | 1.862406 |
| 19 | 1.740251 | 1.851257 |
| 20 | 1.725061 | 1.840051 |

Fresh Qt browser trial displayed this completed job, both published-step curves,
worker metric log and final test loss. Screenshot is local/ignored at
`.computer-use/full-llm-dashboard.jpg`.
The final trial includes the user-found validation-toggle brush fix: a fresh
Qt/bridge process shows both curve strokes without the spurious blue polygon,
including after toggling validation off and back on. Rendered regression rejects
the old fill behavior; corrected Qt test suites pass 6/6.

Exact 22-file snapshot verification passed. Downloaded artifacts:

* Weights: 437212 bytes; SHA256
  `60be4c793eb013fdad1e28f8b235c5c0c0108b34b17d1ff430c1c0fd932bbecb`.
* Wheel: 468083 bytes;
  `nnmodel_a64143c8_07c2_40e7_8f89_37c33d64970b-0.1.0-py3-none-any.whl`;
  SHA256 `efd2e36435cce8ba5e22121925cbdff1af13773f6b5494d912fdbd9a260de33d`.

Isolated target installation and raw-text inference passed without resolving
the public runtime SDK. Bundled and explicit weights-path predictions match.
Registered trainable parameters: 100161. Service download routes are
`/v1/jobs/a64143c8-07c2-40e7-8f89-37c33d64970b/weights` and `/wheel`.

## Quality comparison

Baselines use the same character targets as the fixed complete windows, not
different source slices. Target counts: 892288 train, 111488 validation,
111488 test. Uniform prediction over 65 characters gives loss ln(65) = 4.174387.
The unigram baseline estimates only training-target character frequencies with
add-one smoothing: p(c) = (count(c) + 1) / (892288 + 65). It uses no context.

| Baseline | Train loss | Validation loss | Test loss | Test perplexity |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 4.174387 | 4.174387 | 4.174387 | 65 |
| Training unigram | 3.309378 | 3.307429 | 3.347878 | 28.4423 |
| Trained decoder | 1.725061 | 1.840051 | 1.947120 | 7.0085 |

Loss is mean next-character cross entropy in natural-log units; character
perplexity is exp(loss). Held-out perplexity improves by about 4.06 times over
the training-only unigram baseline. Final train/validation gap is 0.115 nats,
test/validation gap 0.107 nats. Validation falls at every epoch through epoch 20;
there is no observed validation-loss rebound in this run. This supports useful
learning and held-out generalization for the small decoder; it does not measure
long-form semantic coherence.

Qualitative diagnostic: prompt `ROMEO:` followed by newline, 200 generated
characters, context truncated to the latest 128. Both diagnostics used the
isolated trained wheel and two CPU threads; neither changed the model or API.
Greedy generation repeatedly produced "the so" phrases. Seed-0 sampling at
temperature 0.8 reduced that loop but produced malformed words and weak syntax.
Local outputs: `/tmp/nnmodelling-full20-generation.txt` and
`/tmp/nnmodelling-full20-generation-temperature-0.8.txt`.

Assessment: numerical learning is good for this 100161-parameter initial
decoder and held-out loss improves substantially over context-free baselines.
Language quality remains weak; this is a working small character model, not
evidence of coherent dialogue or reliable long-text generation. Final validation
still falls, so this run does not establish a training plateau or prove that
more epochs alone would solve generation quality.

Before example relocation, Python verification: 52 passed, including full-preparation boundaries,
destination preservation and excluded-environment/included-resource symlinks.

## Standalone wheel application

`examples/implementation/llm/` now contains the executable uv consumer;
editable graph projects moved to `examples/models/`. Its downloader reads this
job's `/wheel` endpoint, verifies the exact SHA256 above before replacement,
then uv installs the local wheel and locked CPU dependencies. The consumer is
an independent uv workspace; the lock contains no public runtime SDK or backend.
Only downloaded wheel, virtual environment and weights remain local/ignored.

`just test-llm-consumer` performs verified download, locked sync and an eight
character greedy inference smoke. Actual output begins `ROMEO:` then `The shal`.
The principal also ran 32 characters offline, observing repetitive `the so`
phrases. The code uses only `Model.inference(text)` and `Model.infer(text)`,
latest 128-character context, two CPU threads, and bundled weights by default.
An optional compatible safetensors path also works.

Luna copied only consumer files and the downloaded wheel into `/tmp`; locked
offline sync and inference passed there. The public SDK could not be resolved
and repository checkout was absent from `sys.path`. Inference also passed with
the SDK deliberately available, proving independence without rejecting valid
coinstallation. Downloader tests cover verified replacement and preservation
of the previous wheel after checksum mismatch. Fresh core and Qt gates after
path migration pass 13/13 and 6/6; principal's final fresh Qt trial repeated
validation off/on without the filled polygon, then closed its temporary session.
Final integrated Python gate after relocation and consumer review: **54 passed**.
Mermaid validation: 35 diagrams in 40 KB files; `git diff --check` is clean.
