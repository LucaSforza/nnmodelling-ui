# Tiny Decoder LLM

A compact, executable decoder-only language model built from a blank native
project. It keeps the decoder `Repeat` and its four-head `HorizontalRepeat`
editable while expressing attention as ordinary graph nodes.

The model uses a 65-token vocabulary, context length 128, model width 64, two
decoder blocks, four attention heads of width 16, and a 256-wide feed-forward
layer. Each attention head projects Q, K, and V separately, transposes K,
computes `Q × Kᵀ / √16`, masks future positions, applies softmax, and multiplies
by V. The four head outputs concatenate to width 64. A project-owned token
cross-entropy module accepts logits `[B,T,V]` and targets `[B,T]` and reduces
them to a scalar mean loss.

The project-owned causal-mask, token-loss, and legacy causal-attention
resources each include PyTorch entrypoints and standalone `pyproject.toml`
metadata. Attention scaling uses the preserved `core.scale` resource. The
graph's explicit attention uses the generic core MatMul join; its build context
receives the number of connected input edges, so each QK and attention-value
multiplication constructs the correct two-input executor.

The dataset is responsible for mapping raw text to token IDs and producing
next-token targets. Its manifest carries the vocabulary as an inference asset;
training samples stay separate from exported model resources. The graph is
untrained until a backend training job writes weights.

## Reproduce the full-corpus follow-up

After the first implementation commit, prepare a separate project copy using
the verified cached Tiny Shakespeare source:

```sh
uv run --group examples python tools/prepare_full_llm.py \
  --output-project .computer-use/full-llm-project
```

The output path must not already exist. The command verifies the complete
1,115,394-character source and its pinned SHA256, copies this project without
environment/cache folders, and writes full-corpus splits, vocabulary and
`training.json` into the new directory. The checked-in bounded fixture and this
source project remain unchanged. It records the frozen 20-epoch configuration
(batch 64, learning rate 0.001, seed 0, publish cadence 100), contiguous 80/10/10
character offsets, complete 129-character windows at stride 128, and omitted
tail counts. Expected windows: 6,971 train, 871 validation and 871 test. The
full corpus, trained weights and wheel belong in local job artifacts, not Git.
To train the generated project in the Qt application, open that new project,
choose **Training**, and enter the values from its `training.json`: 20 epochs,
batch size 64, learning rate 0.001, seed 0 and publish cadence 100. The dialog
does not load that file automatically.
