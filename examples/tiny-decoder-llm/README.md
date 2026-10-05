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

The project-owned causal-mask, scaling, token-loss, and legacy causal-attention
resources each include PyTorch entrypoints and standalone `pyproject.toml`
metadata. The graph's explicit attention uses the generic core MatMul join; its
build context receives the number of connected input edges, so each QK and
attention-value multiplication constructs the correct two-input executor.

The dataset is responsible for mapping raw text to token IDs and producing
next-token targets. Its manifest carries the vocabulary as an inference asset;
training samples stay separate from exported model resources. The graph is
untrained until a backend training job writes weights.
