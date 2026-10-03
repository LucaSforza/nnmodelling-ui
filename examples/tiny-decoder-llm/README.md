# Tiny Decoder LLM

Created from a blank project through the native Qt GUI using Codex computer
use, including the project-owned dataset and causal-attention stereotype.

The dataset describes token IDs and next-token targets, both `int64[B,128]`.
Vocabulary size is 32000; model width is 512. The root graph is:

```mermaid
flowchart TD
    Tokens[Input: tokens] --> Embedding[Embedding: 32000 × 512]
    Embedding --> Position[Positional Encoding: context 128]
    Position --> Stack[Repeat: 6 decoder blocks]
    Stack --> Norm[LayerNorm: 512]
    Norm --> Head[Linear: 512 → 32000]
    Head --> Output[Output: logits]
    Head --> CE[Cross Entropy]
    CE --> Loss[Loss Output: scalar]
```

Enter Repeat to inspect the pre-normalized decoder block:

```mermaid
flowchart TD
    Input[Inherited Input] --> N1[LayerNorm: 512]
    N1 --> Attention[Causal Self Attention: 8 heads × 64]
    Input --> A1[Add: attention residual]
    Attention --> A1
    A1 --> N2[LayerNorm: 512]
    N2 --> Expand[Linear: 512 → 2048]
    Expand --> Activation[ReLU]
    Activation --> Project[Linear: 2048 → 512]
    A1 --> A2[Add: feed-forward residual]
    Project --> A2
    A2 --> Output[Mapped Output: out]
```

Expected tensors: embedding, attention, residuals, normalization and block
output `float32[B,128,512]`; expanded FFN `float32[B,128,2048]`; logits
`float32[B,128,32000]`; loss `float32[]`.

This is an editable architecture and shape-analysis example. The custom
attention rule checks its expected rank, dtype and width and retains shape;
its definition describes causal masking, Q/K/V projections and eight heads.
It does not execute attention. Dataset files contain metadata, not token samples.
There are no trained weights or numerical backend. The preserved Cross Entropy
rule checks floating logits/rank but does not fully validate target compatibility.

The computer-use findings and fixes are recorded in
`docs/knowledge/testing/ui-llm-authoring-2026-10-03.md`.
