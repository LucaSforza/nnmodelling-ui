# miniDeepSeek V4: first 1B checkpoint

Scaled implementation of the supplied DeepSeek-V4 report (`2606.19348v1`).
Verified registered parameter total: **997,982,444**, all trainable.
The editable component graph was composed through the native Qt application.
Recursive Lua counts, meta-device construction and actual CUDA construction agree.

Visible repeated subflows implement 14 blocks, width 512, four mHC residual streams, 32 routed
experts (four active) plus one shared expert, SwiGLU width 1120, and one MTP
block sharing token embedding and output weights. Attention uses shared KV,
low-rank query and grouped output projections, attention sinks, partial RoPE,
128-token sliding windows, CSA ratio 4 with overlap and a sparse indexer, and
HCA ratio 128. First three MoE layers use token hashes; later layers use learned
routing, load-bias updates and sequence balancing. The objective combines main
cross-entropy, 0.3-weighted MTP loss and auxiliary losses. See each resource's
`pytorch.py` and `parameters.lua` for executable mathematics.

This is a scaled research implementation. It uses FP32 parameters with BF16
CUDA autocast and Muon/AdamW; it does not reproduce the paper's FP8/FP4 kernels,
1M context, 32T-token corpus or reported model quality. This run is a first
checkpoint, not a completed original-scale pretraining.

Corpus: [FineWeb-Edu sample/10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. The manifest acquires its
first two Parquet shards directly on Slurm and verifies byte size and SHA256.
Original [DeepSeek-V4 tokenizer](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference),
revision `b5968e9190ef611bbf34a7229255be88a0e937c1`, vocabulary 129,280.
Document-ID SHA256 chooses deterministic held-out splits; 512-token windows
never join documents. Training budget is **99,999,744** input tokens (100M
rounded down to complete windows), with 128 validation and 128 test windows.
Atomic token caches survive allocation handoffs without exporting corpus data.

GPU calibration of the saved modular graph on Slurm job 1031863: Quadro RTX
6000, CUDA PyTorch 2.6.0+cu124, BF16, sequence 512, batch 1, two real
forward/backward/Muon updates. Parameter count 997,982,444; peak allocated
12,876,547,072 bytes, reserved 13,698,596,864 bytes; second step 6.26 seconds.
Losses 15.7616 and 15.8077 used synthetic tokens; this is not a trained
checkpoint. Successful pretraining provenance will record the actual job
and exported artifact after completion.

Open this project in Qt, inspect **Model > Parameter counts**, then use
**Training** with Slurm GPU allocation. Standalone inference consumer lives at
`examples/implmentation/miniDeepSeekv4` (requested spelling).

Dataset preparation verified on Slurm: download job 1031849 and token probe
1031852 both completed with exit 0:0. Actual split counts: 195,312 train windows
(99,999,744 tokens), and 128 windows (65,536 tokens) each for validation/test.
Both Parquet sizes and SHA256 hashes match the pinned manifest.
