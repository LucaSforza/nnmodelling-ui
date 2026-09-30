# MNIST variational autoencoder (design only)

Open this directory or use **New MNIST VAE**. Encoder and decoder are nested
`core.subflow-proxy` nodes: double-click either card, return with Parent/Scope.
Their Input boundary inherits the parent tensor; it does not reload the dataset.

Encoder: 784 -> 128 -> ReLU -> two 32-feature heads, mean and log variance.
The Gaussian join packs them as `[B,2,32]` (mean first). Reparameterization
models `z = mu + exp(0.5 * log_variance) * epsilon`, `epsilon ~ N(0,I)`.
Decoder: 32 -> 128 -> ReLU -> 784 -> Sigmoid, yielding reconstruction `[B,784]`.
KL is a separate per-sample `[B]` branch against the standard normal prior:
`-0.5 * sum(1 + log_variance - mu^2 - exp(log_variance))`.

All VAE-specific stereotypes are project-owned, not global/core packages.
Lua performs shape/dtype analysis only: it does not sample, reconstruct images,
load MNIST, train weights or execute a numerical objective. Dataset metadata
declares image input and flattened-image reconstruction target. A future backend
may combine reconstruction and KL terms; no such backend exists here.
