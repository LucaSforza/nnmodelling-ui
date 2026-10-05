# MNIST variational autoencoder

Open this project in NNModelling to inspect the encoder and decoder subflows.
Each subflow receives its parent tensor through an inherited Input and returns
its mapped `out` boundary; neither subflow reloads the dataset.

The root graph flattens each grayscale 28×28 image to `[B,784]`, encodes it
through `784 -> 128 -> ReLU`, and produces separate 32-feature mean and
log-variance heads. The ordered project-owned `vae.diagonal-gaussian` join
packs these as `[B,2,32]`. `vae.reparameterize` samples
`z = mean + exp(0.5 * log_variance) * epsilon` while training, and uses the
posterior mean during evaluation. The decoder maps `[B,32]` through
`32 -> 128 -> ReLU -> 784 -> Sigmoid`, returning reconstructed pixels `[B,784]`.

The checked-in dataset contains 64 training, 16 validation, and 16 test images
from the official MNIST IDX files. The training and validation indices are
disjoint deterministic subsets of the official training split; test images
come from the official test split. `datasets/mnist/data.json` records the
published source checksums, download hashes, seed, and selected indices. The
adapter supplies normalized grayscale images `[B,1,28,28]` as the `image`
input. Each target is the same image flattened to `[B,784]`, so reconstruction
MSE compares every output pixel with its source.
The `vae.kl-divergence` branch computes one standard-normal KL value per sample:

```text
-0.5 * sum(1 + log_variance - mean² - exp(log_variance))
```

`core.mse-loss` returns scalar batch-mean reconstruction MSE. The explicit
`vae.total-loss` join adds the mean of the per-sample KL vector, preserving a
scalar objective for the root Loss Output. Every project-owned VAE stereotype
now has both Lua shape analysis and a Python `build(parameters, context,
services)` implementation; Python modules do not change the saved graph or
core package assets.

For a small CPU smoke job in the Training dashboard, use 1 epoch, batch size 32,
learning rate 0.001, seed 0, and publish every 10 optimizer steps. No container
training result or artifact is claimed here until the run is verified in the
local backend.
