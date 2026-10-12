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

`core.mse-loss` returns scalar MSE averaged across the batch and 784 pixels.
`vae.total-loss` multiplies that scalar by 784 before adding mean per-sample KL,
so objective equals batch-mean summed pixel error plus KL. This scaling matches
the 28×28 image shape. Every project-owned VAE stereotype has both Lua shape
analysis and a Python `build(parameters, context, services)` implementation;
Python modules do not change the saved graph or core package assets.

The project declares two wheel operations. After training and installing the
exported wheel, `model.encode(image)` uses the dataset adapter and returns the
deterministic posterior mean `[1,32]`; `model.decode(z)` accepts a batched
latent tensor `[1,32]` and returns one normalized NumPy image `[28,28]`. Sample
from the standard-normal prior by calling `model.decode(torch.randn(1,32))`.
These operations use the same graph modules and trained weights as
`model.infer(image)`.

For this bounded example, use 100 epochs, batch size 16, learning rate 0.001,
seed 0, and publish every 10 optimizer steps. The 2026-10-08 corrected run on
64 training images completed with training loss 40.3861, validation loss
48.6109, and final test loss 42.6717. These values use pixel-summed MSE; older
artifacts used a different reduction. The exported wheel supports deterministic
`encode`, `decode(encode(image))`, and decoding points from the standard-normal
prior. The standalone consumer also generates a 3-to-7 latent interpolation.
