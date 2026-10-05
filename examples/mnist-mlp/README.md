# MNIST MLP classifier

This editable classifier consumes grayscale MNIST images and returns a digit
ID from 0 to 9. Its project-owned dataset adapter accepts a PNG path, PIL image
or NumPy array, normalizes pixels to `[0,1]`, and decodes logits to an integer.
Training uses a small verified subset of the official MNIST IDX files: 64 train
images, 16 validation images and 16 images from the separate official test
split. The payload records source hashes, split indices and counts.

From the repository root, run `uv sync --all-packages --group examples` and
`uv run --group examples python tools/prepare_example_data.py` to reproduce the
payload. The preparation script verifies the published MNIST MD5 checksums;
the worker does not download data. See [examples README](../README.md) for the
other projects and the backend skill for local training setup.
