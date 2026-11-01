# MNIST MLP classifier

This editable classifier consumes grayscale MNIST images and returns a digit
ID from 0 to 9. Its project-owned dataset adapter accepts a PNG path, PIL image
or NumPy array, normalizes pixels to `[0,1]`, and decodes logits to an integer.
The default dataset contains all 60,000 official training images (55,000 train
and 5,000 validation) and all 10,000 test images. The dataset manager also
offers a small verified subset of the official MNIST IDX files: 64 train
images, 16 validation images and 16 images from the separate official test
split. The payload records source hashes, split indices and counts.

From the repository root, run `uv sync --all-packages --group examples` and
`uv run --group examples python tools/prepare_example_data.py` to reproduce the
small payload. The complete compressed IDX files are included for offline use;
both use published MNIST checksums. The worker does not download data. See [examples README](../../README.md) for the
other projects and the backend skill for local training setup.
