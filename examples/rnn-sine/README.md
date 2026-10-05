# RNN — sine-wave prediction

An executable, editable sequence-regression example. A deterministic sum of two
sine waves is split into independent train, validation and test series before
extracting any windows. The checked-in series yield 64, 16 and 16 disjoint
windows respectively; each input has 32 samples and each target is the next
value. No overlapping window crosses a split boundary.

The Python dataset adapter accepts a 32-value sequence, returns a float
prediction, and loads batched tensors lazily from the project-owned series. The
RNN resource reuses one recurrent module across all 32 timesteps, then the
linear head predicts one value. Prepare or verify the generated series from
the repository root with `uv run --group examples python
tools/prepare_example_data.py`. See [examples README](../README.md) for shared
setup and backend instructions.
