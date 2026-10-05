# Local training smoke test

This small regression project verifies the complete local backend path quickly.
Its project-owned adapter loads the checked-in four-row regression payload only
when `load()` is called. It accepts one raw scalar and returns a list of
predictions. The explicit splits contain 4 training, 2 validation and 2 test
samples; no data is synthesized by the worker.

Run it from the repository root with the local backend from the
`nnmodelling-backend` skill. Before submitting, install workspace dependencies
using `uv sync --all-packages --group examples`. The dataset's own
`pyproject.toml` declares `nnmodelling-runtime`, so it can also be installed as
a standalone resource project.
