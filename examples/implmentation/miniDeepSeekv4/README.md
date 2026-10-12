# miniDeepSeekv4 inference

This small consumer loads an installed exported model wheel by its recorded
import package name and runs greedy text generation through the public
`Model.logits` operation. It preserves token IDs between steps, truncates each
input to the frozen 512-token context, and stops at EOS or the requested token
limit. The default device is CUDA; pass `--device cpu` explicitly for CPU
inference. A requested unavailable CUDA device fails instead of falling back.

Sync the consumer requirements and install the completed job wheel into the
same environment, then run:

```sh
uv sync --no-install-project
uv pip install /path/to/completed-job.whl
.venv/bin/python main.py \
  --model-package nnmodel_<recorded-job-import-name> \
  "A prompt for the trained model"
```

By default, the tokenizer and its pinned configuration are loaded from the
wheel's declared inference assets. `--tokenizer /path/to/tokenizer.json` may
select an existing copy of the official frozen tokenizer; an adjacent
`config.json` is used when present, otherwise the wheel's configuration is
used. `--weights /path/to/weights.safetensors` selects a compatible trained
checkpoint explicitly. The model package must already contain trained bundled
weights or receive this override; the consumer does not create random weights.

For example, use `--max-new-tokens 64` to bound output. The final run record
should identify the successful training job and wheel SHA256 so its model and
tokenizer provenance can be reproduced.
