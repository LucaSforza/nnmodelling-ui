#!/usr/bin/env python3
"""Write the expert-study Colab notebooks (docs/knowledge/testing/expert-study.md).

Usage: build_notebooks.py [--ref BRANCH] [--check]

Writes study/experts/notebooks/task{1,2}_*.ipynb (participants) and
solutions/ (facilitator). REF is the GitHub branch the start cell downloads
study_checks.py and the data from; it must match the branch of the Colab links.
--check only reports whether the files are up to date.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "study" / "experts" / "notebooks"
REPO = "LucaSforza/nnmodelling-ui"
DEFAULT_REF = "expert-study-materials"

SETUP = '''#@title ▶ Step 1: run this cell first (starts the task timer)
participant_id = ""  #@param {{type:"string"}}

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import importlib

try:
    import google.colab  # noqa: F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False
# Study helpers: in Google Colab always the current ones from GitHub, else from the repository.
if IN_COLAB:
    import urllib.request

    RAW = "https://raw.githubusercontent.com/{repo}/{ref}/"
    urllib.request.urlretrieve(RAW + "tools/expert_study/study_checks.py", "study_checks.py")
    urllib.request.urlretrieve(RAW + "examples/test/datasets/mnist/data.json", "mnist_small.json")
    sys.path.insert(0, os.getcwd())
    importlib.invalidate_caches()
for rel in ("../../../tools/expert_study", "../../../../tools/expert_study"):
    if os.path.isdir(rel):
        sys.path.append(os.path.abspath(rel))
import study_checks

study_checks.begin("{task}", participant_id)
x, y = study_checks.load_batch()
print("Task {task} started. x", tuple(x.shape), x.dtype, "| y", tuple(y.shape), y.dtype)
'''

PRIMITIVES = '''# Allowed: nn.Linear, nn.ReLU / torch.relu, nn.Flatten / torch.flatten, @ / torch.matmul,
#   transpose, scalar multiplication, softmax, torch.cat, +, nn.ModuleList, Python loops,
#   view / reshape (to split heads), F.cross_entropy / nn.CrossEntropyLoss.
# Forbidden: nn.MultiheadAttention, F.scaled_dot_product_attention,
#   F.multi_head_attention_forward, nn.Transformer*.

'''

TRY = '''# Optional, run it as often as you like: shapes, loss and parameter count of your model on the
# batch (what NNModelling shows in the Inspector and in Model > Parameter counts).
# It does not check your answer.
model = {construct}
logits = model(x)
loss = compute_loss(logits, y)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("logits", tuple(logits.shape), "| loss", round(loss.item(), 4), "| trainable parameters", params)
'''

SUBMIT = '''# Remove the "#" at the start of the next line, then run this cell ONCE.
# study_checks.submit("{task}", {model}, compute_loss)
'''

INTRO = '''# {title}

**Step 1: before anything else, type your participant number (written at the top of your task \
sheet, digits only) in the `participant_id` field below and run the cell** (click ▶ on its left). \
It starts the timer of this task.

**Step 2:** open *Your model* (click ▸ next to the title) and implement the model of \
**{sheet}** on the task sheet. *Try it* runs your model on a batch. No training is needed.

**Step 3:** when you are done, open *Submit*. You submit once: the answer is final.'''

SUBMIT_HEADING = '''## ⚠️ Step 3: Submit — remove the `#` in the cell below and run it ONCE

- Your answer is **final**: only the first submission counts, running it again changes nothing.
- No result is shown. A file `nnm-study-<your number>-{task}-….json` is saved to your **Downloads** folder: \
allow the download if the browser asks and leave the file there.
- Then tell the facilitator.'''

TASKS = {
    "T1": {
        "file": "task1_mlp.ipynb",
        "title": "Task 1: MLP classifier",
        "sheet": "Task 1",
        "model": "MLP",
        "construct": "MLP()",
        "model_text": "Implement `MLP` and `compute_loss` as in **Task 1** on the task sheet.",
        "start": (PRIMITIVES + '''class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Flatten -> Linear(784, 64) -> ReLU -> Linear(64, 10)
        raise NotImplementedError

    def forward(self, x):
        # TODO: return the logits, shape [B, 10]
        raise NotImplementedError
''', '''def compute_loss(logits, target):
    # TODO: cross-entropy between logits [B, 10] and target [B]
    raise NotImplementedError
'''),
        "solution": (PRIMITIVES + '''class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten(1, -1)
        self.hidden = nn.Linear(784, 64)
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        return self.out(torch.relu(self.hidden(self.flatten(x))))
''', '''def compute_loss(logits, target):
    return F.cross_entropy(logits, target)
'''),
    },
    "T2": {
        "file": "task2_multihead.ipynb",
        "title": "Task 2: multi-head self-attention with model-wide variables",
        "sheet": "Task 2",
        "model": "MultiHeadRowAttentionClassifier",
        "construct": "MultiHeadRowAttentionClassifier(d_model=32, num_heads=4)  # try other values too",
        "model_text": ("Implement `MultiHeadRowAttentionClassifier` as in **Task 2** on the task sheet. "
                       "`d_model` and `num_heads` play the role of the model-wide variables: your "
                       "submission is checked with `(d_model, num_heads)` = (32, 4), (64, 8) and (32, 2), "
                       "including the attention scale `1 / sqrt(d_model / num_heads)` in every head."),
        "start": (PRIMITIVES + '''class MultiHeadRowAttentionClassifier(nn.Module):
    def __init__(self, d_model=32, num_heads=4):
        super().__init__()
        self.embed = nn.Linear(28, d_model)  # provided
        self.classifier = nn.Linear(28 * d_model, 10)  # provided
        # TODO: per-head Q, K, V layers, each Linear(d_model, d_model // num_heads)
        #       (an nn.ModuleList of heads, or one wide Linear split with view, both work)

    def forward(self, x):
        h = self.embed(torch.flatten(x, 1, 2))  # provided: [B, 28, d_model]
        # ---- TODO: multi-head self-attention --------------------------------
        # per head: softmax(Q @ K^T / sqrt(d_model / num_heads), dim=-1) @ V
        # concatenate the heads on the last dimension -> attn [B, 28, d_model]
        raise NotImplementedError
        # ---------------------------------------------------------------------
        h = h + attn  # provided: residual
        return self.classifier(torch.flatten(h, 1))  # provided
''', '''def compute_loss(logits, target):
    return F.cross_entropy(logits, target)  # provided
'''),
        "solution": (PRIMITIVES + '''class Head(nn.Module):
    def __init__(self, d_model, d_head):
        super().__init__()
        self.q = nn.Linear(d_model, d_head)
        self.k = nn.Linear(d_model, d_head)
        self.v = nn.Linear(d_model, d_head)
        self.scale = 1 / math.sqrt(d_head)

    def forward(self, h):
        scores = self.q(h) @ self.k(h).transpose(-2, -1) * self.scale
        return torch.softmax(scores, dim=-1) @ self.v(h)


class MultiHeadRowAttentionClassifier(nn.Module):
    def __init__(self, d_model=32, num_heads=4):
        super().__init__()
        self.embed = nn.Linear(28, d_model)
        self.classifier = nn.Linear(28 * d_model, 10)
        self.heads = nn.ModuleList(Head(d_model, d_model // num_heads) for _ in range(num_heads))

    def forward(self, x):
        h = self.embed(torch.flatten(x, 1, 2))
        attn = torch.cat([head(h) for head in self.heads], dim=-1)
        h = h + attn
        return self.classifier(torch.flatten(h, 1))
''', '''def compute_loss(logits, target):
    return F.cross_entropy(logits, target)  # provided
'''),
    },
}


def _cell(kind, cell_id, source, **metadata):
    cell = {"cell_type": kind, "id": cell_id, "metadata": {"id": cell_id, **metadata},
            "source": source.splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def notebook(task, variant, ref):
    spec = TASKS[task]
    key = task.lower()
    model_code, loss_code = spec[variant]
    sections = [f"{key}-model", f"{key}-try", f"{key}-submit"]
    cells = [
        _cell("markdown", f"{key}-intro", INTRO.format(**spec)),
        _cell("code", f"{key}-start", SETUP.format(repo=REPO, ref=ref, task=task), cellView="form"),
        _cell("markdown", sections[0], "## Step 2: Your model\n\n" + spec["model_text"]),
        _cell("code", f"{key}-model-code", model_code),
        _cell("code", f"{key}-loss-code", loss_code),
        _cell("markdown", sections[1], "## Try it (optional)\n\nRuns your model on the batch and prints "
              "shapes, loss and parameter count. It does not check your answer."),
        _cell("code", f"{key}-try-code", TRY.format(construct=spec["construct"])),
        _cell("markdown", sections[2], SUBMIT_HEADING.format(task=task)),
        _cell("code", f"{key}-submit-code", SUBMIT.format(task=task, model=spec["model"])),
    ]
    metadata = {
        "colab": {"collapsed_sections": sections, "generative_ai_disabled": True, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}


def render(task, variant, ref):
    return json.dumps(notebook(task, variant, ref), indent=1, ensure_ascii=False) + "\n"


def targets(ref):
    for task, spec in TASKS.items():
        yield OUT / spec["file"], render(task, "start", ref)
        yield OUT / "solutions" / spec["file"], render(task, "solution", ref)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", default=DEFAULT_REF, help=f"GitHub branch for downloads (default {DEFAULT_REF})")
    parser.add_argument("--check", action="store_true", help="fail if a notebook is not up to date")
    args = parser.parse_args(argv)
    stale = []
    for path, text in targets(args.ref):
        if path.is_file() and path.read_text(encoding="utf-8") == text:
            continue
        stale.append(path)
        if not args.check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)}")
    if args.check and stale:
        print("stale: " + ", ".join(str(p.relative_to(ROOT)) for p in stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
