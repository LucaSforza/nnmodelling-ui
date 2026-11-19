# Expert usability study materials (proposed 2026-11-19)

Status: PROPOSAL for the research team (not yet accepted by the co-author).
Study tooling only: no client, model, file-format or backend semantics change.

Experts perform the same three tasks in NNModelling (NNM) and in PyTorch
(Jupyter), in counterbalanced order. One English task sheet serves both tools:
a tool-neutral specification per task plus an "In NNModelling" and an
"In PyTorch" box. No training is performed in either tool; a task is complete
when the model passes the automatic check. SUS and NASA-TLX are owned by the
research team and are not part of these materials.

## Fixed data and dimensions

Dataset: MNIST small (`mnist@0.1.0`, `examples/test/datasets/mnist`), input slot
`image` float32 `[B,1,28,28]`, target `target` int64 `[B]`. Batch size used by
checks: 16 from the `train` split.

| Task | Architecture | Trainable parameters |
| --- | --- | --- |
| T1 MLP | Flatten(1,-1) → Linear(784,64) → ReLU → Linear(64,10) → logits; CrossEntropy(logits,target) | 50 890 |
| T2 Row self-attention, 1 head | Flatten(1,2) → `[B,28,28]` (28 row tokens × 28 px) → Linear(28,32)=h; Q,K,V=Linear(32,32)(h); A=softmax(Q·Kᵀ·1/√32, dim −1); h + A·V; Flatten(1,-1) → Linear(896,10) | 13 066 |
| T3 Multi-head with metavariables | as T2 with d-model=32, num-heads=4: each head Q,K,V=Linear(d, d/h), scale 1/√(d/h), heads concatenated on dim −1 | 3d²+312d+10 → 13 066 (d=32), 32 266 (d=64) |

No output projection, no positional encoding, no causal mask (row
classification is bidirectional). Embedding Linear, residual Add, final
Flatten/Linear/CrossEntropy are provided scaffolding in T2 and T3 in both tools.
Default T3 parameter counts equal T2 by construction, so head count is checked
behaviourally, not by parameter count.

## NNM projects

`study/experts/projects/<task>/{start,solution}`, each a complete project with
both MNIST datasets and `mnist` active, `schemaVersion` 2 (T1, T2) or 3 (T3
solution with `manifest.metavariables` `d-model`=32 and `num-heads`=4).

* T1 start: bound Input, Output, Loss Output.
* T2/T3 start: Input → Flatten(1,2) → Embedding Linear(28,32); a residual Add
  with no inputs; Add → Flatten(1,-1) → Classifier Linear(896,10) → Output and
  → Cross Entropy → Loss Output. The Add is deliberately unwired so the start
  shows model problems (an Add with one input is valid and would report "no
  problems"). Expert connects Embedding → Add `in-1` and builds Embedding → … →
  Add `in-2`. Nodes are laid out in one column with room for the attention block.
* T3 solution: metavariables referenced by Embedding `out_features`
  (`$ref d-model`), Classifier `in_features` (`$lua 28 * meta["d-model"]`),
  Horizontal Repeat `times` (`$ref num-heads`, join Concat dim −1) containing
  Q/K/V Linear (`in=$d-model`, `out=$lua params.in_features / meta["num-heads"]`),
  Transpose(−2,−1), MatMul, Scale (`$lua 1 / math.sqrt(meta["d-model"] /
  meta["num-heads"])`), Softmax(−1), MatMul. The inner scope has the Input and
  Output terminals created with the subflow.

## Equivalent primitives in PyTorch

Allowed: `nn.Linear`, `nn.ReLU`/`torch.relu`, `torch.flatten`/`nn.Flatten`,
`@`/`torch.matmul`, `transpose`, multiplication by a scalar, `softmax`,
`torch.cat`, `+`, `F.cross_entropy`/`nn.CrossEntropyLoss`, `nn.ModuleList`,
Python loops; `view`/`reshape` for splitting heads. Forbidden:
`nn.MultiheadAttention`, `F.scaled_dot_product_attention`,
`F.multi_head_attention_forward`, `nn.Transformer*`.

## Automatic checks (identical criteria for both tools)

`tools/expert_study/study_checks.py` (stdlib + torch only, copied next to the
notebooks) defines the PyTorch-side checks; `tools/expert_study/grade.py`
applies the same criteria to NNM projects by building the resolved graph with
`nnmodelling_runtime.GraphModule`.

1. Forward on the fixed batch: logits float `[16,10]`, finite; loss scalar,
   finite, equal (1e-5) to `F.cross_entropy(logits, target)`.
2. Trainable parameter count equals the table.
3. No forbidden module or function (module tree scan plus
   `torch.overrides.TorchFunctionMode` during forward).
4. Attention probe (T2/T3): softmax calls whose last two dims are `(28,28)`
   and whose rows sum to 1 over the last dimension account for exactly
   `num_heads` maps per sample (1 for T2). The runtime Horizontal Repeat runs
   its copies under `torch.func.vmap`, which hides the head axis; `grade.py`
   therefore replaces its forward with an equivalent per-copy loop over the
   same stacked parameters before checking.
5. T3 only: also construct/resolve with (d-model=64, num-heads=8) and
   (32, 2); checks 1–4 must hold with the corresponding counts. NNM additionally
   requires the two metavariables with the exact names and defaults.
6. NNM only: native analysis complete with no problems, via
   `build/qt/study_probe` (`tests/study_probe.c`), which opens a project
   read-only, optionally applies `--set name=value` metavariable overrides in
   memory, and prints `{complete, problems, tensors, parameters,
   metavariables, resolved_model}`. It never saves.

Notebook execution timing: `study_checks.start_session(task)` registers an
IPython `post_run_cell` hook appending `{ts, task, execution_count, success,
error}` JSON lines to `activity/<task>.jsonl` beside the notebook; check
results are appended to the same file. NNM timing uses the existing
[activity log](../contracts/activity-log.md) with one task file per task,
`tools/activity_log/tasks_expert_t{1,2,3}.json` (sub-goals `E1a`–`E3k`). The
T2 and T3 start projects are identical, so state predicates cannot tell tasks
apart in a combined log: `tools/expert_study/split_activity_log.py` first splits
each log at its `project.open` events into per-task folders
(`task1-mlp`, `task2-attention`, `task3-multihead`, plus `unassigned` for records
before any task project opens) and `tools/expert_study/analyse_activity.py`
parses each folder with its own task file. The parser cannot match scopes,
handles or `$lua` text, so those sub-goals are approximated and documented in
the files' `comment` fields.

`study_probe` is built by the core and Qt CMake trees when testing is enabled;
it is not part of the shipped client. Tests: `just test-expert-study`.

## GUI trial (2026-11-19)

The principal solved all three start projects in the real Qt client through
the VNC skill; `grade.py` passed the three resulting projects. Observations
for the study (tooling limits excluded): new nodes are all placed at the
same point and overlap until Arrange; Arrange of a 14-node scope makes cards
and handles very small, and each Arrange resets zoom; the Packages "Add to
graph" button moves after a package is selected (double-click adds
directly); Inspector rows shift when a node gains a successful output or a
binding gains a "Resolved" row, so the field under the pointer changes; the
metavariable form defaults to type boolean with value `null` and only shows
a "Value" label after another type is chosen. Formulas from the sheet
(`$(1/math.sqrt(32))`, `$d-model`, `$(28 * meta["d-model"])`,
`$(params.in_features / meta["num-heads"])`) were accepted and resolved.

## Lab flow

`tools/expert_study/prepare_usb.py OUT` writes `OUT/master/` (start projects,
notebooks, `study_checks.py`, `mnist_small.json`, task sheet PDF when
`pdflatex` exists), `OUT/facilitator/` (solutions) and
`OUT/new_session.py`, which copies `master/` into a fresh participant folder
without touching earlier ones. `grade.py SESSION_DIR` grades the three
projects and notebooks and writes `grades.json`/`grades.csv`.
