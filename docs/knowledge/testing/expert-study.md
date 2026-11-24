# Expert usability study materials (proposed 2026-11-19, revised 2026-11-20)

Status: PROPOSAL for the research team. Revised after a pilot session
(2026-11-20): the single-head attention task was removed, leaving two tasks.
Study tooling only: no client, model, file-format or backend semantics change.

Experts perform the same two tasks in NNModelling (NNM) and in PyTorch
(Google Colab), in counterbalanced order. One English task sheet serves both tools:
a tool-neutral specification per task plus an "In NNModelling" and an
"In PyTorch" box. No training is performed in either tool; a task ends when
the expert submits, and the submission is graded afterwards by the automatic
checks. SUS and NASA-TLX are owned by the
research team and are not part of these materials.

## Participant ID

Every participant, student or expert, has one numeric ID assigned by the
research team; numbers are never reused across the two groups. An ID is
digits only (outer spaces ignored), canonical form `str(int(id))` without
leading zeros, `0` invalid. The ID is the only identity anywhere: written by
the facilitator in the ID box of the paper task sheet, typed in the Colab
start cell, used by `new_session.py`, `collect.sh` and the analysis, which
takes it from the file or folder name. Artefacts per ID: `<id>.log`
(activity log), `~/nnm-expert-study/<id>/` (expert session folder),
`nnm-study-<id>-<task>-<UTC>-<attempt>.json` (Colab submission, field
`participant`). `study_checks.participant_id` implements the rule in Python;
`collect.sh` and `new_session.py` repeat it in bash and stdlib Python.

## Fixed data and dimensions

Dataset: MNIST small (`mnist@0.1.0`, `examples/test/datasets/mnist`), input slot
`image` float32 `[B,1,28,28]`, target `target` int64 `[B]`. Batch size used by
checks: 16 from the `train` split.

| Task | Architecture | Trainable parameters |
| --- | --- | --- |
| T1 MLP | Flatten(1,-1) → Linear(784,64) → ReLU → Linear(64,10) → logits; CrossEntropy(logits,target) | 50 890 |
| T2 Multi-head row self-attention with metavariables | Flatten(1,2) → `[B,28,28]` (28 row tokens × 28 px) → Linear(28,d)=H; per head Q,K,V=Linear(d, d/h)(H), A=softmax(Q·Kᵀ·1/√(d/h), dim −1), A·V; heads concatenated on dim −1; H + concat; Flatten(1,-1) → Linear(28d,10); d-model=32, num-heads=4 while building, 64/8 at the end | 3d²+312d+10 → 13 066 (d=32), 32 266 (d=64) |

No output projection, no positional encoding, no causal mask (row
classification is bidirectional). Embedding Linear, residual Add, final
Flatten/Linear/CrossEntropy are provided scaffolding in T2 in both tools.
The parameter count does not depend on h, so head count is checked
behaviourally.

## NNM projects

`study/experts/projects/<task>/{start,solution}`, each a complete project with
MNIST small only and `mnist` active, `schemaVersion` 2, or 3 for the T2
solution (`manifest.metavariables` `d-model`=64 and `num-heads`=8, the final
values the sheet asks for).

* T1 start: bound Input, Output, Loss Output.
* T2 start: Input → Flatten(1,2) → Embedding Linear(28,32); a residual Add
  with no inputs; Add → Flatten(1,-1) → Classifier Linear(896,10) → Output and
  → Cross Entropy → Loss Output. The Add is deliberately unwired so the start
  shows model problems (an Add with one input is valid and would report "no
  problems"). Expert connects Embedding → Add `in-1` and builds Embedding → … →
  Add `in-2`. Nodes are laid out in one column with room for the attention block.
* T2 solution: metavariables referenced by Embedding `out_features`
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
notebooks) defines the PyTorch-side checks (notebook protocol below);
`tools/expert_study/grade.py`
applies the same criteria to NNM projects by building the resolved graph with
`nnmodelling_runtime.GraphModule`.

1. Forward on the fixed batch: logits float `[16,10]`, finite; loss scalar,
   finite, equal (1e-5) to `F.cross_entropy(logits, target)`.
2. Trainable parameter count equals the table.
3. No forbidden module or function (module tree scan plus
   `torch.overrides.TorchFunctionMode` during forward).
4. Attention probe (T2): softmax calls whose last two dims are `(28,28)`
   and whose rows sum to 1 over the last dimension account for exactly
   `num_heads` maps per sample. The runtime Horizontal Repeat runs
   its copies under `torch.func.vmap`, which hides the head axis; `grade.py`
   therefore replaces its forward with an equivalent per-copy loop over the
   same stacked parameters before checking.
5. Scale (T2): in every head the softmax input equals the score
   `Q·Kᵀ` (the `(28,28)` matmul output) times `1/√(d/h)`; scalar factors
   applied to the matmul operands (pre-scaled Q and/or K, also through
   head-split views) count towards the effective factor.
6. Variants (T2): build/resolve with (d-model, num-heads) = (32,4), (64,8)
   and (32,2) through `--set` overrides, whatever values were saved; checks
   1–5 must hold for each. NNM additionally requires metavariables named
   exactly `d-model` and `num-heads`, of type `integer` (or `number` with an
   integral value). The sheet asks to finish with 64/8; saved values 64/8 or
   32/4 both pass and are reported as an informational item; other saved
   values fail.
7. NNM only: native analysis complete with no problems, via
   `build/qt/study_probe` (`tests/study_probe.c`), which opens a project
   read-only, optionally applies `--set name=value` metavariable overrides in
   memory, and prints `{complete, problems, tensors, parameters,
   metavariables, resolved_model}`. It never saves.

Notebook timing comes from the submission file (protocol below). NNM timing uses the existing
[activity log](../contracts/activity-log.md) with one task file per task,
`tools/activity_log/tasks_expert_t{1,2}.json` (sub-goals `E1a`–`E2k`).
State predicates of different tasks can match each other's graphs, so
`tools/expert_study/split_activity_log.py` first splits each log at its
`project.open` events into per-task folders (`task1-mlp`, `task2-multihead`,
plus `unassigned` for records before any task project opens) and `tools/expert_study/analyse_activity.py`
parses each folder with its own task file. The parser cannot match scopes,
handles or `$lua` text, so those sub-goals are approximated and documented in
the files' `comment` fields. `tools/expert_study/plot_session.py` (needs
matplotlib) draws one participant's per-task timeline, time split, Model
Problems curve and action counts.

`study_probe` is built by the core and Qt CMake trees when testing is enabled;
it is not part of the shipped client. Tests: `just test-expert-study`.

## GUI trial (2026-11-19)

The principal solved the start projects (then three tasks) in the real Qt client through
the VNC skill; `grade.py` passed the resulting projects. Observations
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

## PyTorch notebook protocol (Google Colab, accepted 2026-11-24)

Experts run the notebooks in Google Colab in the browser with their own
Google account; there is no local Jupyter path. Each notebook
(`study/experts/notebooks/taskN_*.ipynb`, nbformat 4.5, cell ids mirrored in
`metadata.id`) carries `metadata.colab.generative_ai_disabled: true`
(Colab's per-notebook "Hide generative AI features"); experts also turn off
AI inline completions in Tools → Settings → AI Assistance, an account
setting. Cells, in order:

1. **Start cell** — first, visible, code hidden (`cellView: form`), titled as
   the first thing to run. A form field `participant_id` takes the numeric
   participant ID (see Participant ID). It imports
   `study_checks` from the notebook folder or the repository
   (`tools/expert_study`); if neither has it, it downloads `study_checks.py`
   and `examples/test/datasets/mnist/data.json` (as `mnist_small.json`) from
   the public repository's raw files at the branch holding the study
   materials (`expert-study-materials` until merged, then `main`; links in
   `study/experts/README.md`). It then calls
   `study_checks.begin(task, participant)` and loads the fixed batch `x, y`.
2. **Your model** section — collapsed when the notebook opens
   (`metadata.colab.collapsed_sections`): model class and `compute_loss`.
3. **Try it** section — collapsed: optional, repeatable cell printing logits
   shape, loss and trainable parameter count. This matches what NNM shows
   natively (Inspector shapes, Model → Parameter counts, Model Problems) and
   reveals no check result.
4. **Submit** section — collapsed, large warning: to submit, remove the `#`
   from the single line `study_checks.submit(task, Model, compute_loss)` and
   run the cell once; the answer is final. The line is commented so that
   Run all never submits.

`begin` registers an IPython `post_run_cell` hook appending `{ts, task,
execution_count, success, error}` JSON lines to `activity/<task>.jsonl`
(working directory) and appends `{event: "start", participant}`; re-running
it appends another `start`, the first one counts. An invalid or empty ID
raises an error before anything is logged, so the timer never starts without
an ID. `submit`:

- runs the same checks as `check_task1`/`check_task2` but prints no result
  and does not set `LAST_REPORT`; it prints only that the answer was
  submitted, the attempt number and where the file is saved;
- numbers the attempt from the `submit` events already in the log; attempts
  after the first are recorded as `resubmit` and never graded, so running it
  again cannot improve the result;
- appends `{event: "submit"|"resubmit", attempt, passed, checks}` to the log;
- writes `nnm-study-<id>-<task>-<UTC stamp>-<attempt>.json` (`anon` when
  `begin` never ran) with
  `{format: "nnm-expert-study-submission", version: 1, task, participant,
  attempt, started, submitted, code, report, events}`, where `code` is the
  IPython input history (every executed cell, in order) at submission time
  and `events` the whole log; in Colab it triggers a browser download
  (`google.colab.files.download`) into the expert's Downloads folder,
  elsewhere it only writes the file to the working directory.

`check_task1`/`check_task2` stay public and still print; any call logs a
`check` event, which the grader reports (self-checks before submission).
`study_checks.py` is public on GitHub: hiding the result is a deterrent, not
a security boundary; the log and the code history make deviations visible.

The session folder name is the participant ID; the grader warns when it is
not a canonical numeric ID and fails the check `participant id` of any
submission whose `participant` differs from it (including `anon`), so a file
placed in the wrong folder is never silently attributed.

The grader grades the code the expert submitted, not the notebook state
afterwards: it replays `code` (skipping IPython `%`/`!` lines, with
`submit` disabled) in a subprocess, then calls the task's check on
`MLP`/`MultiHeadRowAttentionClassifier` and `compute_loss`. Per task it
grades the lowest attempt present and reports as information the time from
the first `start` to that submission, later attempts and self-checks. A
`.ipynb` (solutions, tests) is graded the same way from its code cells.

## Task sheet (revised 2026-11-24)

`study/experts/tasks/expert-tasks.tex` follows the Colab protocol:

- One completion rule for both tools: when you think the model is complete
  you submit, and the submission is final; neither tool says whether it is
  correct. PyTorch: remove the `#` from the submit line and run it once.
  NNM: save the project (File → Save) and tell the facilitator; the NNM end
  time is the last `project.save` before that (open: confirm with the
  research team, and whether the facilitator notes the time).
- PyTorch tasks open in Colab from the facilitator's link; the start cell
  runs first (numeric participant ID, timer). The PyTorch box is titled "In PyTorch
  (Google Colab)"; T1 mentions the optional Try it section; T2 says the
  submission is checked with (32,4), (64,8) and (32,2).
- Less guidance (PhD reviewer, 2026-11-24): the sheet no longer lists the
  value of each T2 parameter. It explains the field syntax only — a number,
  `$name` for a variable, `$(...)` for a Python expression where
  `meta["name"]` is a variable's value and `math` is available — with a
  generic example, and states that every parameter depending on d or h must
  use the variables. Translating the specification into values is part of
  the task, as writing the code is in PyTorch.
- An ID box ("Participant ID: ____") at the top, filled in by the
  facilitator; the Colab start cell asks for that number.

The student sheet (`study/students/tasks/student-tasks.tex`, bilingual,
T1–T12 of the guided MLP) has the same bilingual ID box; students never type
the ID, the facilitator uses it with `collect.sh`.

## Lab flow and data collection (revised 2026-11-24)

The lab network is unknown, so collection relies only on the USB stick (or a
shared drive): no network transfer, one command per participant on the lab
machine, one command at home.

`tools/expert_study/prepare_usb.py OUT` writes `OUT/README.md` (exact
commands for each time the stick is plugged in), `OUT/master/` (start
projects, `notebooks/` with the two start notebooks as Colab fallback and a
README with the Colab links, task sheet PDF when `pdflatex` exists),
`OUT/facilitator/` (solutions; never on the lab's shared drive),
`OUT/new_session.py ID` (copies `master/` into `~/nnm-expert-study/<id>/`
without touching earlier ones and prints the commands for that ID),
`OUT/collect.py`, `OUT/collect.sh` (log only, for ssh use) and an empty
`OUT/data/`.

`collect.py ID student|expert [--home DIR] [--keep]`
(`tools/expert_study/collect_participant.py`, standard library only, ID
rule repeated locally) runs on the lab machine at the end of a
participant's activity and writes next to itself
`data/<group>/<id>/`:

- `<id>.log` from `~/nnmodelling.log` (or `NNMODELLING_ACTIVITY_LOG`);
- experts only: `projects/` from `~/nnm-expert-study/<id>/projects/` and
  `colab/nnm-study-<id>-*.json` found in `~/Scaricati`, `~/Downloads`, the
  XDG download folder and `~/nnm-expert-study/<id>/notebooks/`.

It never overwrites an existing `data/<group>/<id>/`, prints what was found
and what is missing (log, project per task, submission per task), lists
submissions of other IDs on the machine without copying them, and, unless
`--keep`, moves the log to `~/nnmodelling.log.collected-<stamp>` and the
collected submissions to `<download>/nnm-study-collected/` so the next
participant on that machine starts clean. Missing items are warnings: what
exists is always collected.

`tools/expert_study/import_usb.py DATA DEST [--participants CSV] [--grade]`
runs on the researcher's computer: it copies `DATA/<group>/<id>/` into
`DEST/<group>/<id>/`, skipping identical files and stopping on a differing
one; checks numeric IDs, that no ID appears in both groups and, with
`--participants` (CSV with columns `id,group`), missing or unexpected IDs
and group mismatches. With `--grade` it writes `DEST/grades.csv`/`.json`
(`grade.py` on every expert folder; a folder's `colab/` counts as its
submission folder, like `notebooks/`) and `DEST/analysis/students/` (activity
log parser on the student logs).

`tools/expert_study/report_all.py DEST` (needs matplotlib and pdflatex with
pdfpages; `uv run --with matplotlib python …`) writes `DEST/report/results.pdf`:
participants table (what was collected per ID, against `participants.csv`),
expert grades with failed checks and timing notes, each expert's NNM activity
figures (`plot_session.py`) and the student report (`paper_report.py`). It
reuses `grades.csv` and `analysis/students/` when present and only writes
under `DEST/report/` otherwise. The stick's `README.md` lists every command.

## Pilot session (2026-11-20)

One colleague performed the then three NNM tasks (8.4 / 19.7 / 20.8 min,
about 50 min in total, twice the estimate). T1 and T2 passed; T3 failed:
num-heads left at 8 after trying 64/8, scale `√(d/h)` instead of
`1/√(d/h)` (not detected by the grader at the time), variable names `D`/`H`,
and type `number` (wrongly rejected at the time). All four T3 errors shown by
the client concerned binding syntax (`H`, `$$H`, `$D/$H`). T2 needed 86 zoom
actions. Report, figures and the colleague's saved projects:
`study/experts/report/pilot/`. Resulting changes: single-head task removed;
the sheet gives navigation tips (arrow keys, group selection and move,
Ctrl +/−, Fit, Arrange) with a
toolbar screenshot, shows the Enter subflow button, lists every parameter
value to enter (removed again on 2026-11-24, see Task sheet), explains Scale
and asks for final values 64/8; the grader
accepts `number` metavariables and checks the scale factor.
