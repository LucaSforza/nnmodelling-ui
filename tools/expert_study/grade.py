#!/usr/bin/env python3
"""Grade expert-study artefacts (NNM projects and PyTorch notebooks).

Usage: grade.py [--probe PATH] [--core DIR] [--task T] [--json OUT] [--csv OUT] TARGET...

TARGET is a session directory (projects/<task>/ and the Colab submission files
nnm-study-*.json in notebooks/, as made by prepare_usb.py), one NNM project
directory, one submission file or one .ipynb. The criteria are
those of docs/knowledge/testing/expert-study.md; study_checks.check_module
applies them to the resolved graph of an NNM project. Run it with an
interpreter that has torch and nnmodelling_runtime installed. Broken
participant artefacts become failed rows, never crashes.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import study_checks as sc  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

MANIFEST_TASKS = {"expert-task1-mlp": "T1", "expert-task2-multihead": "T2"}
FIELDS = ["participant", "tool", "task", "target", "passed", "failed_checks", "detail", "info"]
NOTEBOOK_TIMEOUT = 120
PROBE_TIMEOUT = 120
METAVARIABLES = ("d-model", "num-heads")
SAVED_VALUES = ((32, 4), (64, 8))  # the sheet asks for 64/8 but 32/4 is accepted
HARNESS = '''
import json, sys
spec_path, out_path = sys.argv[1], sys.argv[2]
with open(spec_path, encoding="utf-8") as handle:
    spec = json.load(handle)
import study_checks
study_checks.submit = lambda *args, **kwargs: None  # replaying never submits again
namespace = {"__name__": "__main__"}
errors = []
for index, source in enumerate(spec["sources"]):
    source = "\\n".join("" if line.lstrip().startswith(("%", "!")) else line for line in source.splitlines())
    try:
        exec(compile(source, f"<cell {index}>", "exec"), namespace)
    except BaseException as exc:  # participant code
        errors.append(f"cell {index}: {type(exc).__name__}: {exc}")
report = None
missing = [name for name in (spec["model"], "compute_loss") if name not in namespace]
if missing:
    errors.append("not defined: " + ", ".join(missing))
else:
    try:
        report = study_checks.evaluate(spec["task"], namespace[spec["model"]], namespace["compute_loss"])
    except Exception as exc:
        errors.append(f"study_checks: {type(exc).__name__}: {exc}")
with open(out_path, "w", encoding="utf-8") as handle:
    json.dump({"report": report.to_dict() if report is not None else None, "errors": errors}, handle)
'''
MODEL_CLASSES = {"T1": "MLP", "T2": "MultiHeadRowAttentionClassifier"}


def check(name, passed, detail="", info=False):
    """info=True marks a note that never fails the task but is reported in JSON/CSV."""
    return {"name": name, "passed": bool(passed), "detail": str(detail), **({"info": True} if info else {})}


def row(participant, tool, task, target, checks):
    failed = [c["name"] for c in checks if not c["passed"]]
    detail = "; ".join(f'{c["name"]}: {c["detail"]}' for c in checks if not c["passed"])
    return {
        "participant": participant, "tool": tool, "task": task, "target": str(target),
        "passed": not failed and bool(checks), "failed_checks": failed,
        "detail": detail[:600], "checks": checks,
        "info": "; ".join(c["detail"] for c in checks if c.get("info")),
    }


def error_row(participant, tool, task, target, name, detail):
    return row(participant, tool, task, target, [check(name, False, detail)])


def find_probe(explicit):
    if explicit:
        return Path(explicit)
    for build in ("build/qt", "build/core", "build"):
        path = ROOT / build / "study_probe"
        if path.is_file():
            return path
    return ROOT / "build/qt/study_probe"


# --- NNM projects -------------------------------------------------------------


def run_probe(probe, core, project, overrides=()):
    """Return (probe JSON or None, error text)."""
    command = [str(probe), "--core", str(core)]
    for item in overrides:
        command += ["--set", item]
    command.append(str(project))
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=PROBE_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"probe could not run: {exc}"
    try:
        data = json.loads(done.stdout)
    except ValueError:
        return None, f"probe printed no JSON (exit {done.returncode}): {done.stderr.strip()[:200]}"
    if isinstance(data, dict) and "error" in data:
        return None, str(data["error"])
    if not isinstance(data, dict):
        return None, "probe output is not an object"
    return data, ""


class _GraphAdapter(nn.Module):
    """x -> logits from the root Output terminal; loss from the Loss Output."""

    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self._last_input = None

    def forward(self, x):
        self._last_input = x
        outputs = self.graph(x)
        if "prediction" not in outputs:
            raise ValueError("graph has no connected root Output terminal")
        return outputs["prediction"]

    def loss(self, logits, target):
        outputs = self.graph(self._last_input, {"target": target}, include_loss=True)
        if "loss" not in outputs:
            raise ValueError("graph has no connected Loss Output terminal")
        return outputs["loss"]


def _unroll_horizontal_repeats(graph):
    """Run Horizontal Repeat heads one by one instead of under vmap.

    The stereotype stacks head parameters and applies vmap, which hides the head
    axis from a TorchFunctionMode (one softmax call, and allclose is not
    vmap-compatible). The loop below computes the same function with the same
    parameters, so study_checks can count one 28x28 softmax map per head.
    """
    from torch.func import functional_call

    def unrolled(self, x):
        params, buffers = self._params(), self._stacked_buffers()
        outputs = [
            functional_call(self._base, ({k: v[i] for k, v in params.items()},
                                         {k: v[i] for k, v in buffers.items()}), (x,))
            for i in range(self.count)
        ]
        return self.join(*outputs)

    for module in graph.modules():
        if type(module).__name__ == "HorizontalRepeat" and hasattr(module, "_base"):
            module.forward = types.MethodType(unrolled, module)
    return graph


def graph_checks(project, resolved, core, expected_params, expected_heads, head_width, prefix=""):
    """Build GraphModule from the resolved model in a temporary copy, then run check_module."""
    from nnmodelling_runtime import GraphModule

    if not isinstance(resolved, dict):
        return [check(prefix + "graph", False, "no resolved model")]
    with tempfile.TemporaryDirectory(prefix="nnm-grade-") as temporary:
        copy = Path(temporary) / "project"
        try:
            shutil.copytree(project, copy, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
            (copy / "model.json").write_text(json.dumps(resolved), encoding="utf-8")
            adapter = _GraphAdapter(_unroll_horizontal_repeats(GraphModule(copy, core)))
        except Exception as exc:  # noqa: BLE001 - participant artefact
            return [check(prefix + "graph", False, f"{type(exc).__name__}: {exc}")]
        report = sc.check_module(
            "", adapter, adapter.loss, expected_params=expected_params,
            expected_heads=expected_heads, head_width=head_width,
        )
    return [check(prefix + item.name, item.passed, item.detail) for item in report.checks]


def analysis_check(probe_data, prefix=""):
    problems = probe_data.get("problems") or []
    ok = probe_data.get("complete") is True and not problems
    if ok:
        return check(prefix + "native analysis", True, "complete, no problems")
    messages = "; ".join(f'{p.get("node")}: {p.get("message")}' for p in problems[:3])
    return check(prefix + "native analysis", False, f"incomplete or {len(problems)} problem(s): {messages}")


def project_task(project, override):
    if override:
        return override
    try:
        manifest = json.loads((Path(project) / "model.json").read_text(encoding="utf-8"))["manifest"]
        task = MANIFEST_TASKS.get(manifest.get("id"))
        if task:
            return task
    except (OSError, ValueError, KeyError, TypeError):
        pass
    match = re.search(r"task([12])", Path(project).name)
    return f"T{match.group(1)}" if match else None


def configs_for(task):
    """(prefix, --set overrides, expected params, expected heads, head width) per graded variant."""
    if task == "T2":  # every variant goes through overrides, whatever values were saved
        return [(f"d={d},h={h}: ", [f"d-model={d}", f"num-heads={h}"], sc.expected_task2_params(d), h, d // h)
                for d, h in sc.EXPECTED["T2"]["configs"]]
    return [("", [], sc.EXPECTED[task]["params"], sc.EXPECTED[task]["heads"], None)]


def metavariables_checks(variables):
    """(checks, usable): exact names, integral type, saved values 32/4 or 64/8 (informational)."""
    found = {v.get("name"): v for v in variables or [] if isinstance(v, dict)}
    problems, values = [], []
    for name in METAVARIABLES:
        item = found.get(name)
        value = None if item is None else item.get("value")
        if item is None:
            problems.append(f"missing metavariable '{name}'")
        elif item.get("type") == "integer" and type(value) is int:
            values.append(value)
        elif item.get("type") == "number" and type(value) in (int, float) and float(value).is_integer():
            values.append(int(value))
        else:
            problems.append(f"'{name}' must be an integer, found {item.get('type')} {value!r}")
    usable = not problems
    if usable and tuple(values) not in SAVED_VALUES:
        problems.append(f"saved values {values[0]}/{values[1]}, expected 32/4 or 64/8")
    checks = [check("metavariables", not problems, "; ".join(problems) or "d-model, num-heads")]
    if not problems:
        note = "final values 64/8 as requested" if tuple(values) == (64, 8) else "final values 32/4 (sheet asks 64/8)"
        checks.append(check("final values", True, note, info=True))
    return checks, usable


def grade_project(project, probe, core, participant="-", task_override=None):
    project = Path(project)
    task = project_task(project, task_override)
    if task not in sc.EXPECTED:
        return error_row(participant, "nnm", task or "?", project, "task", "cannot infer task (use --task)")
    checks = []
    try:
        base, error = run_probe(probe, core, project)  # the project as saved
        if base is None:
            return error_row(participant, "nnm", task, project, "probe", error)
        checks.append(analysis_check(base))
        usable = True
        if task == "T2":
            found, usable = metavariables_checks(base.get("metavariables"))
            checks.extend(found)
        for prefix, overrides, params, heads, width in configs_for(task):
            if not usable:
                checks.append(check(prefix + "variant", False, "skipped: metavariables missing or wrong"))
                continue
            data, error = (base, "") if not overrides else run_probe(probe, core, project, overrides)
            if data is None:
                checks.append(check(prefix + "probe", False, error))
                continue
            if overrides:
                checks.append(analysis_check(data, prefix))
            resolved = data.get("resolved_model")
            if resolved is None and data.get("resolved_error"):
                checks.append(check(prefix + "resolved model", False, data["resolved_error"]))
                continue
            checks.extend(graph_checks(project, resolved, core, params, heads, width, prefix))
    except Exception as exc:  # noqa: BLE001 - never crash on a participant artefact
        checks.append(check("grader", False, f"{type(exc).__name__}: {exc}"))
    return row(participant, "nnm", task, project, checks)


# --- notebooks ----------------------------------------------------------------


def notebook_task(path, override):
    if override:
        return override
    match = re.search(r"task([12])", Path(path).name)
    return f"T{match.group(1)}" if match else None


def run_sources(sources, task):
    """Replay code in a fresh interpreter and check it: (report dict or None, errors, failure text)."""
    data_env = ROOT / "examples/test/datasets/mnist/data.json"
    with tempfile.TemporaryDirectory(prefix="nnm-grade-") as temporary:
        harness, spec, output = (Path(temporary) / name for name in ("harness.py", "spec.json", "report.json"))
        harness.write_text(HARNESS, encoding="utf-8")
        spec.write_text(json.dumps({"task": task, "model": MODEL_CLASSES[task], "sources": sources}), encoding="utf-8")
        work = Path(temporary) / "work"
        work.mkdir()
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(HERE), env.get("PYTHONPATH", "")]))
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["HF_HUB_OFFLINE"] = env["TRANSFORMERS_OFFLINE"] = "1"
        if data_env.is_file():
            env["NNM_STUDY_DATA"] = str(data_env)
        try:
            done = subprocess.run(
                [sys.executable, str(harness), str(spec), str(output)], cwd=work,
                env=env, capture_output=True, text=True, timeout=NOTEBOOK_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return None, [], f"timeout after {NOTEBOOK_TIMEOUT} s"
        except OSError as exc:
            return None, [], str(exc)
        try:
            result = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            tail = (done.stderr or done.stdout).strip()[-300:]
            return None, [], f"code did not complete: {tail}"
    return result.get("report"), result.get("errors") or [], ""


def code_row(participant, task, target, sources, extra=()):
    if task not in MODEL_CLASSES:
        return error_row(participant, "pytorch", task or "?", target, "task", "cannot infer task (use --task)")
    report, errors, failure = run_sources(sources, task)
    if failure:
        return error_row(participant, "pytorch", task, target, "execution", failure)
    if report is None:
        detail = "no check report produced" + (f" ({'; '.join(errors)[:300]})" if errors else "")
        return row(participant, "pytorch", task, target, [check("check", False, detail), *extra])
    checks = [check(item["name"], item["passed"], item.get("detail", "")) for item in report.get("checks", [])]
    if not checks:
        checks = [check("check", report.get("passed", False), "empty report")]
    if errors:
        checks.append(check("cells", report.get("passed", False), "; ".join(errors)[:300]))
    if not report.get("passed", False) and all(c["passed"] for c in checks):
        checks.append(check("report", False, "check reported failure"))
    return row(participant, "pytorch", report.get("task") or task, target, checks + list(extra))


def grade_notebook(path, participant="-", task_override=None):
    """A notebook (solutions, tests): its code cells in order."""
    path = Path(path).resolve()
    task = notebook_task(path, task_override)
    try:
        cells = json.loads(path.read_text(encoding="utf-8")).get("cells", [])
        sources = ["".join(c.get("source", "")) if isinstance(c.get("source"), list) else c.get("source", "")
                   for c in cells if c.get("cell_type") == "code"]
    except (OSError, ValueError, AttributeError) as exc:
        return error_row(participant, "pytorch", task or "?", path, "notebook", f"unreadable: {exc}")
    return code_row(participant, task, path, sources)


def canonical_id(value):
    """Numeric participant ID in canonical form, or None (docs/knowledge/testing/expert-study.md)."""
    try:
        return sc.participant_id(value)
    except ValueError:
        return None


def read_submission(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("format") == sc.SUBMISSION_FORMAT else None


def _minutes(start, end):
    from datetime import datetime

    try:
        return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 60
    except (TypeError, ValueError):
        return None


def grade_submission(path, participant=None, task_override=None, later=0):
    """A Colab submission file: the code history at submission time.

    With `participant` (the session folder's ID) the file's own participant must match it.
    """
    path = Path(path).resolve()
    data = read_submission(path)
    if data is None:
        return error_row(participant or "-", "pytorch", task_override or "?", path, "submission", "not a submission file")
    task = task_override or data.get("task")
    declared = str(data.get("participant") or "")
    id_check = []
    if participant is not None:
        folder_id = canonical_id(participant)
        same = folder_id is not None and canonical_id(declared) == folder_id
        id_check = [check("participant id", same, f"file says {declared or 'anon'!r}, folder is {participant!r}")]
    participant = participant or declared or "-"
    events = data.get("events") or []
    submitted = data.get("submitted")
    self_checks = sum(1 for e in events if e.get("event") == "check" and (e.get("ts") or "") < (submitted or ""))
    minutes = _minutes(data.get("started"), submitted)
    extra = [
        check("time", True, "no start event" if minutes is None else f"{minutes:.1f} min from start to submission", info=True),
        check("attempt", True, f"attempt {data.get('attempt')} graded"
              + (f", {later} later attempt(s) ignored" if later else ""), info=True),
    ]
    if self_checks:
        extra.append(check("self-checks", True, f"{self_checks} check call(s) before submission", info=True))
    return code_row(participant, task, path, [str(s) for s in data.get("code") or []], id_check + extra)


def grade_submissions(folder, participant, task_override=None):
    """Per task, the lowest attempt present; later attempts are counted, never graded."""
    by_task = {}
    for path in sorted(Path(folder).glob("*.json")):
        data = read_submission(path)
        if data is not None:
            by_task.setdefault(task_override or data.get("task"), []).append((data.get("attempt") or 0, data.get("submitted") or "", path))
    rows = []
    for task, found in sorted(by_task.items(), key=lambda item: str(item[0])):
        found.sort()
        later = max(attempt for attempt, _, _ in found) - found[0][0]
        rows.append(grade_submission(found[0][2], participant, task, later))
    return rows


# --- targets ------------------------------------------------------------------


def is_project(path):
    return (path / "model.json").is_file()


def grade_session(session, probe, core, task_override=None):
    participant = session.name
    if canonical_id(participant) != participant:
        print(f"warning: session folder {session} is not named by a numeric participant ID", file=sys.stderr)
    rows, projects, notebooks = [], session / "projects", session / "notebooks"
    seen = set()
    for project in sorted(projects.iterdir()) if projects.is_dir() else []:
        if project.is_dir() and is_project(project):
            result = grade_project(project, probe, core, participant, task_override)
            seen.add(("nnm", result["task"]))
            rows.append(result)
    found = grade_submissions(notebooks, participant, task_override) if notebooks.is_dir() else []
    if not found:  # no Colab submissions: notebooks copied in by hand (solutions, tests)
        folder = notebooks if any(notebooks.glob("*.ipynb")) else notebooks / "solutions"
        found = [grade_notebook(notebook, participant, task_override)
                 for notebook in (sorted(folder.glob("*.ipynb")) if folder.is_dir() else [])]
    for result in found:
        seen.add(("pytorch", result["task"]))
        rows.append(result)
    if not task_override:
        for tool in ("nnm", "pytorch"):
            for task in sorted(sc.EXPECTED):
                if (tool, task) not in seen:
                    rows.append(error_row(participant, tool, task, session, "artefact", "missing"))
    return rows


def grade_target(target, probe, core, task_override=None):
    target = Path(target)
    try:
        if target.is_file() and target.suffix == ".ipynb":
            return [grade_notebook(target, target.parent.name, task_override)]
        if target.is_file() and target.suffix == ".json":
            return [grade_submission(target, None, task_override)]
        if target.is_dir() and is_project(target):
            return [grade_project(target, probe, core, target.parent.name, task_override)]
        if target.is_dir() and ((target / "projects").is_dir() or (target / "notebooks").is_dir()):
            return grade_session(target, probe, core, task_override)
    except Exception as exc:  # noqa: BLE001
        return [error_row(target.name, "?", "?", target, "grader", f"{type(exc).__name__}: {exc}")]
    return [error_row(target.name, "?", "?", target, "target", "not a session, project or notebook")]


def write_outputs(rows, json_path, csv_path):
    if json_path:
        Path(json_path).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            for item in rows:
                writer.writerow({**item, "failed_checks": "|".join(item["failed_checks"])})


def print_table(rows):
    header = ["participant", "tool", "task", "result", "failed checks"]
    table = [header] + [
        [r["participant"], r["tool"], r["task"], "PASS" if r["passed"] else "FAIL", ", ".join(r["failed_checks"])]
        for r in rows
    ]
    widths = [max(len(line[i]) for line in table) for i in range(len(header))]
    for line in table:
        print("  ".join(cell.ljust(width) for cell, width in zip(line, widths)).rstrip())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe", help="study_probe executable (default build/qt/study_probe)")
    parser.add_argument("--core", default=str(ROOT / "stereotype-packages" / "core"))
    parser.add_argument("--task", help="override the inferred task (T1 or T2)")
    parser.add_argument("--json", help="write rows as JSON")
    parser.add_argument("--csv", help="write rows as CSV")
    parser.add_argument("targets", nargs="+", metavar="TARGET")
    args = parser.parse_args(argv)
    task = args.task
    if task:
        task = task.upper() if task.upper().startswith("T") else f"T{task}"
        if task not in sc.EXPECTED:
            parser.error("--task must be T1 or T2")
    probe, core = find_probe(args.probe), Path(args.core).resolve()
    rows = []
    for target in args.targets:
        found = grade_target(target, probe, core, task)
        rows.extend(found)
        if not args.json and not args.csv and Path(target).is_dir() and not is_project(Path(target)):
            write_outputs(found, Path(target) / "grades.json", Path(target) / "grades.csv")
    print_table(rows)
    write_outputs(rows, args.json, args.csv)
    return 0 if rows and all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
