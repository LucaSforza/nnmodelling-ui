"""Expert-study probe, grader, USB preparation and NNM task predicates.

The probe tests need `just test-expert-study` (or a build of `study_probe`).
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "expert_study"
PROJECTS = ROOT / "study" / "experts" / "projects"
NOTEBOOKS = ROOT / "study" / "experts" / "notebooks"
CORE = ROOT / "stereotype-packages" / "core"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(ROOT / "tools" / "activity_log"))

import grade  # noqa: E402
import parse_activity_log as pal  # noqa: E402
import prepare_usb  # noqa: E402

TASKS = ("task1-mlp", "task2-multihead")
PARAMS = {"task1-mlp": 50890, "task2-multihead": 32266}  # the T2 solution saves d-model=64, num-heads=8
SOLUTION = PROJECTS / "task2-multihead" / "solution"
PROBE = next((p for p in (ROOT / "build/qt/study_probe", ROOT / "build/core/study_probe") if p.is_file()), None)
needs_probe = pytest.mark.skipif(
    PROBE is None, reason="study_probe not built (run `just test-expert-study` or build target study_probe)"
)


def run_probe(project, *overrides, check=True):
    command = [str(PROBE), "--core", str(CORE)]
    for item in overrides:
        command += ["--set", item]
    done = subprocess.run(command + [str(project)], capture_output=True, text=True, cwd=ROOT)
    if check:
        assert done.returncode == 0, done.stdout + done.stderr
    return done, json.loads(done.stdout)


def tree_hash(directory):
    digest = hashlib.sha256()
    for path in sorted(Path(directory).rglob("*")):
        digest.update(str(path.relative_to(directory)).encode())
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def project_row(project, **kwargs):
    return grade.grade_project(project, PROBE, CORE, **kwargs)


@pytest.fixture(scope="module")
def graded():
    """Grade the four study projects once per module."""
    return {
        (task, variant): project_row(PROJECTS / task / variant)
        for task in TASKS for variant in ("solution", "start")
    }


# --- probe --------------------------------------------------------------------


@needs_probe
@pytest.mark.parametrize("task", TASKS)
def test_probe_solutions_are_complete(task):
    _, data = run_probe(PROJECTS / task / "solution")
    assert data["complete"] is True and data["problems"] == []
    assert data["parameters"] == {"total": PARAMS[task], "trainable": PARAMS[task], "complete": True}
    assert data["tensors"] and data["resolved_model"]["nodes"]
    names = [v["name"] for v in data["metavariables"]]
    assert names == (["d-model", "num-heads"] if task == "task2-multihead" else [])


@needs_probe
@pytest.mark.parametrize("task", TASKS)
def test_probe_starts_are_incomplete(task):
    _, data = run_probe(PROJECTS / task / "start")
    assert data["complete"] is False and data["problems"]


@needs_probe
def test_probe_overrides_resolve_in_memory_only():
    project = SOLUTION
    before = tree_hash(project)
    _, saved = run_probe(project)
    assert {v["name"]: v["value"] for v in saved["metavariables"]} == {"d-model": 64, "num-heads": 8}
    assert saved["complete"] and saved["parameters"]["total"] == 3 * 64 * 64 + 312 * 64 + 10 == 32266
    nodes = {n["id"]: n["data"]["params"] for n in saved["resolved_model"]["nodes"]}
    assert nodes["embed"]["out_features"] == 64 and nodes["heads"]["times"] == 8
    assert nodes["headsquery"]["out_features"] == 8
    _, narrow = run_probe(project, "d-model=32", "num-heads=2")
    assert narrow["complete"] and narrow["parameters"]["total"] == 13066
    assert {v["name"]: v["value"] for v in narrow["metavariables"]} == {"d-model": 32, "num-heads": 2}
    nodes = {n["id"]: n["data"]["params"] for n in narrow["resolved_model"]["nodes"]}
    assert nodes["embed"]["out_features"] == 32 and nodes["headsquery"]["out_features"] == 16
    assert tree_hash(project) == before


@needs_probe
def test_probe_rejects_bad_input_without_touching_files():
    project = SOLUTION
    before = tree_hash(project)
    done, data = run_probe(project, "nope=1", check=False)
    assert done.returncode == 1 and "nope" in data["error"]
    done, data = run_probe(project, "d-model=abc", check=False)
    assert done.returncode == 1 and "integer" in data["error"]
    done, data = run_probe(project, "d-model", check=False)
    assert done.returncode == 1 and "NAME=VALUE" in data["error"]
    done, data = run_probe(ROOT / "does-not-exist", check=False)
    assert done.returncode == 1 and "error" in data
    assert tree_hash(project) == before


# --- grade.py -----------------------------------------------------------------


@needs_probe
@pytest.mark.parametrize("task", TASKS)
def test_grader_passes_solutions(graded, task):
    result = graded[(task, "solution")]
    assert result["passed"], result["detail"]
    assert result["tool"] == "nnm" and result["task"] == "T" + task[4]
    if task == "task2-multihead":
        assert "attention scale" in " ".join(c["name"] for c in result["checks"])


@needs_probe
@pytest.mark.parametrize("task", TASKS)
def test_grader_fails_starts(graded, task):
    result = graded[(task, "start")]
    assert not result["passed"] and result["failed_checks"]


def edited_solution(tmp_path, edit, name="variant"):
    """Copy the T2 solution and apply edit(model dict) to its model.json."""
    project = tmp_path / name
    shutil.copytree(SOLUTION, project)
    path = project / "model.json"
    model = json.loads(path.read_text())
    edit(model)
    path.write_text(json.dumps(model))
    return project


def node_params(model, node_id):
    return next(n["data"]["params"] for n in model["nodes"] if n["id"] == node_id)


def by_name(result):
    return {c["name"]: c for c in result["checks"]}


VARIANTS = ("d=32,h=4: ", "d=64,h=8: ", "d=32,h=2: ")


@needs_probe
def test_t2_metavariable_variants_are_resolved_by_grader(graded):
    result = graded[("task2-multihead", "solution")]
    checks = by_name(result)
    assert "metavariables" in checks and checks["final values"]["info"] is True
    assert checks["final values"]["detail"] == "final values 64/8 as requested"
    assert result["info"] == "final values 64/8 as requested"
    for prefix in VARIANTS:
        for name in ("native analysis", "attention heads", "attention scale", "parameters"):
            assert checks[prefix + name]["passed"], (prefix + name, checks[prefix + name])


@needs_probe
def test_literal_t2_passes_32_4_but_fails_d_model_64(tmp_path):
    literal = {
        ("embed", "out_features"): 32, ("classifier", "in_features"): 896,
        ("headsquery", "in_features"): 32, ("headskey", "in_features"): 32,
        ("headsvalue", "in_features"): 32, ("headsquery", "out_features"): 8,
        ("headskey", "out_features"): 8, ("headsvalue", "out_features"): 8,
        ("headsscale", "factor"): 8 ** -0.5,
    }

    def edit(model):
        for node_id, key in literal:
            node_params(model, node_id)[key] = literal[(node_id, key)]

    result = project_row(edited_solution(tmp_path, edit))
    checks = by_name(result)
    assert all(c["passed"] for name, c in checks.items() if name.startswith("d=32,h=4: ")), result["detail"]
    assert not checks["d=64,h=8: parameters"]["passed"]
    assert not result["passed"]


@needs_probe
def test_scale_node_with_sqrt_instead_of_inverse_fails_scale_check(tmp_path):
    def edit(model):
        node_params(model, "headsscale")["factor"] = {"$lua": 'math.sqrt(meta["d-model"] / meta["num-heads"])'}

    result = project_row(edited_solution(tmp_path, edit))
    checks = by_name(result)
    assert result["failed_checks"] == [prefix + "attention scale" for prefix in VARIANTS]
    assert "scale 2.828 per head, expected 0.3536 = 1/sqrt(8)" in checks["d=32,h=4: attention scale"]["detail"]
    assert "scale 4 per head, expected 0.25 = 1/sqrt(16)" in checks["d=32,h=2: attention scale"]["detail"]


@needs_probe
def test_metavariables_of_type_number_with_integral_values_pass(tmp_path):
    def edit(model):
        for item in model["manifest"]["metavariables"]:
            item["type"] = "number"
            item["value"] = float(item["value"])

    result = project_row(edited_solution(tmp_path, edit))
    assert result["passed"], result["detail"]
    assert by_name(result)["final values"]["detail"] == "final values 64/8 as requested"


@needs_probe
def test_saved_32_4_passes_with_informational_note(tmp_path):
    def edit(model):
        for item, value in zip(model["manifest"]["metavariables"], (32, 4)):
            item["value"] = value

    result = project_row(edited_solution(tmp_path, edit))
    assert result["passed"], result["detail"]
    assert result["info"] == "final values 32/4 (sheet asks 64/8)"
    csv_path = tmp_path / "g.csv"
    grade.write_outputs([result], None, csv_path)
    assert "final values 32/4 (sheet asks 64/8)" in csv_path.read_text()


@needs_probe
def test_other_saved_values_fail_metavariables_but_variants_are_graded(tmp_path):
    def edit(model):
        for item, value in zip(model["manifest"]["metavariables"], (48, 6)):
            item["value"] = value

    result = project_row(edited_solution(tmp_path, edit))
    checks = by_name(result)
    assert result["failed_checks"] == ["metavariables"] and "48/6" in checks["metavariables"]["detail"]
    assert "final values" not in checks and checks["d=64,h=8: attention scale"]["passed"]


@needs_probe
@pytest.mark.parametrize("kind", ["names", "type"])
def test_wrong_metavariable_names_or_types_fail(tmp_path, kind):
    project = edited_solution(tmp_path, lambda model: None)
    path = project / "model.json"
    if kind == "names":  # a complete, valid model whose metavariables are called D and H
        path.write_text(path.read_text().replace("d-model", "D").replace("num-heads", "H"))
    else:
        model = json.loads(path.read_text())
        model["manifest"]["metavariables"][0].update(type="number", value=64.5)
        path.write_text(json.dumps(model))
    result = project_row(project)
    assert not result["passed"] and "metavariables" in result["failed_checks"]
    assert "final values" not in by_name(result)
    assert not by_name(result)["d=32,h=4: variant"]["passed"]  # skipped: overrides need the exact names


@needs_probe
def test_grader_never_crashes_on_broken_artefacts(tmp_path):
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "model.json").write_text("{not json")
    result = project_row(broken, task_override="T1")
    assert not result["passed"] and result["failed_checks"]
    notebook = tmp_path / "task1_broken.ipynb"
    notebook.write_text("{not json")
    result = grade.grade_notebook(notebook)
    assert not result["passed"] and result["task"] == "T1"
    assert not grade.grade_target(tmp_path / "missing", PROBE, CORE)[0]["passed"]


def test_notebook_timeout_becomes_failed_row(tmp_path, monkeypatch):
    monkeypatch.setattr(grade, "NOTEBOOK_TIMEOUT", 3)
    notebook = tmp_path / "task2_loop.ipynb"
    cell = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": ["while True: pass"]}
    notebook.write_text(json.dumps({"cells": [cell], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}))
    result = grade.grade_notebook(notebook)
    assert not result["passed"] and "timeout" in result["detail"]


def test_grader_cli_writes_json_and_csv(tmp_path):
    notebook = NOTEBOOKS / "solutions" / "task1_mlp.ipynb"
    done = subprocess.run(
        [sys.executable, str(TOOLS / "grade.py"), "--json", str(tmp_path / "g.json"),
         "--csv", str(tmp_path / "g.csv"), str(notebook)],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    rows = json.loads((tmp_path / "g.json").read_text())
    assert [(r["tool"], r["task"], r["passed"]) for r in rows] == [("pytorch", "T1", True)]
    header, line = (tmp_path / "g.csv").read_text().splitlines()
    assert header.startswith("participant,tool,task") and ",True," in line


# --- prepare_usb.py and new_session.py ----------------------------------------


@pytest.fixture(scope="module")
def usb(tmp_path_factory):
    out = tmp_path_factory.mktemp("usb")
    prepare_usb.prepare(out)
    return out


def new_session(usb, dest, participant):
    return subprocess.run(
        [sys.executable, str(usb / "new_session.py"), "--dest", str(dest), participant],
        capture_output=True, text=True,
    )


def test_prepare_usb_layout_and_refusals(usb, tmp_path):
    for task in TASKS:
        assert (usb / "master/projects" / task / "model.json").is_file()
    notebooks = usb / "master/notebooks"
    assert sorted(p.name for p in notebooks.iterdir()) == ["README.txt", "task1_mlp.ipynb", "task2_multihead.ipynb"]
    for name in ("task1_mlp.ipynb", "task2_multihead.ipynb"):  # start notebooks, never the solutions
        assert (notebooks / name).read_bytes() == (NOTEBOOKS / name).read_bytes()
    readme = (notebooks / "README.txt").read_text()
    for name in ("task1_mlp.ipynb", "task2_multihead.ipynb"):
        assert f"https://colab.research.google.com/github/LucaSforza/nnmodelling-ui/blob/" in readme and name in readme
    assert not (usb / "facilitator").exists() and not list(usb.rglob("solution*"))  # solutions stay in the repo
    for task in TASKS:  # start projects, never the solutions
        start = ROOT / "study/experts/projects" / task / "start"
        assert (usb / "master/projects" / task / "model.json").read_bytes() == (start / "model.json").read_bytes()
    assert (usb / "collect.sh").read_text() == (ROOT / "tools/activity_log/collect.sh").read_text()
    assert (usb / "master/tasks/expert-tasks.pdf").is_file() or (usb / "master/tasks/expert-tasks.tex").is_file()
    assert (usb / "collect.py").read_text() == (TOOLS / "collect_participant.py").read_text()
    assert (usb / "data").is_dir()
    guide = (usb / "README.md").read_text()
    for command in ('"$D/new_session.py"', '"$D/collect.py"', "import_usb.py", "report_all.py"):
        assert command in guide, command
    assert not list(usb.rglob("__pycache__"))
    with pytest.raises(SystemExit, match="not empty"):
        prepare_usb.prepare(usb)
    (usb / "data/student/1").mkdir(parents=True)  # collected data survives a rebuild
    (usb / "data/student/1/1.log").write_text("keep me\n")
    prepare_usb.prepare(usb, force=True)
    assert (usb / "data/student/1/1.log").read_text() == "keep me\n"
    (tmp_path / "x").mkdir()
    with pytest.raises(SystemExit, match="not empty"):
        prepare_usb.prepare(tmp_path)


def test_new_session_copies_and_refuses_overwrite(usb, tmp_path):
    done = new_session(usb, tmp_path / "sessions", "001")
    assert done.returncode == 0, done.stderr
    assert "Participant ID: 1" in done.stdout
    for task in TASKS:
        assert str(tmp_path / "sessions/1/projects" / task) in done.stdout
    assert str(tmp_path / "sessions/1/notebooks") in done.stdout
    assert f"python3 {usb / 'collect.py'} 1 expert" in done.stdout
    marker = tmp_path / "sessions/1/projects/task1-mlp/marker.txt"
    marker.write_text("participant work")
    again = new_session(usb, tmp_path / "sessions", "1")
    assert again.returncode != 0 and "refusing" in again.stderr
    assert marker.read_text() == "participant work"
    assert new_session(usb, tmp_path / "sessions", "../escape").returncode != 0
    for bad in ("P02", "0", "mario-rossi-1234567", "../escape"):
        assert new_session(usb, tmp_path / "sessions", bad).returncode != 0, bad
    assert new_session(usb, tmp_path / "sessions", "2").returncode == 0


@needs_probe
def test_session_grading_fresh_fails_and_finished_passes(usb, tmp_path):
    assert new_session(usb, tmp_path, "1").returncode == 0
    assert new_session(usb, tmp_path, "2").returncode == 0
    finished = tmp_path / "2"
    shutil.rmtree(finished / "projects")
    for task in TASKS:
        shutil.copytree(ROOT / "study/experts/projects" / task / "solution", finished / "projects" / task)
    for notebook in NOTEBOOKS.joinpath("solutions").glob("*.ipynb"):
        shutil.copy(notebook, finished / "notebooks" / notebook.name)
    rows = grade.grade_session(finished, PROBE, CORE)
    assert sorted((r["tool"], r["task"]) for r in rows) == sorted(
        (tool, f"T{n}") for tool in ("nnm", "pytorch") for n in (1, 2))
    assert all(r["passed"] for r in rows), [(r["tool"], r["task"], r["detail"]) for r in rows if not r["passed"]]
    fresh_rows = grade.grade_session(tmp_path / "1", PROBE, CORE)
    assert len(fresh_rows) == 4 and not any(r["passed"] for r in fresh_rows)
    shutil.rmtree(finished / "projects/task2-multihead")
    rows = grade.grade_session(finished, PROBE, CORE)
    assert any(r["failed_checks"] == ["artefact"] and r["task"] == "T2" for r in rows)


# --- Colab submission files -----------------------------------------------------


def notebook_code(path):
    cells = json.loads(Path(path).read_text())["cells"]
    return ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]


def write_submission(folder, task, attempt, code, participant="9", minute=10):
    record = {"ts": f"2026-11-24T10:{minute:02d}:00+00:00", "task": task, "event": "submit" if attempt == 1 else "resubmit",
              "attempt": attempt}
    data = {
        "format": grade.sc.SUBMISSION_FORMAT, "version": 1, "task": task, "participant": participant,
        "attempt": attempt, "started": "2026-11-24T10:00:00+00:00", "submitted": record["ts"], "code": code,
        "report": {}, "events": [{"ts": "2026-11-24T10:00:00+00:00", "task": task, "event": "start"},
                                 {"ts": "2026-11-24T10:05:00+00:00", "task": task, "event": "check"}, record],
    }
    path = Path(folder) / f"nnm-study-{task}-{participant}-{attempt}.json"
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize("good_first", [True, False])
def test_only_the_first_submission_is_graded(tmp_path, good_first):
    solution = notebook_code(NOTEBOOKS / "solutions/task1_mlp.ipynb")
    start = notebook_code(NOTEBOOKS / "task1_mlp.ipynb")
    first, second = (solution, start) if good_first else (start, solution)
    write_submission(tmp_path, "T1", 1, first + ['study_checks.submit("T1", MLP, compute_loss)'])
    write_submission(tmp_path, "T1", 2, second, minute=20)
    rows = grade.grade_submissions(tmp_path, "9")
    assert len(rows) == 1 and rows[0]["passed"] is good_first and rows[0]["participant"] == "9"
    assert "10.0 min from start to submission" in rows[0]["info"]
    assert "1 later attempt(s) ignored" in rows[0]["info"] and "1 check call(s)" in rows[0]["info"]


def test_submitted_t2_code_is_replayed_and_graded(tmp_path):
    path = write_submission(tmp_path, "T2", 1, notebook_code(NOTEBOOKS / "solutions/task2_multihead.ipynb"))
    result = grade.grade_target(path, PROBE, CORE)[0]
    assert result["passed"] and result["tool"] == "pytorch" and result["task"] == "T2", result["detail"]
    broken = write_submission(tmp_path, "T2", 1, ["x = 1"], participant="10")
    result = grade.grade_target(broken, PROBE, CORE)[0]
    assert not result["passed"] and "not defined" in result["detail"]


@pytest.mark.parametrize("declared", ["8", "", "anon"])
def test_submission_of_another_participant_fails_the_id_check(tmp_path, declared):
    solution = notebook_code(NOTEBOOKS / "solutions/task1_mlp.ipynb")
    write_submission(tmp_path, "T1", 1, solution, participant=declared)
    result, = grade.grade_submissions(tmp_path, "9")
    assert not result["passed"] and result["failed_checks"] == ["participant id"], result["detail"]
    if declared == "8":  # the same file in folder 8 (any spelling of the ID) passes
        assert grade.grade_submissions(tmp_path, "008")[0]["passed"]


# --- task predicates for the activity-log parser -------------------------------


def snapshot_records(project):
    model = json.loads((project / "model.json").read_text())
    data = {
        "project": model["manifest"]["id"], "dirty": True, "active_dataset": {"id": "mnist", "version": "0.1.0"},
        "nodes": [{"id": n["id"], "scope": n["data"].get("scope", ""), "package": n["data"]["package"]["id"],
                   "version": n["data"]["package"].get("version"), "name": n["id"], "params": n["data"].get("params", {})}
                  for n in model["nodes"]],
        "edges": [{"id": e["id"], "source": e["source"], "source_handle": e.get("sourceHandle"),
                   "target": e["target"], "target_handle": e.get("targetHandle")} for e in model["edges"]],
    }
    return [{"session": "s", "seq": 1, "event": "model.snapshot", "src": "core", "ok": True, "data": data}]


@pytest.mark.parametrize("task", TASKS)
def test_expert_task_predicates(task):
    tasks = pal.load_tasks(ROOT / f"tools/activity_log/tasks_expert_t{task[4]}.json")
    mine = [t for t in tasks if t["id"].startswith("E" + task[4])]
    assert mine
    solved = pal.find_completions(snapshot_records(PROJECTS / task / "solution"), mine)
    assert set(solved) == {t["id"] for t in mine}
    assert pal.find_completions(snapshot_records(PROJECTS / task / "start"), mine) == {}
