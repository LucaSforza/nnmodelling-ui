"""Per-task splitting and parsing of NNM activity logs for the expert study."""
import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT / "study" / "experts" / "projects"
sys.path.insert(0, str(ROOT / "tools" / "expert_study"))

import analyse_activity as analyse  # noqa: E402
import split_activity_log as split  # noqa: E402

TASKS = split.TASKS
BASE = datetime(2026, 11, 19, 10, 0, 0)


def snapshot(project, variant):
    model = json.loads((PROJECTS / project / variant / "model.json").read_text())
    return {
        "project": model["manifest"]["id"], "dirty": True, "active_dataset": {"id": "mnist", "version": "0.1.0"},
        "nodes": [{"id": n["id"], "scope": n["data"].get("scope", ""), "package": n["data"]["package"]["id"],
                   "version": "0.1.0", "name": n["id"], "params": n["data"].get("params", {})} for n in model["nodes"]],
        "edges": [{"id": e["id"], "source": e["source"], "source_handle": e.get("sourceHandle"),
                   "target": e["target"], "target_handle": e.get("targetHandle")} for e in model["edges"]],
    }


class Log:
    def __init__(self):
        self.lines, self.seq, self.clock = [], {}, 0

    def add(self, session, event, data=None, ok=True, src="core"):
        self.seq[session] = self.seq.get(session, 0) + 1
        self.clock += 2
        ts = (BASE + timedelta(seconds=self.clock)).isoformat()
        self.lines.append(json.dumps({"v": 1, "ts": ts, "t": self.clock * 1000, "session": session,
                                      "seq": self.seq[session], "src": src, "event": event, "ok": ok,
                                      "data": data or {}}))

    def open(self, session, task, ok=True):
        self.add(session, "project.open", {"path": f"/home/p/trial/{task}"}, ok)

    def write(self, path):
        Path(path).write_text("\n".join(self.lines) + "\n")
        return Path(path)


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_split_two_sessions_one_per_task(tmp_path):
    log = Log()
    for index, task in enumerate(TASKS):
        session = f"s{index}"
        log.add(session, "session.start", src="ui")
        log.open(session, task)
        log.add(session, "model.snapshot", snapshot(task, "start"))
    source = log.write(tmp_path / "1.log")
    original = source.read_text()
    counts = split.split_log(source, tmp_path / "out")
    assert counts == {"unassigned": 2, **{t: 2 for t in TASKS}}
    for index, task in enumerate(TASKS):
        found = records(tmp_path / "out" / task / "1.jsonl")
        assert [r["event"] for r in found] == ["project.open", "model.snapshot"]
        assert {r["session"] for r in found} == {f"s{index}"} and [r["seq"] for r in found] == [2, 3]
    assert source.read_text() == original


def test_split_session_opening_two_projects_at_boundary(tmp_path):
    log = Log()
    log.add("s", "session.start", src="ui")
    log.open("s", "task1-mlp")
    log.add("s", "model.snapshot", snapshot("task1-mlp", "start"))
    log.add("s", "project.open", {"path": "/home/p/elsewhere"})  # not a task project: no boundary
    log.open("s", "task2-multihead", ok=False)  # failed open: no boundary
    log.add("s", "edit.undo")
    log.open("s", "task2-multihead")
    log.add("s", "model.snapshot", snapshot("task2-multihead", "start"))
    source = log.write(tmp_path / "2.log")
    with open(source, "a") as handle:
        handle.write("{truncated\n")
    warnings = []
    counts = split.split_log(source, tmp_path / "out", warnings.append)
    assert counts == {"unassigned": 1, "task1-mlp": 5, "task2-multihead": 2}
    assert len(warnings) == 1 and "malformed" in warnings[0]
    first = records(tmp_path / "out/task1-mlp/2.jsonl")
    second = records(tmp_path / "out/task2-multihead/2.jsonl")
    assert [r["seq"] for r in first] == [2, 3, 4, 5, 6]
    assert [r["seq"] for r in second] == [7, 8]
    assert second[0]["data"] == {"path": "/home/p/trial/task2-multihead"} and second[0]["src"] == "core"


def test_each_task_folder_completes_only_its_own_subgoals(tmp_path):
    log = Log()
    for task in TASKS:  # one app session per task project, solved
        log.open(task, task)
        log.add(task, "model.snapshot", snapshot(task, "solution"))
    log.write(tmp_path / "3.log")
    table = analyse.analyse([tmp_path / "3.log"], tmp_path / "out")
    assert {row[0] for row in table} == set(TASKS)
    for number, task in enumerate(TASKS, 1):
        rows = [row for row in table if row[0] == task]
        assert rows and all(row[2].startswith(f"E{number}") for row in rows)
        assert all(row[3] == "True" for row in rows), rows
        with open(tmp_path / "out" / task / "tasks.csv", newline="") as handle:
            assert {r["task"][:2] for r in csv.DictReader(handle)} == {f"E{number}"}


def test_start_projects_complete_nothing(tmp_path):
    log = Log()
    for task in TASKS:
        log.open(task, task)
        log.add(task, "model.snapshot", snapshot(task, "start"))
    log.write(tmp_path / "P04.log")
    table = analyse.analyse([tmp_path / "P04.log"], tmp_path / "out")
    assert table and not any(row[3] == "True" for row in table)


def test_analyse_mixed_session_leaks_nothing_across_boundary(tmp_path):
    log = Log()
    log.open("s", "task1-mlp")
    log.add("s", "model.snapshot", snapshot("task1-mlp", "solution"))
    log.open("s", "task2-multihead")
    log.add("s", "model.snapshot", snapshot("task2-multihead", "start"))
    log.write(tmp_path / "P05.log")
    table = analyse.analyse([tmp_path / "P05.log"], tmp_path / "out")
    done = {(row[0], row[2]) for row in table if row[3] == "True"}
    assert {t for _, t in done} == {"E1a", "E1b", "E1c", "E1d", "E1e", "E1f"}
    assert not any(task == "task2-multihead" for task, _ in done)
