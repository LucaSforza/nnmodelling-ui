import csv
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools" / "activity_log"
sys.path.insert(0, str(TOOLS))
import parse_activity_log as pal  # noqa: E402

BASE = datetime(2026, 11, 12, 10, 0, 0)


class LogBuilder:
    """Writes contract-shaped records and keeps the model for snapshots."""

    def __init__(self, session="s1", start=BASE, seq=1):
        self.session, self.now, self.seq, self.n = session, start, seq, 0
        self.lines, self.t = [], 0
        self.nodes, self.edges, self.ds = [], [], {"id": "mnist", "version": "1"}
        self.history, self.future = [], []

    def tick(self, s=2):
        self.now += timedelta(seconds=s)
        self.t += s * 1000

    def emit(self, event, data=None, src="core", ok=True, error=None, dt=2):
        self.tick(dt)
        rec = {"v": 1, "ts": self.now.strftime("%Y-%m-%dT%H:%M:%S.000+01:00"),
               "t": self.t, "session": self.session, "seq": self.seq,
               "src": src, "event": event, "ok": ok}
        if error:
            rec["error"] = error
        rec["data"] = data or {}
        self.seq += 1
        self.lines.append(json.dumps(rec))
        return rec

    def snapshot(self):
        self.emit("model.snapshot", {
            "project": "test", "dirty": True, "active_dataset": self.ds,
            "nodes": json.loads(json.dumps(self.nodes)),
            "edges": json.loads(json.dumps(self.edges))}, dt=0)

    def _save(self):
        self.history.append((json.dumps(self.nodes), json.dumps(self.edges)))
        self.future.clear()

    def start(self):
        self.emit("session.start", {"app": "nn"}, src="ui")

    def add(self, package, params=None):
        self.emit("palette.add", {"package": package}, src="ui")
        self._save()
        self.n += 1
        nid = f"n{self.n}"
        self.nodes.append({"id": nid, "scope": "root", "package": package,
                           "version": "1", "name": package, "params": params or {}})
        self.emit("node.add", {"id": nid, "package": package})
        self.snapshot()
        return nid

    def connect(self, a, b):
        self._save()
        self.n += 1
        self.edges.append({"id": f"e{self.n}", "source": a, "source_handle": "out",
                           "target": b, "target_handle": "in"})
        self.emit("edge.connect", {"source": a, "target": b})
        self.snapshot()

    def fail_connect(self, a, b):
        self.emit("edge.connect", {"source": a, "target": b}, ok=False,
                  error="incompatible")

    def param(self, nid, key, value):
        self._save()
        node = next(n for n in self.nodes if n["id"] == nid)
        node["params"][key] = value
        self.emit("param.set", {"node": nid, "key": key, "new": str(value)})
        self.snapshot()

    def undo(self):
        self.future.append((json.dumps(self.nodes), json.dumps(self.edges)))
        n, e = self.history.pop()
        self.nodes, self.edges = json.loads(n), json.loads(e)
        self.emit("edit.undo")
        self.snapshot()

    def redo(self):
        self.history.append((json.dumps(self.nodes), json.dumps(self.edges)))
        n, e = self.future.pop()
        self.nodes, self.edges = json.loads(n), json.loads(e)
        self.emit("edit.redo")
        self.snapshot()

    def text(self):
        return "\n".join(self.lines) + "\n"


def initial(b):
    b.nodes = [
        {"id": "out", "scope": "root", "package": "core.output", "version": "1",
         "name": "Output", "params": {}},
        {"id": "loss", "scope": "root", "package": "core.loss-output", "version": "1",
         "name": "Loss", "params": {}}]


def walk_all(b, string_params=False, select_dataset=True):
    """Walk the ten tasks; return {task: ts of its completing snapshot}."""
    conv = str if string_params else (lambda v: v)
    at = {}

    def mark(task):
        at[task] = b.lines and json.loads(b.lines[-1])["ts"]

    b.start()
    inp = b.add("core.input")
    mark("T1")
    if select_dataset:
        b.emit("dataset.select", {"id": "mnist", "version": "1"})
        b.snapshot()
    else:
        b.param(inp, "binding", "mnist")
    mark("T2")
    flat = b.add("core.flatten")
    mark("T3")
    b.fail_connect(inp, flat)
    b.connect(inp, flat)
    mark("T4")
    lin = b.add("core.linear")
    mark("T5")
    relu = b.add("core.relu")
    mark("T6")
    b.connect(flat, lin)
    b.param(lin, "in_features", conv(784))
    b.param(lin, "out_features", conv(64))
    mark("T7")
    b.connect(lin, relu)
    mark("T8a")
    lin2 = b.add("core.linear")
    b.connect(relu, lin2)
    mark("T8b")
    b.param(lin2, "in_features", conv(64))
    b.param(lin2, "out_features", conv(10))
    mark("T8c")
    ce = b.add("core.cross-entropy")
    b.connect(lin2, ce)
    b.connect(ce, "loss")
    mark("T9")
    b.connect(lin2, "out")
    mark("T10")
    b.emit("session.end", {"exit_code": 0}, src="ui")
    return at


def run(tmp_path, texts):
    paths = []
    for name, text in texts.items():
        p = tmp_path / f"{name}.jsonl"
        p.write_text(text)
        paths.append(str(p))
    out = tmp_path / "out"
    warnings = []
    tasks = pal.load_tasks(TOOLS / "tasks_bachelor.json")
    ev, ses, tk, quality, parts = pal.process(paths, tasks, 30, warnings.append)
    return tasks, ev, ses, tk, quality, warnings


def task_map(tk, participant="P1"):
    return {r["task"]: r for r in tk if r["participant"] == participant}


def test_all_tasks_complete_in_order(tmp_path):
    b = LogBuilder()
    initial(b)
    at = walk_all(b)
    _, _, _, tk, _, _ = run(tmp_path, {"P1": b.text()})
    rows = task_map(tk)
    assert list(rows) == ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8a",
                          "T8b", "T8c", "T9", "T10"]
    for tid, row in rows.items():
        assert row["completed"], tid
        assert row["completed_at"] == at[tid], tid
        assert row["out_of_order"] is False
    assert rows["T4"]["failed_ops"] == 1
    assert rows["T4"]["elapsed_s"] > 0


def test_string_params_compare_numerically(tmp_path):
    b = LogBuilder()
    initial(b)
    walk_all(b, string_params=True)
    _, _, _, tk, _, _ = run(tmp_path, {"P1": b.text()})
    assert all(r["completed"] for r in tk)


def test_t2_dataset_select_branch(tmp_path):
    b = LogBuilder()
    initial(b)
    b.start()
    b.add("core.input")
    b.emit("dataset.select", {"id": "mnist", "version": "1"})
    _, _, _, tk, _, _ = run(tmp_path, {"P1": b.text()})
    assert task_map(tk)["T2"]["completed"]


def test_t2_requires_explicit_dataset_selection(tmp_path):
    # The study project already has mnist active and Input auto-binds its slot.
    b = LogBuilder()
    initial(b)
    b.start()
    inp = b.add("core.input")
    b.param(inp, "binding", "image")
    _, _, _, tk, _, _ = run(tmp_path, {"P1": b.text()})
    assert not task_map(tk)["T2"]["completed"]
    b.emit("dataset.select", {"id": "mnist", "version": "1"})
    _, _, _, tk, _, _ = run(tmp_path, {"P1": b.text()})
    assert task_map(tk)["T2"]["completed"]


def test_undo_redo_completes_at_first_satisfying_time(tmp_path):
    b = LogBuilder()
    initial(b)
    b.start()
    b.add("core.input")
    flat = b.add("core.flatten")
    first = json.loads(b.lines[-1])["ts"]
    b.undo()
    b.redo()
    _, _, _, tk, _, _ = run(tmp_path, {"P1": b.text()})
    row = task_map(tk)["T3"]
    assert row["completed_at"] == first
    assert row["undos"] == 0  # undo occurs after completion window
    assert flat


def test_undo_counted_and_out_of_order(tmp_path):
    b = LogBuilder()
    initial(b)
    b.start()
    b.add("core.input")
    b.add("core.relu")      # T6 before T3
    b.undo()
    b.redo()
    b.add("core.flatten")
    _, _, _, tk, _, _ = run(tmp_path, {"P1": b.text()})
    rows = task_map(tk)
    assert rows["T6"]["out_of_order"] is True
    assert rows["T3"]["undos"] == 2


def test_bad_lines_gaps_sessions_and_outputs(tmp_path, capsys):
    b1 = LogBuilder("s1")
    initial(b1)
    walk_all(b1)
    # second, earlier-written session appended later in file order but later ts
    b2 = LogBuilder("s2", start=BASE + timedelta(hours=1))
    b2.start()
    b2.add("core.input")
    lines = b1.lines + ["{not json"] + b2.lines
    # seq gap in s2: drop one record
    del lines[-3]
    text = "\n".join(lines) + "\n" + '{"v":1,"ts":"2026-11-12T12:00'  # truncated
    p = tmp_path / "P1.jsonl"
    p.write_text(text)
    out = tmp_path / "res"
    assert pal.main([str(p), "--out", str(out)]) == 0
    err = capsys.readouterr().err
    assert "skipped line" in err and "seq gap" in err
    for name in ("events.csv", "sessions.csv", "tasks.csv", "summary.md"):
        assert (out / name).stat().st_size > 0
    sessions = list(csv.DictReader(open(out / "sessions.csv")))
    assert [s["session"] for s in sessions] == ["1", "2"]
    assert sessions[0]["ended_cleanly"] == "True"
    assert sessions[1]["ended_cleanly"] == "False"
    tasks = list(csv.DictReader(open(out / "tasks.csv")))
    assert all(t["completed"] == "True" for t in tasks)
    events = list(csv.DictReader(open(out / "events.csv")))
    assert events[0]["event"] == "session.start"
    summary = (out / "summary.md").read_text()
    assert "Malformed/skipped lines: 2" in summary
    assert "Seq gaps: 1" in summary
    assert "without session.end: 1" in summary


def test_unknown_version_and_events_ignored(tmp_path):
    b = LogBuilder()
    initial(b)
    b.start()
    b.lines.append(json.dumps({"v": 2, "ts": "x", "session": "s1", "seq": 99,
                               "event": "x", "data": {}}))
    b.emit("future.event", {"zzz": 1})
    _, ev, _, _, quality, _ = run(tmp_path, {"P1": b.text()})
    assert len(quality[0]["bad"]) == 1
    assert any(r["event"] == "future.event" for r in ev)


def test_collect_script(tmp_path):
    src = tmp_path / "log"
    src.write_text("x\n")
    env = {"NNMODELLING_ACTIVITY_LOG": str(src), "PATH": "/usr/bin:/bin"}
    dest = tmp_path / "dest"
    r = subprocess.run(["bash", str(TOOLS / "collect.sh"), str(dest)], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0
    name = r.stdout.strip()
    assert name.startswith("P-") and name.endswith(".jsonl") and len(name) == 24
    assert (dest / name).read_text() == "x\n"
    r = subprocess.run(["bash", str(TOOLS / "collect.sh"), "--reset", str(dest)],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0 and not src.exists()
    r = subprocess.run(["bash", str(TOOLS / "collect.sh"), str(dest)], env=env,
                       capture_output=True, text=True)
    assert r.returncode != 0
