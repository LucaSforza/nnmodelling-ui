#!/usr/bin/env python3
"""Generate SYNTHETIC activity logs (schema v1) for five archetypal students.

Usage: simulate_sessions.py OUTDIR [--seed N]

The records follow docs/knowledge/contracts/activity-log.md, but the behaviour
is scripted: use them only to rehearse the analysis pipeline, never as data.
Each file is named P-sim<k>-<archetype>.jsonl (one participant per file).
"""
import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

START = datetime(2026, 11, 20, 9, 30, 0)
SCALE = 2.6  # novices are slower than the nominal gaps below
STEP = {"core.input": "Input", "core.flatten": "Flatten", "core.linear": "Linear",
        "core.relu": "ReLU", "core.cross-entropy": "Cross Entropy",
        "core.sigmoid": "Sigmoid", "core.softmax": "Softmax",
        "core.mse-loss": "MSE Loss", "core.conv2d": "Conv2d"}
ERR = {"occupied": "target handle is occupied",
       "type": "output type is incompatible with terminal",
       "cycle": "edge would create a cycle"}


class Sim:
    def __init__(self, rng, start, session):
        self.rng, self.now, self.t = rng, start, 0
        self.session, self.seq, self.n = session, 1, 0
        self.lines, self.hist, self.fut = [], [], []
        self.nodes = [self._node("loss", "core.loss-output", "Loss Output"),
                      self._node("out", "core.output", "Output")]
        self.edges, self.ds = [], {"id": "mnist", "version": "0.1.0"}

    @staticmethod
    def _node(nid, pkg, name):
        return {"id": nid, "scope": "", "package": pkg, "version": "0.1.0",
                "name": name, "params": {}}

    # -- time and records
    def wait(self, mean, sigma=0.5):
        s = max(0.4, self.rng.lognormvariate(0, sigma) * mean * SCALE)
        self.now += timedelta(seconds=s)
        self.t += int(s * 1000)

    def emit(self, event, data=None, src="core", ok=True, error=None, gap=0.0):
        if gap:
            self.wait(gap)
        else:
            self.now += timedelta(milliseconds=15)
            self.t += 15
        rec = {"v": 1, "ts": self.now.strftime("%Y-%m-%dT%H:%M:%S.")
               + f"{self.now.microsecond // 1000:03d}+01:00",
               "t": self.t, "session": self.session, "seq": self.seq,
               "src": src, "event": event, "ok": ok}
        if error:
            rec["error"] = error
        rec["data"] = data or {}
        self.seq += 1
        self.lines.append(json.dumps(rec, separators=(",", ":")))

    def ui(self, event, data=None, gap=0.0):
        self.emit(event, data, src="ui", gap=gap)

    def snapshot(self):
        self.emit("model.snapshot", {
            "project": "test", "dirty": True, "active_dataset": self.ds,
            "nodes": json.loads(json.dumps(self.nodes)),
            "edges": json.loads(json.dumps(self.edges))})

    def save_hist(self):
        self.hist.append(json.dumps([self.nodes, self.edges]))
        self.fut.clear()

    # -- user level operations
    def start(self):
        self.ui("session.start", {"app": "nnmodelling-ui", "version": "0.1.0",
                                  "platform": "xcb", "screen": [1920, 1080],
                                  "os": "Ubuntu 22.04.5 LTS", "automation": False,
                                  "project_arg": "~/examples/test"})

    def blur(self, away):
        """Looks away (asks the supervisor): app.blur, long gap, app.focus."""
        self.ui("app.blur", gap=2)
        self.ui("app.focus", gap=away)

    def search(self, queries, pick, via="button", browse=()):
        """Palette interaction: type queries, optionally preview others, add."""
        for q in queries:
            self.ui("palette.search", {"query": q}, gap=self.rng.uniform(2, 5))
        for pkg in browse:
            self.ui("palette.select", {"package": pkg, "version": "0.1.0"}, gap=2.5)
        self.ui("palette.select", {"package": pick, "version": "0.1.0"}, gap=1.5)
        self.ui("palette.add", {"package": pick, "version": "0.1.0", "via": via}, gap=1.2)
        self.n += 1
        nid = f"n{self.n}"
        self.save_hist()
        self.nodes.append(self._node(nid, pick, STEP.get(pick, pick)))
        self.emit("node.add", {"id": nid, "package": pick, "version": "0.1.0",
                               "scope": "", "x": self.rng.randint(0, 900),
                               "y": self.rng.randint(0, 600)})
        self.snapshot()
        return nid

    def drag(self, nid):
        self.ui("drag.start", {"nodes": [nid]}, gap=1.5)
        self.emit("node.move", {"id": nid, "x": self.rng.randint(0, 900),
                                "y": self.rng.randint(0, 600)}, gap=0.8)

    def connect(self, a, b, fails=(), cancels=()):
        """Draft edge; each cancel/failure is a retry before the final success."""
        for reason in cancels:
            self.ui("connect.start", {"node": a, "handle": "out", "output": True}, gap=2)
            self.ui("connect.cancel", {"reason": reason}, gap=1.5)
        for kind in fails:
            self.ui("connect.start", {"node": a, "handle": "out", "output": True}, gap=2)
            self.emit("edge.connect", {"source": a, "source_handle": "out",
                                       "target": b, "target_handle": "in"},
                      ok=False, error=ERR[kind], gap=1.2)
            self.ui("error.shown", {"where": "canvas", "message": ERR[kind]}, gap=0.1)
        self.ui("connect.start", {"node": a, "handle": "out", "output": True}, gap=2.5)
        self.save_hist()
        self.n += 1
        self.edges.append({"id": f"e{self.n}", "source": a, "source_handle": "out",
                           "target": b, "target_handle": "in"})
        self.emit("edge.connect", {"id": f"e{self.n}", "source": a, "source_handle": "out",
                                   "target": b, "target_handle": "in"}, gap=1.2)
        self.snapshot()

    def param(self, nid, key, value, think=5):
        node = next(x for x in self.nodes if x["id"] == nid)
        self.ui("selection.change", {"nodes": [nid], "edges": []}, gap=1.5)
        old = node["params"].get(key)
        self.save_hist()
        node["params"][key] = value
        self.emit("param.set", {"node": nid, "key": key,
                                "old": None if old is None else str(old),
                                "new": str(value)}, gap=think)
        self.snapshot()

    def undo(self):
        self.ui("ui.action", {"action": "undo", "via": "shortcut"}, gap=1.5)
        self.fut.append(json.dumps([self.nodes, self.edges]))
        self.nodes, self.edges = json.loads(self.hist.pop())
        self.emit("edit.undo")
        self.snapshot()

    def remove(self, nid):
        node = next(x for x in self.nodes if x["id"] == nid)
        self.ui("selection.change", {"nodes": [nid], "edges": []}, gap=2)
        self.save_hist()
        self.nodes = [x for x in self.nodes if x["id"] != nid]
        self.edges = [e for e in self.edges if nid not in (e["source"], e["target"])]
        self.emit("node.remove", {"id": nid, "package": node["package"]}, gap=1.2)
        self.snapshot()

    def select_dataset(self, via_menu=True, hesitate=3):
        self.ui("ui.action", {"action": "datasets", "via": "menu"}, gap=hesitate)
        self.ui("dialog.open", {"dialog": "dataset_chooser"}, gap=0.4)
        self.ui("dialog.close", {"dialog": "dataset_chooser", "result": "accepted"}, gap=3)
        self.emit("dataset.select", {"id": "mnist", "version": "0.1.0"})
        self.snapshot()

    def problems_check(self, click=False):
        self.ui("problems.toggle", {"expanded": True}, gap=2)
        if click:
            self.ui("problem.click", {"node": "n6", "message": "input features mismatch"},
                    gap=3)
        self.ui("problems.toggle", {"expanded": False}, gap=4)

    def end(self, clean=True):
        if clean:
            self.ui("session.end", {"exit_code": 0}, gap=3)


def uid(rng):
    h = "%032x" % rng.getrandbits(128)
    return f"{h[:8]}-{h[8:12]}-4{h[13:16]}-a{h[17:20]}-{h[20:32]}"


# ------------------------------------------------------------- archetypes
def run_tasks(s, p):
    """Drive the ten tasks; p tunes search vocabulary, errors and hesitation."""
    w = p.get("pace", 1.0)
    ids = {}
    order = p.get("order", "linear")
    s.start()
    s.wait(8 * w)
    ids["in"] = s.search(p.get("q_input", ["input"]), "core.input")
    s.drag(ids["in"]) if p.get("drags") else None
    s.select_dataset(hesitate=p.get("t2_hesitate", 3) * w)
    if order == "front_load":      # places everything before wiring (scattered)
        ids["fl"] = s.search(["flat"], "core.flatten")
        ids["l1"] = s.search(p.get("q_linear", ["linear"]), "core.linear",
                             browse=p.get("browse", ()))
        ids["re"] = s.search(["relu"], "core.relu")
        ids["l2"] = s.search(["linear"], "core.linear", via="double_click")
        ids["ce"] = s.search(["cross"], "core.cross-entropy")
    else:
        ids["fl"] = s.search(p.get("q_flatten", ["flatten"]), "core.flatten")
        s.wait(3 * w)
    s.connect(ids["in"], ids["fl"], **p.get("c_in_fl", {}))
    if order != "front_load":
        if p.get("wrong_act"):    # picks the wrong activation first
            bad = s.search(["activation"], "core.sigmoid", browse=("core.softmax",))
            s.wait(5)
            s.remove(bad)
        ids["l1"] = s.search(p.get("q_linear", ["linear"]), "core.linear",
                             browse=p.get("browse", ()))
        ids["re"] = s.search(["relu"], "core.relu")
    s.drag(ids["l1"]) if p.get("drags") else None
    s.connect(ids["fl"], ids["l1"], **p.get("c_fl_l1", {}))
    for key, val, think in p.get("p_l1", [("in_features", 784, 6),
                                          ("out_features", 64, 5)]):
        s.param(ids["l1"], key, val, think=think * w)
    if p.get("check_problems"):
        s.problems_check(click=True)
    s.connect(ids["l1"], ids["re"], **p.get("c_l1_re", {}))
    if order != "front_load":
        ids["l2"] = s.search(["linear"], "core.linear", via="double_click")
    s.connect(ids["re"], ids["l2"], **p.get("c_re_l2", {}))
    for key, val, think in p.get("p_l2", [("in_features", 64, 5),
                                          ("out_features", 10, 4)]):
        s.param(ids["l2"], key, val, think=think * w)
        if p.get("undo_after") == (key, val):
            s.undo()
            s.param(ids["l2"], key, val, think=3)
    if order != "front_load":
        ids["ce"] = s.search(p.get("q_ce", ["cross entropy"]), "core.cross-entropy")
    s.connect(ids["l2"], ids["ce"], **p.get("c_l2_ce", {}))
    s.connect(ids["ce"], "loss", **p.get("c_ce_loss", {}))
    s.connect(ids["l2"], "out", **p.get("c_l2_out", {}))
    s.wait(2)
    return ids


def archetype_fluent(rng, sid):
    s = Sim(rng, START, sid)
    run_tasks(s, {"pace": 0.55, "t2_hesitate": 1.5})
    s.end()
    return [s]


def archetype_vocab(rng, sid):
    """Searches by concept ('dense', 'activation') before the package name."""
    s = Sim(rng, START + timedelta(minutes=20), sid)
    run_tasks(s, {"pace": 1.0,
                  "q_linear": ["dense", "fully connected", "lin", "linear"],
                  "q_ce": ["loss", "cross", "cross entropy"],
                  "q_flatten": ["reshape", "flat"],
                  "browse": ("core.matmul", "core.scale"),
                  "wrong_act": True})
    s.end()
    return [s]


def archetype_wiring(rng, sid):
    """Struggles with ports: cancelled drags, occupied/incompatible handles."""
    s = Sim(rng, START + timedelta(minutes=45), sid)
    run_tasks(s, {"pace": 1.2, "drags": True,
                  "c_in_fl": {"cancels": ["no_target", "no_target"]},
                  "c_fl_l1": {"cancels": ["no_target"], "fails": ["type"]},
                  "c_l1_re": {"cancels": ["escape"], "fails": ["occupied"]},
                  "c_re_l2": {"fails": ["occupied", "type"], "cancels": ["no_target"]},
                  "c_l2_ce": {"cancels": ["no_target", "no_target", "escape"]},
                  "c_l2_out": {"fails": ["cycle"], "cancels": ["no_target"]}})
    s.end()
    return [s]


def archetype_params(rng, sid):
    """Confuses features, swaps in/out, relies on undo and the problems panel."""
    s = Sim(rng, START + timedelta(minutes=70), sid)
    run_tasks(s, {"pace": 1.4, "check_problems": True,
                  "p_l1": [("in_features", 64, 9), ("out_features", 784, 7),
                           ("in_features", 784, 8), ("out_features", 64, 6)],
                  "p_l2": [("in_features", 10, 8), ("out_features", 64, 7),
                           ("in_features", 64, 6), ("out_features", 10, 5)],
                  "undo_after": ("out_features", 64)})
    s.end()
    return [s]


def archetype_interrupted(rng, sid):
    """Front-loads the nodes, asks the supervisor, app crash and restart."""
    s = Sim(rng, START + timedelta(minutes=95), sid)
    s.start()
    s.wait(10)
    ids = {"in": s.search(["input"], "core.input")}
    s.blur(away=45)                                  # asks the supervisor
    s.select_dataset(hesitate=20)
    for key, pkgs in (("fl", ("flatten", "core.flatten")),
                      ("l1", ("linear", "core.linear")),
                      ("re", ("relu", "core.relu")),
                      ("l2", ("linear", "core.linear")),
                      ("ce", ("cross", "core.cross-entropy"))):
        ids[key] = s.search([pkgs[0]], pkgs[1], via="button")
    s.connect(ids["in"], ids["fl"])
    s.connect(ids["fl"], ids["l1"])
    s.param(ids["l1"], "in_features", 784, think=18)
    s.param(ids["l1"], "out_features", 64, think=8)
    s.blur(away=60)                                 # second long consultation
    # crash: no session.end, new process, same project reopened from disk
    first = s
    nodes, edges = json.loads(json.dumps(s.nodes)), json.loads(json.dumps(s.edges))
    s2 = Sim(rng, s.now + timedelta(seconds=75), uid(rng))
    s2.nodes, s2.edges, s2.n = nodes, edges, s.n
    s2.start()
    s2.ui("dialog.open", {"dialog": "project_open"}, gap=4)
    s2.ui("dialog.close", {"dialog": "project_open", "result": "accepted"}, gap=8)
    s2.emit("project.open", {"path": "~/examples/test"})
    s2.snapshot()
    s2.connect(ids["l1"], ids["re"])
    s2.connect(ids["re"], ids["l2"])
    s2.param(ids["l2"], "in_features", 64, think=6)
    s2.param(ids["l2"], "out_features", 10, think=5)
    s2.connect(ids["l2"], ids["ce"])
    s2.connect(ids["ce"], "loss")
    s2.connect(ids["l2"], "out")
    s2.end()
    return [first, s2]


ARCHETYPES = [("fluent", archetype_fluent), ("vocabulary", archetype_vocab),
              ("wiring", archetype_wiring), ("parameters", archetype_params),
              ("interrupted", archetype_interrupted)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outdir")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for k, (name, fn) in enumerate(ARCHETYPES, 1):
        rng = random.Random(a.seed * 100 + k)
        sims = fn(rng, uid(rng))
        text = "\n".join(line for s in sims for line in s.lines) + "\n"
        (out / f"P-sim{k}-{name}.jsonl").write_text(text, encoding="utf-8")
        print(f"P-sim{k}-{name}.jsonl: {sum(len(s.lines) for s in sims)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
