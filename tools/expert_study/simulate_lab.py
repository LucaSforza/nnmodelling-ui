#!/usr/bin/env python3
"""Rehearse a lab day with scripted participants (simulated data, never study data).

Usage: simulate_lab.py OUT [--speed N] [--only ID,ID...]

Each participant (IDs 9xx, PROFILES below) gets a fake lab machine
OUT/machines/pc-<id>/home and the real NNModelling app (HOME = that home, Qt VNC
platform, --socket) driven by nnmodelctl with realistic pauses. Experts are
prepared with the stick's new_session.py and run the real Colab notebooks in a
local IPython kernel whose stand-in google.colab.files.download saves into that
home's Scaricati/. The T2 metavariables go through the real Manage
metavariables dialog with scripted VNC input (automation has no such
operation). Nothing is collected: the facilitator runs the stick's collect.py
per participant (commands printed at the end). --speed divides every pause
(tests only). Rules: docs/knowledge/testing/expert-study.md (Lab rehearsal).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
APP = ROOT / "build" / "qt" / "nnmodelling-ui"
CTL = ROOT / "build" / "nnmodelctl"
VNC = ROOT / ".agents" / "skills" / "nnmodelling-vnc" / "scripts" / "vnc.py"
LOOPBACK = ROOT / ".agents" / "skills" / "nnmodelling-browser" / "scripts" / "loopback.c"
KERNEL_PYTHON = ROOT / ".venv" / "bin" / "python"
STUDY = ROOT / "study" / "experts"

# --- participants -------------------------------------------------------------------------------------------------

PROFILES = {
    901: {"group": "student", "profile": "fluent: T1-T12 complete", "pace": 1.0},
    902: {"group": "expert", "profile": "NNM then PyTorch; everything correct", "order": "nnm-first",
          "pace": 1.0, "t2_code": "loop", "try_error": True},
    903: {"group": "student", "profile": "wrong activation (Softmax, removed); wrong in_features corrected",
          "pace": 1.2, "wrong_activation": True, "wrong_param": True},
    904: {"group": "expert", "profile": "PyTorch then NNM; NNM T2 keeps 32/4; PyTorch T2 scale 1/sqrt(d)",
          "order": "pytorch-first", "pace": 0.9, "t2_code": "wrong_scale", "skip_final_values": True},
    905: {"group": "student", "profile": "wiring errors: occupied handle, wrong edge disconnected, loss into Output",
          "pace": 1.1, "wiring_errors": True},
    906: {"group": "expert", "profile": "NNM then PyTorch; NNM T2 embedding literal 32; PyTorch T1 submitted twice",
          "order": "nnm-first", "pace": 1.1, "t2_code": "view", "literal_embedding": True, "resubmit_t1": True},
    907: {"group": "student", "profile": "slow; never activates MNIST small", "pace": 1.6, "skip_dataset": True},
    908: {"group": "expert", "profile": "PyTorch then NNM; NNM T2 without residual; PyTorch T2 not submitted",
          "order": "pytorch-first", "pace": 1.0, "t2_code": "loop", "no_residual": True, "no_submit_t2": True},
    909: {"group": "student", "profile": "stops after T9 (incomplete)", "pace": 1.3, "stop_after": 9},
    910: {"group": "student", "profile": "fluent with a long break", "pace": 0.9, "away": 240},
}

# --- PyTorch code typed by the experts (also checked by tests/test_expert_study_simulate.py) -------------------

T1_MODEL = '''class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x):
        return self.net(x)
'''
T1_LOSS = '''def compute_loss(logits, target):
    return F.cross_entropy(logits, target)
'''
_T2_LOOP = '''class Head(nn.Module):
    def __init__(self, d_model, d_head):
        super().__init__()
        self.q = nn.Linear(d_model, d_head)
        self.k = nn.Linear(d_model, d_head)
        self.v = nn.Linear(d_model, d_head)
        self.scale = SCALE

    def forward(self, h):
        a = torch.softmax(self.q(h) @ self.k(h).transpose(-2, -1) * self.scale, dim=-1)
        return a @ self.v(h)


class MultiHeadRowAttentionClassifier(nn.Module):
    def __init__(self, d_model=32, num_heads=4):
        super().__init__()
        self.embed = nn.Linear(28, d_model)  # provided
        self.classifier = nn.Linear(28 * d_model, 10)  # provided
        self.heads = nn.ModuleList(Head(d_model, d_model // num_heads) for _ in range(num_heads))

    def forward(self, x):
        h = self.embed(torch.flatten(x, 1, 2))  # provided: [B, 28, d_model]
        attn = torch.cat([head(h) for head in self.heads], dim=-1)
        h = h + attn  # provided: residual
        return self.classifier(torch.flatten(h, 1))  # provided
'''
T2_CODE = {
    "loop": _T2_LOOP.replace("SCALE", "1 / math.sqrt(d_head)"),
    "wrong_scale": _T2_LOOP.replace("SCALE", "1 / math.sqrt(d_model)"),
    "view": '''class MultiHeadRowAttentionClassifier(nn.Module):
    def __init__(self, d_model=32, num_heads=4):
        super().__init__()
        self.embed = nn.Linear(28, d_model)  # provided
        self.classifier = nn.Linear(28 * d_model, 10)  # provided
        self.h, self.dh = num_heads, d_model // num_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

    def forward(self, x):
        h = self.embed(torch.flatten(x, 1, 2))  # provided: [B, 28, d_model]
        B, T, D = h.shape
        q, k, v = (f(h).view(B, T, self.h, self.dh).transpose(1, 2) for f in (self.q, self.k, self.v))
        a = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.dh), dim=-1)
        attn = (a @ v).transpose(1, 2).reshape(B, T, D)
        h = h + attn  # provided: residual
        return self.classifier(torch.flatten(h, 1))  # provided
'''}
T2_LOSS = '''def compute_loss(logits, target):
    return F.cross_entropy(logits, target)  # provided
'''
EXPECTED_T2 = {"loop": True, "view": True, "wrong_scale": False}  # passes study_checks.evaluate("T2", ...)

# --- helpers -------------------------------------------------------------------------------------------------------


class Participant:
    def __init__(self, pid: int, out: Path, speed: float, seed: int):
        self.pid, self.p, self.speed = pid, PROFILES[pid], speed
        self.rng = random.Random(seed * 1000 + pid)
        self.home = out / "machines" / f"pc-{pid}" / "home"
        self.run = out / "run" / str(pid)
        self.port = 6000 + pid
        self.sock = self.run / "app.sock"
        self.app = None
        self.transcript = (out / "transcripts" / f"{pid}.txt").open("a", encoding="utf-8")

    def say(self, text: str) -> None:
        self.transcript.write(f"{time.strftime('%H:%M:%S')} {text}\n")
        self.transcript.flush()

    def pause(self, mean: float, sigma: float = 0.45) -> None:
        seconds = max(0.5, self.rng.lognormvariate(0, sigma) * mean * self.p.get("pace", 1.0))
        time.sleep(seconds / self.speed)

    # NNModelling ------------------------------------------------------------------------------------------------
    def start_app(self, project: Path) -> None:
        self.run.mkdir(parents=True, exist_ok=True)
        if self.sock.exists():
            self.sock.unlink()
        env = {k: v for k, v in os.environ.items()
               if k not in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME",
                            "NNMODELLING_ACTIVITY_LOG", "NNMODELLING_SOCKET")}
        env.update(HOME=str(self.home), NN_SOURCE_DIR=str(ROOT), LD_PRELOAD=str(self.run.parent / "loopback.so"),
                   QT_QPA_PLATFORM=f"vnc:size=1600x1000:port={self.port}", QT_STYLE_OVERRIDE="Fusion")
        self.app = subprocess.Popen([str(APP), "--socket", str(self.sock), str(project)], env=env,
                                    stdout=(self.run / "app.log").open("a"), stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL)
        for _ in range(150):
            if self.sock.exists() and self.ctl("project.snapshot", quiet=True).get("ok"):
                break
            time.sleep(0.2)
        else:
            raise RuntimeError(f"{self.pid}: app did not start (see {self.run / 'app.log'})")
        self.say(f"app started on {project.name} (VNC port {self.port})")

    def stop_app(self) -> None:
        if self.app and self.app.poll() is None:
            self.app.terminate()
            self.app.wait(timeout=20)
        self.say("app closed")

    def ctl(self, op: str, args: dict | None = None, quiet: bool = False) -> dict:
        done = subprocess.run([str(CTL), "--socket", str(self.sock), op, json.dumps(args or {})],
                              capture_output=True, text=True)
        try:
            reply = json.loads(done.stdout or "{}")
        except ValueError:
            reply = {"ok": False, "error": done.stderr.strip()}
        if not quiet:
            self.say(f"{op} {json.dumps(args or {})} -> " + ("ok" if reply.get("ok") else f"ERROR {reply.get('error')}"))
        return reply

    def add(self, package: str, scope: str = "", think: float = 12) -> str:
        self.pause(think)
        nid = f"node-{uuid.UUID(int=self.rng.getrandbits(128))}"
        self.ctl("node.add", {"id": nid, "package": package, "version": "0.1.0", "scope": scope,
                              "x": self.rng.randint(-200, 200), "y": self.rng.randint(0, 600)})
        return nid

    def edge(self, source: str, target: str, source_handle: str = "out", target_handle: str = "in",
             think: float = 6) -> dict:
        self.pause(think)
        eid = str(uuid.UUID(int=self.rng.getrandbits(128)))
        return self.ctl("edge.connect", {"id": eid, "source": source, "sourceHandle": source_handle,
                                         "target": target, "targetHandle": target_handle}) | {"id": eid}

    def param(self, nid: str, key: str, value, think: float = 7) -> None:
        self.pause(think)
        self.ctl("node.parameter", {"id": nid, "key": key, "value": str(value)})

    def save(self) -> None:
        self.pause(4)
        self.ctl("project.save")

    def vnc(self, *steps) -> None:
        subprocess.run([sys.executable, str(VNC), "--port", str(self.port), *map(str, steps)],
                       capture_output=True, text=True, check=True)

    def metavariables(self, values: dict, new: bool) -> None:
        """Real Manage metavariables dialog (coordinates for a 1600x1000 VNC screen)."""
        self.pause(10)
        self.vnc("click", 116, 18, "wait", 0.6, "click", 198, 144, "wait", 0.8)
        for row, (name, value) in enumerate(values.items()):
            if new:
                self.vnc("click", 418, 513, "wait", 0.6, "click", 685, 268, "type", name, "key", "Tab", "type", "i")
            else:
                self.vnc("click", 600, 198 + 34 * row, "click", 478, 513, "wait", 0.6)
            self.pause(5)
            self.vnc("click", 685, 368, "key", "ctrl+a", "type", value, "wait", 0.3, "click", 668, 419, "wait", 0.5)
        self.pause(3)
        self.vnc("click", 842, 565, "wait", 0.8)
        self.say(f"metavariables dialog: {values} ({'new' if new else 'edit'})")

    # Colab --------------------------------------------------------------------------------------------------------
    def colab(self, task: str, model: str, loss: str, submit: int = 1, try_error: bool = False) -> None:
        from jupyter_client import BlockingKernelClient  # noqa: PLC0415 - simulation-only dependency

        notebook = json.loads((STUDY / "notebooks" / ("task1_mlp.ipynb" if task == "T1" else
                                                       "task2_multihead.ipynb")).read_text(encoding="utf-8"))
        cells = ["".join(c["source"]) for c in notebook["cells"]]
        content = self.run / f"colab-{task}"
        content.mkdir(parents=True, exist_ok=True)
        connection = self.run / f"kernel-{task}.json"
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(self.pylib()),
                   SIM_DOWNLOADS=str(self.home / "Scaricati"), HOME=str(self.home))
        kernel = subprocess.Popen([str(KERNEL_PYTHON), "-m", "ipykernel_launcher", "-f", str(connection)],
                                  cwd=content, env=env, stdout=(self.run / f"kernel-{task}.log").open("w"),
                                  stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        for _ in range(300):
            if connection.exists():
                break
            time.sleep(0.2)
        kc = BlockingKernelClient(connection_file=str(connection))
        kc.load_connection_file()
        kc.start_channels()
        kc.wait_for_ready(timeout=120)
        self.say(f"colab {task}: notebook opened")

        def run(index: int, source: str) -> None:
            reply = kc.execute_interactive(source, store_history=True, timeout=600,
                                           output_hook=lambda m: self._colab_out(task, index, m))
            self.say(f"colab {task} cell [{index}] -> {reply['content']['status']}")

        try:
            self.pause(20)
            run(1, cells[1].replace('participant_id = ""', f'participant_id = "{self.pid}"', 1))
            self.pause(240 if task == "T1" else 480, sigma=0.3)  # reading and typing the model
            run(3, model)
            if try_error:
                self.pause(20)
                run(6, cells[6])  # NameError/NotImplementedError: loss cell not run yet
                self.pause(30)
            self.pause(25)
            run(4, loss)
            self.pause(15)
            run(6, cells[6])
            for attempt in range(submit):
                self.pause(40 if attempt == 0 else 90)
                run(8, cells[8].replace("# study_checks.submit", "study_checks.submit"))
        finally:
            kc.stop_channels()
            kernel.terminate()  # closing the browser tab
            kernel.wait(timeout=20)

    def _colab_out(self, task, index, msg) -> None:
        content = msg.get("content", {})
        text = content.get("text") or content.get("evalue") or ""
        if text:
            self.say(f"colab {task} [{index}] output: {text.strip()[:200]}")

    @staticmethod
    def pylib():
        return [str(OUT_PYLIB), str(OUT_FAKECOLAB)]


OUT_PYLIB = Path()  # set in main
OUT_FAKECOLAB = Path()

# --- scripts -------------------------------------------------------------------------------------------------------


def student(s: Participant) -> None:
    p = s.p
    project = s.home / "test"
    s.start_app(project)
    snap = s.ctl("project.snapshot", quiet=True)["result"]
    terminals = {n["package"]["id"]: n["id"] for n in snap["nodes"]}
    s.pause(30)  # reads the sheet
    inp = s.add("core.input", think=25)                                       # T1
    if not p.get("skip_dataset"):                                              # T2
        s.pause(35)
        s.ctl("dataset.select", {"id": "mnist", "version": "0.1.0"})
    flat = s.add("core.flatten", think=20)                                    # T3
    s.edge(inp, flat, think=12)                                               # T4
    lin1 = s.add("core.linear", think=18)                                     # T5
    if p.get("wrong_activation"):
        bad = s.add("core.softmax", think=25)
        s.pause(30)
        s.ctl("node.remove", {"id": bad})
    relu = s.add("core.relu", think=15)                                       # T6
    if p.get("wiring_errors"):
        wrong = s.edge(inp, lin1, think=10)                                   # wrong source
        s.pause(20)
        s.ctl("edge.disconnect", {"id": wrong["id"]})
    s.edge(flat, lin1, think=10)                                              # T7
    if p.get("wiring_errors"):
        s.edge(flat, lin1, think=8)                                           # retried: occupied
    s.param(lin1, "in_features", 748 if p.get("wrong_param") else 784, think=20)
    s.param(lin1, "out_features", 64, think=10)
    if p.get("wrong_param"):
        s.param(lin1, "in_features", 784, think=40)
    s.edge(lin1, relu, think=10)                                              # T8
    if p.get("away"):
        s.say(f"away for about {p['away']} s")
        s.pause(p["away"], sigma=0.1)
    lin2 = s.add("core.linear", think=15)                                     # T9
    s.edge(relu, lin2, think=10)
    if p.get("stop_after") == 9:
        s.save()
        s.stop_app()
        return
    s.param(lin2, "in_features", 64, think=15)                                # T10
    s.param(lin2, "out_features", 10, think=8)
    ce = s.add("core.cross-entropy", think=20)                                # T11
    s.edge(lin2, ce, think=10)
    if p.get("wiring_errors"):
        s.edge(ce, terminals["core.output"], "loss", "in", think=10)          # loss into Output: rejected
    s.edge(ce, terminals["core.loss-output"], "loss", "in", think=10)
    s.edge(lin2, terminals["core.output"], think=12)                          # T12
    s.save()
    s.pause(15)
    s.stop_app()


def nnm_t1(s: Participant) -> None:
    s.pause(40)
    flat = s.add("core.flatten", think=25)
    s.edge("input", flat)
    l1 = s.add("core.linear", think=15)
    s.edge(flat, l1)
    s.param(l1, "in_features", 784)
    s.param(l1, "out_features", 64)
    relu = s.add("core.relu", think=12)
    s.edge(l1, relu)
    l2 = s.add("core.linear", think=10)
    s.edge(relu, l2)
    s.param(l2, "in_features", 64)
    s.param(l2, "out_features", 10)
    ce = s.add("core.cross-entropy", think=15)
    s.edge(l2, ce)
    s.edge(l2, "output")
    s.edge(ce, "loss-output", "loss", "in")
    s.pause(30)  # checks Model Problems
    s.save()


def nnm_t2(s: Participant) -> None:
    p = s.p
    s.pause(90)  # reads the long task
    s.metavariables({"d-model": "32", "num-heads": "4"}, new=True)
    s.param("embed", "out_features", "32" if p.get("literal_embedding") else "$d-model", think=20)
    s.param("classifier", "in_features", "$(28 * meta[\"d-model\"])", think=40)
    heads = s.add("core.horizontal-repeat", think=30)
    s.param(heads, "times", "$num-heads", think=15)
    s.edge("embed", heads)
    if not p.get("no_residual"):
        s.edge("embed", "residual", target_handle="in-1")
        s.edge(heads, "residual", target_handle="in-2")
    snap = s.ctl("project.snapshot", quiet=True)["result"]
    inside = {n["package"]["id"]: n["id"] for n in snap["nodes"] if n["scope"] == heads}
    hin, hout = inside["core.input"], inside["core.output"]
    s.pause(45)  # enters the subflow
    width = "$(params.in_features / meta[\"num-heads\"])"
    qkv = []
    for _ in range(3):
        node = s.add("core.linear", heads, think=15)
        s.edge(hin, node)
        s.param(node, "in_features", "$d-model", think=10)
        s.param(node, "out_features", width, think=25)
        qkv.append(node)
    q, k, v = qkv
    kt = s.add("core.transpose", heads, think=20)
    s.edge(k, kt)
    scores = s.add("core.matmul", heads, think=20)
    s.edge(q, scores, target_handle="in-1")
    s.edge(kt, scores, target_handle="in-2")
    scale = s.add("core.scale", heads, think=20)
    s.edge(scores, scale)
    s.param(scale, "factor", "$(1 / math.sqrt(meta[\"d-model\"] / meta[\"num-heads\"]))", think=45)
    soft = s.add("core.softmax", heads, think=15)
    s.edge(scale, soft)
    s.param(soft, "dim", -1)
    weighted = s.add("core.matmul", heads, think=15)
    s.edge(soft, weighted, target_handle="in-1")
    s.edge(v, weighted, target_handle="in-2")
    s.edge(weighted, hout)
    s.pause(60)  # checks problems, back to Root
    if not p.get("skip_final_values"):
        s.metavariables({"d-model": "64", "num-heads": "8"}, new=False)
        s.pause(30)
    s.save()


def expert(s: Participant) -> None:
    p = s.p
    session = s.home / "nnm-expert-study" / str(s.pid)
    projects = session / "projects"

    def nnm():
        s.start_app(projects / "task1-mlp")
        nnm_t1(s)
        s.pause(20)
        s.ctl("project.open", {"path": str(projects / "task2-multihead")})
        nnm_t2(s)
        s.pause(10)
        s.stop_app()

    def pytorch():
        s.colab("T1", T1_MODEL, T1_LOSS, submit=2 if p.get("resubmit_t1") else 1, try_error=p.get("try_error", False))
        s.pause(60)
        s.colab("T2", T2_CODE[p["t2_code"]], T2_LOSS, submit=0 if p.get("no_submit_t2") else 1)

    for part in (nnm, pytorch) if p["order"] == "nnm-first" else (pytorch, nnm):
        part()
        s.pause(120)  # break between tools


def run_participant(pid: int, out: Path, usb: Path, speed: float, seed: int, errors: list) -> None:
    s = Participant(pid, out, speed, seed)
    try:
        s.home.mkdir(parents=True, exist_ok=True)
        (s.home / "Scaricati").mkdir(exist_ok=True)
        s.say(f"participant {pid} ({s.p['group']}): {s.p['profile']}")
        if s.p["group"] == "student":
            shutil.copytree(ROOT / "examples" / "test", s.home / "test", dirs_exist_ok=False)
            student(s)
        else:
            subprocess.run([sys.executable, str(usb / "new_session.py"), str(pid)], check=True,
                           env=dict(os.environ, HOME=str(s.home)), capture_output=True)
            expert(s)
        s.say("finished")
    except Exception as exc:  # noqa: BLE001 - report and keep the other participants running
        errors.append(f"{pid}: {exc!r}")
        s.say(f"FAILED: {exc!r}")
        s.stop_app()


def main(argv=None) -> int:
    global OUT_PYLIB, OUT_FAKECOLAB
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", metavar="OUT")
    parser.add_argument("--speed", type=float, default=1.0, help="divide every pause (tests only)")
    parser.add_argument("--only", help="comma-separated participant IDs")
    parser.add_argument("--seed", type=int, default=24)
    args = parser.parse_args(argv)
    out = Path(args.out).expanduser().resolve()
    ids = [int(x) for x in args.only.split(",")] if args.only else sorted(PROFILES)
    if any((out / "machines" / f"pc-{pid}").exists() for pid in ids):
        parser.error(f"{out} already holds some of these participants: use a new OUT")
    for tool in (APP, CTL, KERNEL_PYTHON):
        if not tool.exists():
            parser.error(f"missing {tool}")
    (out / "run").mkdir(parents=True, exist_ok=True)
    (out / "transcripts").mkdir(exist_ok=True)
    subprocess.run(["cc", "-shared", "-fPIC", str(LOOPBACK), "-ldl", "-o", str(out / "run" / "loopback.so")], check=True)
    OUT_PYLIB, OUT_FAKECOLAB = out / "pylib", out / "fakecolab"
    if any(PROFILES[pid]["group"] == "expert" for pid in ids):
        if not (OUT_PYLIB / "jupyter_client").is_dir():
            subprocess.run(["uv", "pip", "install", "-q", "--python", str(KERNEL_PYTHON), "--target", str(OUT_PYLIB),
                            "ipykernel", "jupyter_client"], check=True)
        sys.path.insert(0, str(OUT_PYLIB))
        package = OUT_FAKECOLAB / "google" / "colab"
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text('"""Simulation stand-in for google.colab."""\n')
        (package / "files.py").write_text(
            "import os, shutil\n\n\ndef download(path):\n"
            "    folder = os.environ['SIM_DOWNLOADS']  # the lab PC's browser download folder\n"
            "    os.makedirs(folder, exist_ok=True)\n"
            "    shutil.copy(path, os.path.join(folder, os.path.basename(path)))\n")
    usb = out / "usb" / "nnm-study"
    if not usb.is_dir():
        subprocess.run([sys.executable, str(HERE / "prepare_usb.py"), str(usb)], check=True, capture_output=True)
    with (out / "participants.csv").open("w", newline="") as full, \
            (out / "participants-sim.csv").open("w", newline="") as short:
        wf, ws = csv.writer(full), csv.writer(short)
        wf.writerow(["id", "group", "machine", "profile"])
        ws.writerow(["id", "group"])
        for pid in sorted(PROFILES):
            wf.writerow([pid, PROFILES[pid]["group"], f"pc-{pid}", PROFILES[pid]["profile"]])
            ws.writerow([pid, PROFILES[pid]["group"]])
    errors, threads = [], []
    for pid in ids:
        thread = threading.Thread(target=run_participant, args=(pid, out, usb, args.speed, args.seed, errors))
        thread.start()
        threads.append(thread)
        time.sleep(2)  # arrivals a moment apart
    for thread in threads:
        thread.join()
    print(f"Simulation finished in {out}; transcripts in {out / 'transcripts'}")
    for error in errors:
        print("ERROR", error)
    print("\nCollect each participant onto the stick (D = the stick's nnm-study folder):")
    for pid in ids:
        print(f'HOME={out / "machines" / f"pc-{pid}" / "home"} python3 "$D/collect.py" {pid} {PROFILES[pid]["group"]}')
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
