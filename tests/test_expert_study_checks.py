import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "expert_study"
NOTEBOOKS = ROOT / "study" / "experts" / "notebooks"
sys.path.insert(0, str(TOOLS))

import study_checks as sc  # noqa: E402


def loss(logits, target):
    return F.cross_entropy(logits, target)


# --- reference models -------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(1, -1), nn.Linear(784, hidden), nn.ReLU(), nn.Linear(hidden, 10))

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    def __init__(self, d, dh, softmax="torch"):
        super().__init__()
        self.q, self.k, self.v = (nn.Linear(d, dh) for _ in range(3))
        self.scale = 1 / math.sqrt(dh)
        self.softmax = softmax

    def forward(self, h):
        s = self.q(h) @ self.k(h).transpose(-2, -1) * self.scale
        a = {"torch": lambda: torch.softmax(s, -1), "F": lambda: F.softmax(s, -1),
             "method": lambda: s.softmax(-1), "module": lambda: nn.Softmax(-1)(s),
             "wrong-dim": lambda: torch.softmax(s, 1)}[self.softmax]()
        return a @ self.v(h)


class MultiHead(nn.Module):
    """Loop style; `attention` replaces the per-head loop when given."""

    def __init__(self, d_model=32, num_heads=4, softmax="torch"):
        super().__init__()
        self.embed = nn.Linear(28, d_model)
        self.classifier = nn.Linear(28 * d_model, 10)
        self.heads = nn.ModuleList(Head(d_model, d_model // num_heads, softmax) for _ in range(num_heads))

    def attend(self, h):
        return torch.cat([head(h) for head in self.heads], dim=-1)

    def forward(self, x):
        h = self.embed(torch.flatten(x, 1, 2))
        return self.classifier(torch.flatten(h + self.attend(h), 1))


class BatchedMultiHead(nn.Module):
    def __init__(self, d_model=32, num_heads=4):
        super().__init__()
        self.h, self.d = num_heads, d_model
        self.embed = nn.Linear(28, d_model)
        self.classifier = nn.Linear(28 * d_model, 10)
        self.q, self.k, self.v = (nn.Linear(d_model, d_model) for _ in range(3))

    def forward(self, x):
        h = self.embed(torch.flatten(x, 1, 2))
        split = lambda t: t.view(t.shape[0], 28, self.h, -1).transpose(1, 2)  # noqa: E731
        q, k, v = split(self.q(h)), split(self.k(h)), split(self.v(h))
        a = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.d // self.h), -1)
        attn = (a @ v).transpose(1, 2).reshape(x.shape[0], 28, self.d)
        return self.classifier(torch.flatten(h + attn, 1))


class SingleHead(MultiHead):
    def __init__(self, d_model=32, num_heads=4):
        super().__init__(d_model, 1)


class Hardcoded32(nn.Module):
    def __init__(self, d_model=32, num_heads=4):
        super().__init__()
        self.inner = MultiHead(32, num_heads if 32 % num_heads == 0 else 4)

    def forward(self, x):
        return self.inner(x)


class T2(MultiHead):
    def __init__(self):
        super().__init__(32, 1)


class SdpaModel(MultiHead):
    def attend(self, h):
        return F.scaled_dot_product_attention(h[:, None], h[:, None], h[:, None])[:, 0]


class MhaModel(MultiHead):
    def __init__(self, d_model=32, num_heads=4):
        super().__init__(d_model, num_heads)
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def attend(self, h):
        return self.mha(h, h, h)[0]


def names(report):
    return {item.name: item for item in report.checks}


# --- reference implementations pass ----------------------------------------


def test_reference_solutions_pass():
    assert sc.check_task1(MLP, loss).passed
    assert sc.check_task2(T2, loss).passed
    assert sc.check_task3(MultiHead, loss).passed


@pytest.mark.parametrize("style", ["torch", "F", "method", "module"])
def test_every_softmax_call_style_is_counted(style):
    report = sc.check_task3(lambda **kw: MultiHead(softmax=style, **kw), loss)
    assert report.passed, str(report)


def test_batched_view_variant_passes_task3():
    assert sc.check_task3(BatchedMultiHead, loss).passed


def test_expected_task3_params():
    assert sc.expected_task3_params(32) == 13066
    assert sc.expected_task3_params(64) == 32266
    assert sum(p.numel() for p in MultiHead(64, 8).parameters()) == 32266


# --- failures are detected --------------------------------------------------


def test_wrong_hidden_size():
    report = sc.check_task1(lambda: MLP(32), loss)
    assert not report.passed and not names(report)["parameters"].passed


def test_wrong_loss():
    report = sc.check_task1(MLP, lambda logits, target: F.cross_entropy(logits, target) * 2)
    assert not names(report)["loss"].passed
    assert not sc.check_task1(MLP, lambda logits, target: F.cross_entropy(logits, target, reduction="none")).passed


def test_multihead_attention_module_is_forbidden():
    item = names(sc.check_task3(MhaModel, loss))["d=32,h=4: forbidden primitives"]
    assert not item.passed
    assert "MultiheadAttention" in item.detail and "multi_head_attention_forward" in item.detail


def test_sdpa_is_forbidden():
    item = names(sc.check_task2(lambda: SdpaModel(32, 1), loss))["forbidden primitives"]
    assert not item.passed and "scaled_dot_product_attention" in item.detail


def test_transformer_module_is_forbidden():
    class Wrapped(MLP):
        def __init__(self):
            super().__init__()
            self.unused = nn.TransformerEncoderLayer(8, 2)

    assert not names(sc.check_task1(Wrapped, loss))["forbidden primitives"].passed


def test_single_head_fails_task3_heads_probe():
    report = sc.check_task3(SingleHead, loss)
    items = names(report)
    assert not report.passed
    assert items["d=32,h=4: attention heads"].passed is False
    assert items["d=32,h=2: attention heads"].passed is False
    assert "d=64,h=8: attention heads" in items and not items["d=64,h=8: attention heads"].passed


def test_softmax_over_wrong_dimension_fails():
    items = names(sc.check_task3(lambda **kw: MultiHead(softmax="wrong-dim", **kw), loss))
    heads = items["d=32,h=4: attention heads"]
    assert not heads.passed and "not normalised over the last dimension" in heads.detail


def test_hardcoded_width_fails_at_d64():
    items = names(sc.check_task3(Hardcoded32, loss))
    assert items["d=32,h=4: parameters"].passed
    assert not items["d=64,h=8: parameters"].passed


def test_exceptions_become_failed_items():
    class BadInit(nn.Module):
        def __init__(self):
            raise RuntimeError("no init")

    class BadForward(MLP):
        def forward(self, x):
            raise ValueError("boom")

    def bad_loss(logits, target):
        raise KeyError("oops")

    assert "no init" in names(sc.check_task1(BadInit, loss))["construct"].detail
    report = sc.check_task1(BadForward, loss)
    assert "boom" in names(report)["logits"].detail and not report.passed
    assert "oops" in names(sc.check_task1(MLP, bad_loss))["loss"].detail
    assert not sc.check_task3(BadInit, loss).passed


def test_non_finite_and_wrong_shape_logits():
    class Nan(MLP):
        def forward(self, x):
            return super().forward(x) * float("nan")

    class Short(MLP):
        def forward(self, x):
            return super().forward(x)[:, :5]

    assert not names(sc.check_task1(Nan, loss))["logits"].passed
    assert not names(sc.check_task1(Short, loss))["logits"].passed


def test_check_module_generic():
    report = sc.check_module("T2", T2(), loss, expected_params=13066, expected_heads=1)
    assert report.passed
    report = sc.check_module("T2", T2(), loss, expected_params=13066, expected_heads=2)
    assert not names(report)["attention heads"].passed
    assert "heads" not in " ".join(names(sc.check_module("T1", MLP(), loss, expected_params=50890, expected_heads=None)))


# --- reports and data -------------------------------------------------------


def test_report_formats():
    report = sc.CheckReport("T1", False, [sc.CheckItem("a", True, "fine"), sc.CheckItem("b<", False, "bad & sad")])
    assert report.to_dict() == {"task": "T1", "passed": False, "checks": [
        {"name": "a", "passed": True, "detail": "fine"}, {"name": "b<", "passed": False, "detail": "bad & sad"}]}
    assert "[FAIL] b<" in str(report)
    html = report._repr_html_()
    assert "#d6f5d6" in html and "#f8d0d0" in html and "b&lt;" in html and "&amp;" in html


def test_load_batch():
    x, y = sc.load_batch()
    assert x.shape == (16, 1, 28, 28) and x.dtype == torch.float32
    assert y.shape == (16,) and y.dtype == torch.int64
    assert 0.0 <= x.min() and x.max() <= 1.0 and x.max() > 0.5
    x2, y2 = sc.load_batch(16)
    assert torch.equal(x, x2) and torch.equal(y, y2)
    assert sc.load_batch(4, "validation")[0].shape == (4, 1, 28, 28)
    expected = json.loads((ROOT / "examples/test/datasets/mnist/data.json").read_text())["splits"]["train"][0]
    assert y[0].item() == expected["label"]
    assert torch.allclose(x[0].flatten(), torch.tensor(expected["image"], dtype=torch.float32) / 255)


def test_load_batch_path_order(tmp_path, monkeypatch):
    rows = lambda label: {"splits": {"train": [{"image": [255] * 784, "label": label}]}}  # noqa: E731
    arg, env, cwd = (tmp_path / n for n in ("arg.json", "env.json", "mnist_small.json"))
    for path, label in ((arg, 1), (env, 2), (cwd, 3)):
        path.write_text(json.dumps(rows(label)))
    monkeypatch.setenv("NNM_STUDY_DATA", str(env))
    monkeypatch.chdir(tmp_path)
    assert sc.load_batch(path=arg)[1].item() == 1
    assert sc.load_batch()[1].item() == 2
    monkeypatch.delenv("NNM_STUDY_DATA")
    assert sc.load_batch()[1].item() == 3


# --- session log ------------------------------------------------------------


@pytest.fixture
def clean_sessions():
    sc._SESSIONS.clear()
    yield
    sc._SESSIONS.clear()


def test_start_session_noop_outside_ipython(clean_sessions, tmp_path):
    assert sc.start_session("T1", tmp_path) is False
    sc.check_task1(MLP, loss)
    assert list(tmp_path.iterdir()) == []


def test_session_hook_writes_json_lines(clean_sessions, tmp_path):
    hooks = []
    ip = SimpleNamespace(events=SimpleNamespace(register=lambda name, fn: hooks.append((name, fn))))
    assert sc.start_session("T1", tmp_path, ip=ip) is True
    assert sc.start_session("T1", tmp_path, ip=ip) is True
    assert [name for name, _ in hooks] == ["post_run_cell"]
    hook = hooks[0][1]
    hook(SimpleNamespace(execution_count=1, success=True, error_before_exec=None, error_in_exec=None))
    hook(SimpleNamespace(execution_count=2, success=False, error_before_exec=None, error_in_exec=ValueError("bad")))
    sc.check_task1(MLP, loss)
    lines = [json.loads(line) for line in (tmp_path / "T1.jsonl").read_text().splitlines()]
    assert [line.get("execution_count") for line in lines[:2]] == [1, 2]
    assert lines[0]["success"] is True and lines[0]["error"] is None
    assert lines[1]["success"] is False and "bad" in lines[1]["error"]
    assert all(line["task"] == "T1" and line["ts"].endswith("+00:00") for line in lines)
    assert lines[2]["event"] == "check" and lines[2]["passed"] is True and lines[2]["checks"]


# --- notebooks --------------------------------------------------------------

RUN = """
import json, sys
cells = json.load(open(sys.argv[1]))["cells"]
ns = {"__name__": "__main__"}
for c in cells:
    if c["cell_type"] == "code":
        exec(compile("".join(c["source"]), sys.argv[1], "exec"), ns)
print("REPORT:" + json.dumps(ns["study_checks"].LAST_REPORT.to_dict()))
"""

NOTEBOOK_NAMES = ["task1_mlp", "task2_attention", "task3_multihead"]


def run_notebook(tmp_path, path):
    runner = tmp_path / "run.py"
    runner.write_text(RUN)
    env = {**os.environ, "PYTHONPATH": str(TOOLS)}
    env.pop("NNM_STUDY_DATA", None)
    done = subprocess.run([sys.executable, str(runner), str(path)], cwd=path.parent, env=env,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    line = next(line for line in done.stdout.splitlines() if line.startswith("REPORT:"))
    return json.loads(line[len("REPORT:"):])


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_solution_notebooks_pass(tmp_path, name):
    report = run_notebook(tmp_path, NOTEBOOKS / "solutions" / f"{name}.ipynb")
    assert report["passed"], report


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_participant_notebooks_fail_without_crashing(tmp_path, name):
    report = run_notebook(tmp_path, NOTEBOOKS / f"{name}.ipynb")
    assert not report["passed"]


@pytest.mark.parametrize("folder", [NOTEBOOKS, NOTEBOOKS / "solutions"])
def test_notebooks_are_clean_nbformat4(folder):
    for name in NOTEBOOK_NAMES:
        nb = json.loads((folder / f"{name}.ipynb").read_text())
        assert nb["nbformat"] == 4 and nb["metadata"]["kernelspec"]["name"] == "python3"
        for cell in nb["cells"]:
            assert cell["cell_type"] in ("markdown", "code")
            if cell["cell_type"] == "code":
                assert cell["outputs"] == [] and cell["execution_count"] is None
