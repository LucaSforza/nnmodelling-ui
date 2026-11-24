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
    """One attention head; `scale` selects how the 1/sqrt(dh) factor is applied."""

    def __init__(self, d, dh, softmax="torch", scale="mul"):
        super().__init__()
        self.q, self.k, self.v = (nn.Linear(d, dh) for _ in range(3))
        self.dh = dh
        self.softmax = softmax
        self.scale = scale

    def scores(self, q, k):
        kt, dk = k.transpose(-2, -1), self.dh
        if self.scale == "mul":
            return q @ kt * (1 / math.sqrt(dk))
        if self.scale == "div":
            return torch.matmul(q, kt) / math.sqrt(dk)
        if self.scale == "pre-q":
            return (q * dk ** -0.5) @ kt
        if self.scale == "quarter-each":
            return (q * dk ** -0.25) @ (k / dk ** 0.25).transpose(-2, -1)
        if self.scale == "inplace":
            s = q @ kt
            s *= dk ** -0.5
            return s
        if self.scale == "tensor-factor":
            return torch.mul(q @ kt, torch.tensor(dk ** -0.5))
        if self.scale == "sqrt":  # wrong: multiplies instead of dividing
            return q @ kt * math.sqrt(dk)
        if self.scale == "full-width":  # wrong: 1/sqrt(d_model) instead of 1/sqrt(head width)
            return q @ kt * (1 / math.sqrt(4 * dk))
        if self.scale == "none":
            return q @ kt
        if self.scale == "einsum":  # correct, but not a matmul the probe can pair with the softmax
            return torch.einsum("bik,bjk->bij", q, k) * (1 / math.sqrt(dk))
        raise ValueError(self.scale)

    def forward(self, h):
        s = self.scores(self.q(h), self.k(h))
        a = {"torch": lambda: torch.softmax(s, -1), "F": lambda: F.softmax(s, -1),
             "method": lambda: s.softmax(-1), "module": lambda: nn.Softmax(-1)(s),
             "wrong-dim": lambda: torch.softmax(s, 1)}[self.softmax]()
        return a @ self.v(h)


class MultiHead(nn.Module):
    """Loop style; `attention` replaces the per-head loop when given."""

    def __init__(self, d_model=32, num_heads=4, softmax="torch", scale="mul"):
        super().__init__()
        self.embed = nn.Linear(28, d_model)
        self.classifier = nn.Linear(28 * d_model, 10)
        self.heads = nn.ModuleList(Head(d_model, d_model // num_heads, softmax, scale) for _ in range(num_heads))

    def attend(self, h):
        return torch.cat([head(h) for head in self.heads], dim=-1)

    def forward(self, x):
        h = self.embed(torch.flatten(x, 1, 2))
        return self.classifier(torch.flatten(h + self.attend(h), 1))


class BatchedMultiHead(nn.Module):
    """Wide Q/K/V Linear split into heads with view; `prescale` scales Q before the split."""

    def __init__(self, d_model=32, num_heads=4, prescale=False, width=None):
        super().__init__()
        self.h, self.d = num_heads, d_model
        self.prescale, self.width = prescale, width
        self.embed = nn.Linear(28, d_model)
        self.classifier = nn.Linear(28 * d_model, 10)
        self.q, self.k, self.v = (nn.Linear(d_model, d_model) for _ in range(3))

    def forward(self, x):
        h = self.embed(torch.flatten(x, 1, 2))
        split = lambda t: t.view(t.shape[0], 28, self.h, -1).transpose(1, 2)  # noqa: E731
        dk = self.width or self.d // self.h
        q = self.q(h) * dk ** -0.5 if self.prescale else self.q(h)
        q, k, v = split(q), split(self.k(h)), split(self.v(h))
        scores = q @ k.transpose(-2, -1)
        a = torch.softmax(scores if self.prescale else scores / math.sqrt(dk), -1)
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
    assert sc.check_task2(MultiHead, loss).passed


@pytest.mark.parametrize("style", ["torch", "F", "method", "module"])
def test_every_softmax_call_style_is_counted(style):
    report = sc.check_task2(lambda **kw: MultiHead(softmax=style, **kw), loss)
    assert report.passed, str(report)


def test_batched_view_variant_passes_task2():
    assert sc.check_task2(BatchedMultiHead, loss).passed


def test_expected_task2_params():
    assert sc.expected_task2_params(32) == 13066
    assert sc.expected_task2_params(64) == 32266
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
    item = names(sc.check_task2(MhaModel, loss))["d=32,h=4: forbidden primitives"]
    assert not item.passed
    assert "MultiheadAttention" in item.detail and "multi_head_attention_forward" in item.detail


def test_sdpa_is_forbidden():
    item = names(sc.check_task2(SdpaModel, loss))["d=32,h=4: forbidden primitives"]
    assert not item.passed and "scaled_dot_product_attention" in item.detail


def test_transformer_module_is_forbidden():
    class Wrapped(MLP):
        def __init__(self):
            super().__init__()
            self.unused = nn.TransformerEncoderLayer(8, 2)

    assert not names(sc.check_task1(Wrapped, loss))["forbidden primitives"].passed


def test_single_head_fails_task2_heads_probe():
    report = sc.check_task2(SingleHead, loss)
    items = names(report)
    assert not report.passed
    assert items["d=32,h=4: attention heads"].passed is False
    assert items["d=32,h=2: attention heads"].passed is False
    assert "d=64,h=8: attention heads" in items and not items["d=64,h=8: attention heads"].passed


def test_softmax_over_wrong_dimension_fails():
    items = names(sc.check_task2(lambda **kw: MultiHead(softmax="wrong-dim", **kw), loss))
    heads = items["d=32,h=4: attention heads"]
    assert not heads.passed and "not normalised over the last dimension" in heads.detail


def test_hardcoded_width_fails_at_d64():
    items = names(sc.check_task2(Hardcoded32, loss))
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
    assert not sc.check_task2(BadInit, loss).passed


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
    report = sc.check_module("T2", MultiHead(), loss, expected_params=13066, expected_heads=4, head_width=8)
    assert report.passed and names(report)["attention scale"].passed
    report = sc.check_module("T2", MultiHead(), loss, expected_params=13066, expected_heads=2)
    assert not names(report)["attention heads"].passed and "attention scale" not in names(report)
    report = sc.check_module("T2", MultiHead(), loss, expected_params=13066, expected_heads=4, head_width=16)
    assert not names(report)["attention scale"].passed
    assert "heads" not in " ".join(names(sc.check_module("T1", MLP(), loss, expected_params=50890, expected_heads=None)))


# --- attention scale ----------------------------------------------------------

VARIANT_NAMES = ("d=32,h=4: ", "d=64,h=8: ", "d=32,h=2: ")


@pytest.mark.parametrize("scale", ["mul", "div", "pre-q", "quarter-each", "inplace", "tensor-factor"])
def test_correct_scale_styles_pass(scale):
    report = sc.check_task2(lambda **kw: MultiHead(scale=scale, **kw), loss)
    assert report.passed, str(report)
    assert all(names(report)[prefix + "attention scale"].passed for prefix in VARIANT_NAMES)
    assert "scale 0.3536 = 1/sqrt(8)" in names(report)["d=64,h=8: attention scale"].detail
    assert "scale 0.25 = 1/sqrt(16)" in names(report)["d=32,h=2: attention scale"].detail


@pytest.mark.parametrize("prescale", [False, True])
def test_view_based_head_split_passes_scale(prescale):
    report = sc.check_task2(lambda **kw: BatchedMultiHead(prescale=prescale, **kw), loss)
    assert report.passed, str(report)


def test_multiplying_by_sqrt_instead_of_dividing_fails():
    report = sc.check_task2(lambda **kw: MultiHead(scale="sqrt", **kw), loss)
    item = names(report)["d=32,h=4: attention scale"]
    assert not report.passed and not item.passed
    assert item.detail == "scale 2.828 per head, expected 0.3536 = 1/sqrt(8)"
    assert names(report)["d=32,h=4: attention heads"].passed and names(report)["d=32,h=4: parameters"].passed


def test_missing_scale_fails():
    item = names(sc.check_task2(lambda **kw: MultiHead(scale="none", **kw), loss))["d=64,h=8: attention scale"]
    assert not item.passed and item.detail == "scale 1 per head, expected 0.3536 = 1/sqrt(8)"


def test_scale_uses_head_width_not_model_width():
    items = names(sc.check_task2(lambda **kw: MultiHead(scale="full-width", **kw), loss))
    assert not items["d=32,h=4: attention scale"].passed
    assert "expected 0.3536 = 1/sqrt(8)" in items["d=32,h=4: attention scale"].detail
    items = names(sc.check_task2(lambda **kw: BatchedMultiHead(width=kw["d_model"], **kw), loss))
    assert not items["d=64,h=8: attention scale"].passed and not items["d=32,h=2: attention scale"].passed


def test_undeterminable_scale_is_a_failing_item_only_without_measurement():
    report = sc.check_task2(lambda **kw: MultiHead(scale="einsum", **kw), loss)
    item = names(report)["d=32,h=4: attention scale"]
    assert not item.passed and "could not determine" in item.detail
    assert names(report)["d=32,h=4: attention heads"].passed


def test_scale_item_skipped_when_forward_fails():
    class Broken(MultiHead):
        def attend(self, h):
            raise ValueError("boom")

    item = names(sc.check_task2(Broken, loss))["d=32,h=4: attention scale"]
    assert not item.passed and "skipped" in item.detail


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
        try:
            exec(compile("".join(c["source"]), sys.argv[1], "exec"), ns)
        except NotImplementedError:  # a participant notebook's TODOs
            pass
task, model = ("T1", "MLP") if "task1" in sys.argv[1] else ("T2", "MultiHeadRowAttentionClassifier")
sc = ns["study_checks"]
assert sc.LAST_REPORT is None  # the notebooks never print or store a check result
print("REPORT:" + json.dumps(sc.evaluate(task, ns[model], ns["compute_loss"]).to_dict()))
"""

NOTEBOOK_NAMES = ["task1_mlp", "task2_multihead"]


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
        assert nb["metadata"]["colab"]["generative_ai_disabled"] is True  # no Gemini in Colab
        for cell in nb["cells"]:
            assert cell["cell_type"] in ("markdown", "code")
            assert cell["id"] == cell["metadata"]["id"]
            if cell["cell_type"] == "code":
                assert cell["outputs"] == [] and cell["execution_count"] is None


@pytest.mark.parametrize("folder", [NOTEBOOKS, NOTEBOOKS / "solutions"])
def test_notebooks_follow_the_colab_protocol(folder):
    for name in NOTEBOOK_NAMES:
        nb = json.loads((folder / f"{name}.ipynb").read_text())
        task = "T1" if "task1" in name else "T2"
        cells = nb["cells"]
        code = [c for c in cells if c["cell_type"] == "code"]
        start = code[0]
        assert cells.index(start) == 1 and start["metadata"]["cellView"] == "form"
        assert "run this cell first" in start["source"][0] and f'study_checks.begin("{task}", participant_id)' in "".join(start["source"])
        headings = [c for c in cells if c["cell_type"] == "markdown" and "".join(c["source"]).startswith("## ")]
        assert nb["metadata"]["colab"]["collapsed_sections"] == [c["id"] for c in headings]
        assert len(headings) == 3 and cells.index(headings[0]) == 2  # everything after the start cell is collapsed
        submit = "".join(code[-1]["source"])
        assert "\n# study_checks.submit(" in submit and "\nstudy_checks.submit(" not in submit  # Run all never submits
        assert "check_task" not in "".join("".join(c["source"]) for c in code)


def test_notebooks_are_generated():
    sys.path.insert(0, str(TOOLS))
    import build_notebooks

    assert build_notebooks.main(["--check"]) == 0


COLAB_SETUP = """
import json, shutil, sys, urllib.request
from pathlib import Path
root, notebook = Path(sys.argv[1]), sys.argv[2]
prefix = "https://raw.githubusercontent.com/LucaSforza/nnmodelling-ui/"
fetched = []
def urlretrieve(url, dst):
    assert url.startswith(prefix), url
    rel = url[len(prefix):].split("/", 1)[1]
    shutil.copy(root / rel, dst)
    fetched.append(rel)
urllib.request.urlretrieve = urlretrieve
colab = Path("fake_site/google/colab")  # pretend to be Colab
colab.mkdir(parents=True)
(colab.parent / "__init__.py").write_text("")
(colab / "__init__.py").write_text("")
Path("study_checks.py").write_text("raise RuntimeError('stale helpers from an earlier run')")
sys.path.insert(0, str(Path("fake_site").resolve()))
cell = next(c for c in json.load(open(notebook))["cells"] if c["cell_type"] == "code")
ns = {}
exec("".join(cell["source"]), ns)
print("SETUP:" + json.dumps({"fetched": fetched, "module": ns["study_checks"].__file__,
                             "x": list(ns["x"].shape)}))
"""


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_notebook_setup_downloads_helpers_outside_repository(tmp_path, name):
    """Colab: the current study_checks is always fetched from GitHub, over any stale copy."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "NNM_STUDY_DATA")}
    done = subprocess.run(
        [sys.executable, "-c", COLAB_SETUP, str(ROOT), str(NOTEBOOKS / f"{name}.ipynb")],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, done.stderr
    line = next(line for line in done.stdout.splitlines() if line.startswith("SETUP:"))
    result = json.loads(line[len("SETUP:"):])
    assert result["fetched"] == ["tools/expert_study/study_checks.py", "examples/test/datasets/mnist/data.json"]
    assert Path(result["module"]).parent == tmp_path.resolve()
    assert result["x"] == [16, 1, 28, 28]


# --- begin / submit -----------------------------------------------------------


def fake_ipython(history):
    return SimpleNamespace(events=SimpleNamespace(register=lambda name, fn: None),
                           history_manager=SimpleNamespace(input_hist_raw=["", *history]))


def test_submit_is_silent_final_and_saves_the_code(clean_sessions, tmp_path, capsys):
    ip = fake_ipython(["import torch", "class MLP: ...", 'study_checks.submit("T1", MLP, compute_loss)'])
    sc.LAST_REPORT = None
    with pytest.raises(ValueError, match="participant number"):
        sc.begin("T1", "P07", tmp_path / "activity", ip=ip)  # the timer never starts without an ID
    assert not (tmp_path / "activity").exists()
    assert sc.begin("T1", " 007 ", tmp_path / "activity", ip=ip) is True
    assert sc.submit("T1", MLP, loss, out_dir=tmp_path, ip=ip) is None  # nothing displayed as Out[]
    first, = tmp_path.glob("nnm-study-7-T1-*-1.json")
    out = capsys.readouterr().out
    assert "PASS" not in out and "parameters" not in out and "final" in out
    assert sc.LAST_REPORT is None
    data = json.loads(first.read_text())
    assert data["format"] == sc.SUBMISSION_FORMAT
    assert data["participant"] == "7" and data["attempt"] == 1 and data["report"]["passed"] is True
    assert data["code"] == ip.history_manager.input_hist_raw[1:]
    assert data["started"] and data["started"] <= data["submitted"]
    assert [e["event"] for e in data["events"]] == ["start", "submit"]

    sc.submit("T1", nn.Linear, loss, out_dir=tmp_path, ip=ip)  # a worse answer, later
    second, = tmp_path.glob("nnm-study-7-T1-*-2.json")
    assert "only the first submission counts" in capsys.readouterr().out
    data = json.loads(second.read_text())
    assert data["attempt"] == 2 and data["report"]["passed"] is False
    log = [json.loads(line) for line in (tmp_path / "activity/T1.jsonl").read_text().splitlines()]
    assert [e.get("event") for e in log] == ["start", "submit", "resubmit"]


def test_submit_without_begin_still_writes_a_file(clean_sessions, tmp_path, capsys):
    sc.submit("T2", MultiHead, loss, out_dir=tmp_path, ip=fake_ipython([]))
    path, = tmp_path.glob("nnm-study-anon-T2-*.json")
    data = json.loads(path.read_text())
    assert data["participant"] == "" and data["started"] is None and data["attempt"] == 1
    assert "Saved" in capsys.readouterr().out


@pytest.mark.parametrize("value, expected", [("7", "7"), (" 007 ", "7"), (12, "12"), ("1234567", "1234567")])
def test_participant_id_is_canonical(value, expected):
    assert sc.participant_id(value) == expected


@pytest.mark.parametrize("value", ["", "0", "000", "P07", "mario-rossi-1234567", "1 2", "-3", "٣", None])
def test_participant_id_rejects_anything_but_a_number(value):
    with pytest.raises(ValueError):
        sc.participant_id(value)


def test_begin_outside_ipython_ignores_the_id(clean_sessions):
    assert sc.begin("T1", "", ip=None) is False  # grader replays of notebooks with an empty field
