"""PyTorch-side checks for the NNModelling expert usability study.

Standard library and torch only, so the file can be copied next to the
notebooks. The criteria are those of docs/knowledge/testing/expert-study.md.
"""

from __future__ import annotations

import html
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

_MODULE_DIR = Path(__file__).resolve().parent

EXPECTED = {
    "T1": {"params": 50890, "heads": None},
    "T2": {"params": 13066, "heads": 1},
    "T3": {"configs": ((32, 4), (64, 8), (32, 2))},
}

_FORBIDDEN_FUNCTIONS = {
    "scaled_dot_product_attention",
    "multi_head_attention_forward",
    "_native_multi_head_attention",
}
_FORBIDDEN_MODULES = (
    nn.MultiheadAttention,
    nn.Transformer,
    nn.TransformerEncoder,
    nn.TransformerDecoder,
    nn.TransformerEncoderLayer,
    nn.TransformerDecoderLayer,
)


def expected_task3_params(d_model: int) -> int:
    """Trainable parameters of the T3 model for any d_model and num_heads."""
    return 3 * d_model * d_model + 312 * d_model + 10


# --- data -------------------------------------------------------------------


def _candidates(path):
    if path:
        yield Path(path)
    if os.environ.get("NNM_STUDY_DATA"):
        yield Path(os.environ["NNM_STUDY_DATA"])
    yield Path.cwd() / "mnist_small.json"
    yield _MODULE_DIR / "mnist_small.json"
    yield _MODULE_DIR.parent.parent / "examples/test/datasets/mnist/data.json"


@lru_cache(maxsize=4)
def _read_splits(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["splits"]


def load_batch(batch_size=16, split="train", path=None):
    """First `batch_size` rows of a split as x float32 [B,1,28,28] in [0,1], y int64 [B]."""
    for candidate in _candidates(path):
        if candidate.is_file():
            rows = _read_splits(str(candidate))[split][:batch_size]
            break
    else:
        searched = ", ".join(str(c) for c in _candidates(path))
        raise FileNotFoundError(f"MNIST study data not found (searched: {searched})")
    if not rows:
        raise ValueError(f"split {split!r} has no rows")
    x = torch.tensor([r["image"] for r in rows], dtype=torch.float32) / 255.0
    y = torch.tensor([r["label"] for r in rows], dtype=torch.int64)
    return x.reshape(-1, 1, 28, 28), y


# --- reports ----------------------------------------------------------------


@dataclass
class CheckItem:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class CheckReport:
    task: str
    passed: bool
    checks: list[CheckItem] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "passed": self.passed,
            "checks": [vars(item).copy() for item in self.checks],
        }

    def __str__(self) -> str:
        lines = [f"{self.task}: {'PASSED' if self.passed else 'NOT PASSED'}"]
        for item in self.checks:
            lines.append(f"  [{'ok' if item.passed else 'FAIL'}] {item.name}: {item.detail}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        rows = "".join(
            f'<tr style="background:{"#d6f5d6" if item.passed else "#f8d0d0"}">'
            f"<td>{'PASS' if item.passed else 'FAIL'}</td>"
            f"<td>{html.escape(item.name)}</td><td>{html.escape(item.detail)}</td></tr>"
            for item in self.checks
        )
        verdict = "PASSED" if self.passed else "NOT PASSED"
        return f"<b>{html.escape(self.task)}: {verdict}</b><table>{rows}</table>"


# --- session log ------------------------------------------------------------

_SESSIONS: dict[str, Path] = {}
LAST_REPORT: CheckReport | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(task: str, record: dict) -> None:
    path = _SESSIONS.get(task)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"ts": _now(), "task": task, **record}) + "\n")
    except OSError:
        pass  # logging must never disturb the participant


def start_session(task, log_dir=None, ip=None) -> bool:
    """Log every executed cell of an IPython session; returns False outside IPython."""
    if task in _SESSIONS:
        return True
    if ip is None:
        try:
            from IPython import get_ipython

            ip = get_ipython()
        except ImportError:
            ip = None
    if ip is None:
        return False
    _SESSIONS[task] = (Path(log_dir) if log_dir else Path.cwd() / "activity") / f"{task}.jsonl"

    def post_run_cell(result):
        error = getattr(result, "error_in_exec", None) or getattr(result, "error_before_exec", None)
        _append(
            task,
            {
                "execution_count": getattr(result, "execution_count", None),
                "success": bool(getattr(result, "success", error is None)),
                "error": None if error is None else str(error),
            },
        )

    ip.events.register("post_run_cell", post_run_cell)
    return True


# --- checks -----------------------------------------------------------------


class _Probe(TorchFunctionMode):
    """Records forbidden function calls and 28x28 softmax maps during a forward pass."""

    def __init__(self, batch_size: int, seq_len: int):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.forbidden: list[str] = []
        self.maps = 0.0
        self.wrong_dim = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        name = getattr(func, "__name__", "")
        if name in _FORBIDDEN_FUNCTIONS:
            self.forbidden.append(name)
        out = func(*args, **(kwargs or {}))
        if name == "softmax" and isinstance(out, torch.Tensor) and out.dim() >= 2:
            if tuple(out.shape[-2:]) == (self.seq_len, self.seq_len):
                # Attention weights must be normalised over the keys (last dimension).
                try:
                    with torch.no_grad():
                        rows = out.detach().float().sum(-1)
                        normalised = bool(torch.allclose(rows, torch.ones_like(rows), atol=1e-4))
                except RuntimeError:  # e.g. under torch.func.vmap: cannot inspect, assume last dim
                    normalised = True
                if normalised:
                    self.maps += out.numel() / (self.batch_size * self.seq_len * self.seq_len)
                else:
                    self.wrong_dim += 1
        return out


def _error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _check_items(model, loss_fn, expected_params, expected_heads, batch, seq_len):
    x, y = batch
    probe = _Probe(x.shape[0], seq_len)
    logits, forward_error = None, None
    try:
        with probe:
            logits = model(x)
    except Exception as exc:  # noqa: BLE001 - participant code
        forward_error = _error(exc)

    items = []
    if forward_error is not None:
        items.append(CheckItem("logits", False, f"forward failed: {forward_error}"))
    elif not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
        items.append(CheckItem("logits", False, "forward must return a float tensor"))
    elif tuple(logits.shape) != (x.shape[0], 10):
        items.append(CheckItem("logits", False, f"shape {tuple(logits.shape)}, expected {(x.shape[0], 10)}"))
    elif not torch.isfinite(logits).all():
        items.append(CheckItem("logits", False, "logits contain NaN or inf"))
    else:
        items.append(CheckItem("logits", True, f"shape {tuple(logits.shape)}, finite"))

    if not items[0].passed:
        items.append(CheckItem("loss", False, "skipped: logits are invalid"))
    else:
        try:
            loss = loss_fn(logits, y)
            reference = F.cross_entropy(logits.detach(), y)
            if not isinstance(loss, torch.Tensor) or loss.dim() != 0:
                items.append(CheckItem("loss", False, "loss must be a scalar tensor"))
            elif not torch.isfinite(loss):
                items.append(CheckItem("loss", False, "loss is NaN or inf"))
            elif not torch.isclose(loss.detach(), reference, atol=1e-5):
                items.append(CheckItem("loss", False, f"{loss.item():.6f} differs from cross_entropy {reference.item():.6f}"))
            else:
                items.append(CheckItem("loss", True, f"{loss.item():.6f} equals cross_entropy"))
        except Exception as exc:  # noqa: BLE001 - participant code
            items.append(CheckItem("loss", False, f"loss failed: {_error(exc)}"))

    try:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        items.append(CheckItem("parameters", params == expected_params, f"{params} trainable, expected {expected_params}"))
    except Exception as exc:  # noqa: BLE001 - participant code
        items.append(CheckItem("parameters", False, _error(exc)))

    banned = sorted({type(m).__name__ for m in model.modules() if isinstance(m, _FORBIDDEN_MODULES)})
    banned += sorted(set(probe.forbidden))
    items.append(CheckItem("forbidden primitives", not banned, ", ".join(banned) if banned else "none used"))

    if expected_heads is not None:
        heads = probe.maps
        ok = math.isclose(heads, expected_heads, abs_tol=1e-6) and not probe.wrong_dim
        detail = f"{heads:g} softmax map(s) of {seq_len}x{seq_len} per sample, expected {expected_heads}"
        if probe.wrong_dim:
            detail += f"; {probe.wrong_dim} softmax call(s) not normalised over the last dimension"
        items.append(CheckItem("attention heads", ok, detail))
    return items


def check_module(task, model, loss_fn, *, expected_params, expected_heads, batch=None, seq_len=28):
    """Check one model instance (x -> logits) and its loss against the study criteria."""
    try:
        batch = batch if batch is not None else load_batch()
    except Exception as exc:  # noqa: BLE001
        return CheckReport(task, False, [CheckItem("data", False, _error(exc))])
    items = _check_items(model, loss_fn, expected_params, expected_heads, batch, seq_len)
    return CheckReport(task, all(item.passed for item in items), items)


def _finish(report: CheckReport) -> CheckReport:
    global LAST_REPORT
    LAST_REPORT = report
    print(report)
    _append(report.task, {"event": "check", "passed": report.passed, "checks": report.to_dict()["checks"]})
    return report


def _check_class(task, model_cls, loss_fn, batch, configs):
    """configs: (prefix, constructor kwargs, expected params, expected heads)."""
    try:
        batch = batch if batch is not None else load_batch()
    except Exception as exc:  # noqa: BLE001
        return _finish(CheckReport(task, False, [CheckItem("data", False, _error(exc))]))
    items = []
    for prefix, kwargs, params, heads in configs:
        try:
            model = model_cls(**kwargs)
        except Exception as exc:  # noqa: BLE001 - participant code
            items.append(CheckItem(f"{prefix}construct", False, _error(exc)))
            continue
        for item in _check_items(model, loss_fn, params, heads, batch, 28):
            items.append(CheckItem(prefix + item.name, item.passed, item.detail))
    return _finish(CheckReport(task, all(item.passed for item in items), items))


def check_task1(model_cls, loss_fn, batch=None):
    spec = EXPECTED["T1"]
    return _check_class("T1", model_cls, loss_fn, batch, [("", {}, spec["params"], spec["heads"])])


def check_task2(model_cls, loss_fn, batch=None):
    spec = EXPECTED["T2"]
    return _check_class("T2", model_cls, loss_fn, batch, [("", {}, spec["params"], spec["heads"])])


def check_task3(model_cls, loss_fn, batch=None):
    configs = [
        (f"d={d},h={h}: ", {"d_model": d, "num_heads": h}, expected_task3_params(d), h)
        for d, h in EXPECTED["T3"]["configs"]
    ]
    return _check_class("T3", model_cls, loss_fn, batch, configs)
