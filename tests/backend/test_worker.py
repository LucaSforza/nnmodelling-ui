from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from backend import worker


@dataclass
class Batch:
    inputs: dict[str, torch.Tensor]
    targets: dict[str, torch.Tensor]


class TinyAdapter:
    def load(self, split: str, _batch_size: int):
        sizes = {"train": (1, 2, 1), "validation": (2,), "test": (2,)}[split]
        for size in sizes:
            inputs = {"x": torch.ones((size, 1))}
            targets = {"y": torch.full((size, 1), 0.75)}
            yield Batch(inputs, targets)


class StochasticLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
        self.training_losses: list[tuple[float, int]] = []
        self.validation_losses: list[float] = []

    def forward(self, inputs, targets=None, include_loss=False):
        noise = random.random() + float(np.random.random()) + torch.rand(())
        prediction = self.weight * inputs["x"] + noise
        loss = (prediction - targets["y"]).square().mean()
        if self.training:
            self.training_losses.append((float(loss.detach()), inputs["x"].shape[0]))
        else:
            self.validation_losses.append(float(loss.detach()))
        return {"loss": loss}


class CountingSGD:
    def __init__(self, model):
        self.inner = torch.optim.SGD(model.parameters(), lr=0.01)
        self.steps = 0

    def zero_grad(self, **kwargs):
        self.inner.zero_grad(**kwargs)

    def step(self):
        self.steps += 1
        self.inner.step()


def run_training(cadence: int):
    random.seed(25)
    np.random.seed(25)
    torch.manual_seed(25)
    model = StochasticLoss()
    optimizer = CountingSGD(model)
    metrics = {"epochs": [], "steps": [], "test_loss": None}
    worker._train_epochs(model, TinyAdapter(), optimizer, 2, 2, cadence, metrics, lambda: None)
    return model, optimizer, metrics


def test_step_windows_are_weighted_flush_epoch_remainders_and_preserve_rng():
    every_two, opt_two, metrics_two = run_training(2)
    every_four, opt_four, metrics_four = run_training(4)

    assert torch.equal(every_two.weight, every_four.weight)
    assert opt_two.steps == opt_four.steps == 6
    assert every_two.training is every_four.training is True

    assert [item["step"] for item in metrics_two["steps"]] == [2, 3, 4, 6]
    assert [item["step"] for item in metrics_four["steps"]] == [3, 4, 6]
    assert [item["epoch"] for item in metrics_two["steps"]] == [1, 1, 2, 2]
    assert [item["step"] for item in metrics_two["steps"]].count(3) == 1

    traces = every_two.training_losses
    expected_windows = ((0, 2), (2, 3), (3, 4), (4, 6))
    for metric, (start, end) in zip(metrics_two["steps"], expected_windows, strict=True):
        selected = traces[start:end]
        total_size = sum(size for _, size in selected)
        weighted_mean = sum(loss * size for loss, size in selected) / total_size
        assert math.isclose(metric["training_loss"], weighted_mean, rel_tol=1e-6)
    assert [row["validation_loss"] for row in metrics_two["steps"]] == every_two.validation_losses

    for index, summary in enumerate(metrics_two["epochs"]):
        epoch_batches = traces[index * 3:(index + 1) * 3]
        total_size = sum(size for _, size in epoch_batches)
        weighted_mean = sum(loss * size for loss, size in epoch_batches) / total_size
        assert math.isclose(summary["training_loss"], weighted_mean, rel_tol=1e-6)
        assert summary["validation_loss"] == metrics_two["steps"][[1, 3][index]]["validation_loss"]


def test_metrics_publication_has_an_explicit_size_limit(tmp_path, monkeypatch):
    path = tmp_path / "metrics.json"
    path.write_text('{"previous":true}')
    monkeypatch.setattr(worker, "MAX_METRICS_BYTES", 8)
    with pytest.raises(ValueError, match="publication limit"):
        worker._write_json(path, {"steps": ["payload too large"]})
    assert path.read_text() == '{"previous":true}'


def _seed_training():
    random.seed(25)
    np.random.seed(25)
    torch.manual_seed(25)


def _adam_training(model, metrics, **kwargs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    worker._train_epochs(model, TinyAdapter(), optimizer, 2, 2, 4, metrics, lambda: None, **kwargs)
    return optimizer


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    _seed_training()
    expected_model = StochasticLoss()
    expected_metrics = {"epochs": [], "steps": [], "test_loss": None}
    _adam_training(expected_model, expected_metrics)
    expected_rng = worker._rng_state()

    _seed_training()
    interrupted_model = StochasticLoss()
    interrupted_metrics = {"epochs": [], "steps": [], "test_loss": None}
    marker = tmp_path / "checkpoint-request.123"
    marker.touch()
    with pytest.raises(SystemExit) as stopped:
        optimizer = torch.optim.Adam(interrupted_model.parameters(), lr=0.01)
        worker._train_epochs(
            interrupted_model, TinyAdapter(), optimizer, 2, 2, 4, interrupted_metrics, lambda: None,
            checkpoint_request=marker, checkpoint_output=tmp_path, job_id="logical-job",
            scheduler_job_id="123",
        )
    assert stopped.value.code == 75
    marker.unlink()

    checkpoint = worker._load_checkpoint(tmp_path, "logical-job")
    assert checkpoint["scheduler_id"] == "123"
    assert checkpoint["global_step"] == 1
    resumed_model = StochasticLoss()
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=0.01)
    resumed_model.load_state_dict(checkpoint["model_state"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer_state"])
    resumed_metrics = checkpoint["metrics"]
    worker._train_epochs(
        resumed_model, TinyAdapter(), resumed_optimizer, 2, 2, 4, resumed_metrics, lambda: None,
        resume=checkpoint,
    )

    assert torch.equal(resumed_model.weight, expected_model.weight)
    assert resumed_metrics == expected_metrics
    actual_rng = worker._rng_state()
    assert actual_rng["python"] == expected_rng["python"]
    assert actual_rng["numpy"][0] == expected_rng["numpy"][0]
    assert np.array_equal(actual_rng["numpy"][1], expected_rng["numpy"][1])
    assert torch.equal(actual_rng["torch"], expected_rng["torch"])


def test_checkpoint_request_uses_attempt_marker_path(tmp_path, monkeypatch):
    marker = tmp_path / "checkpoint-request.456"
    monkeypatch.setenv("NNMODELLING_CHECKPOINT_REQUEST", str(marker))
    assert worker._checkpoint_request_path() == marker
    marker.touch()
    assert worker._checkpoint_request_path().is_file()


@pytest.mark.parametrize("damage", ["manifest", "tensor", "partial"])
def test_incomplete_or_corrupt_checkpoint_is_rejected(tmp_path, damage):
    if damage == "partial":
        (tmp_path / "checkpoint.safetensors").write_bytes(b"orphan")
    elif damage == "manifest":
        (tmp_path / "checkpoint.safetensors").write_bytes(b"tensor")
        (tmp_path / "checkpoint.json").write_text("{broken")
    else:
        model = StochasticLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        worker._save_checkpoint(tmp_path, "logical-job", "123", {
            "global_step": 1, "epoch": 1, "batch_index": 1, "epoch_start_rng": worker._rng_state(),
            "epoch_total": 1.0, "epoch_count": 1, "window_total": 1.0, "window_count": 1,
            "metrics": {"epochs": [], "steps": [], "test_loss": None},
        }, model, optimizer)
        with (tmp_path / "checkpoint.safetensors").open("ab") as stream:
            stream.write(b"tampered")
    with pytest.raises((ValueError, OSError)):
        worker._load_checkpoint(tmp_path, "logical-job")
