from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch
from fastapi import HTTPException

from backend import worker
from backend.store import JobStore
from backend.validation import validate_training


@dataclass
class Batch:
    inputs: dict[str, torch.Tensor]
    targets: dict[str, torch.Tensor]


class DeterministicAdapter:
    def load(self, split, _batch_size):
        if split == "train":
            for _ in range(2):
                yield Batch({"x": torch.ones((1, 1))}, {"y": torch.ones((1, 1))})
            return
        yield Batch({"x": torch.ones((1, 1))}, {"y": torch.ones((1, 1))})


class ControlledLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.validation_weights = []

    def forward(self, inputs, targets=None, include_loss=False):
        if self.training:
            return {"loss": -self.weight * inputs["x"].mean()}
        self.validation_weights.append(float(self.weight.detach()))
        return {"loss": (self.weight - 0.4).square()}


def _metrics():
    return {"epochs": [], "steps": [], "test_loss": None}


@pytest.mark.parametrize("field,invalid", [
    ("early_stopping_patience", True),
    ("early_stopping_patience", 1.0),
    ("early_stopping_patience", -1),
    ("early_stopping_patience", 10001),
    ("early_stopping_min_delta", True),
    ("early_stopping_min_delta", -0.1),
    ("early_stopping_min_delta", math.inf),
    ("early_stopping_min_delta", math.nan),
    ("early_stopping_min_delta", "0.1"),
])
def test_early_stopping_controls_are_validated(field, invalid):
    request = {"epochs": 2, "batch_size": 1, "learning_rate": 0.01, "seed": 1, field: invalid}
    with pytest.raises(HTTPException) as error:
        validate_training(request)
    assert error.value.status_code == 422


def test_early_stopping_defaults_and_bounds():
    base = {"epochs": 2, "batch_size": 1, "learning_rate": 0.01, "seed": 1}
    result = validate_training(base)
    assert result["early_stopping_patience"] == 0
    assert result["early_stopping_min_delta"] == 0.0
    assert validate_training({**base, "early_stopping_patience": 10000,
                              "early_stopping_min_delta": 0})["early_stopping_min_delta"] == 0.0


def _run(cadence):
    model = ControlledLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    metrics = _metrics()
    worker._train_epochs(
        model, DeterministicAdapter(), optimizer, 5, 1, cadence, metrics, lambda: None,
        early_stopping_patience=2,
    )
    return model, metrics


def test_early_stopping_uses_final_epoch_step_and_is_independent_of_publish_cadence():
    frequent_model, frequent_metrics = _run(1)
    sparse_model, sparse_metrics = _run(100)

    assert len(frequent_metrics["epochs"]) == len(sparse_metrics["epochs"]) == 4
    assert [row["validation_loss"] for row in frequent_metrics["epochs"]] == pytest.approx(
        [row["validation_loss"] for row in sparse_metrics["epochs"]]
    )
    assert frequent_metrics["early_stopping"] == sparse_metrics["early_stopping"] == {
        "best_epoch": 2,
        "best_validation_loss": pytest.approx(0.0),
        "non_improving_epochs": 2,
        "stopped_epoch": 4,
    }
    torch.testing.assert_close(frequent_model.weight, torch.tensor(0.4))
    torch.testing.assert_close(sparse_model.weight, frequent_model.weight)


def test_max_step_cutoff_restores_last_completed_best_epoch():
    model = ControlledLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    metrics = _metrics()
    worker._train_epochs(
        model, DeterministicAdapter(), optimizer, 5, 1, 1, metrics, lambda: None,
        max_steps=3, early_stopping_patience=10,
    )

    assert len(metrics["epochs"]) == 1
    torch.testing.assert_close(model.weight, torch.tensor(0.2))


def test_early_stopping_checkpoint_resume_preserves_selection(tmp_path):
    marker = tmp_path / "checkpoint-request.123"

    def baseline():
        model = ControlledLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        metrics = _metrics()
        worker._train_epochs(model, DeterministicAdapter(), optimizer, 5, 1, 1,
                             metrics, lambda: None, early_stopping_patience=2,
                             early_stopping_min_delta=0.01)
        return model, metrics

    expected_model, expected_metrics = baseline()
    interrupted_model = ControlledLoss()
    interrupted_optimizer = torch.optim.SGD(interrupted_model.parameters(), lr=0.1)
    interrupted_metrics = _metrics()

    def publish_after_baseline():
        if len(interrupted_metrics["epochs"]) == 1:
            marker.touch()

    with pytest.raises(SystemExit) as stopped:
        worker._train_epochs(
            interrupted_model, DeterministicAdapter(), interrupted_optimizer, 5, 1, 1,
            interrupted_metrics, publish_after_baseline, checkpoint_request=marker,
            checkpoint_output=tmp_path, job_id="early-stop-job", scheduler_job_id="123",
            early_stopping_patience=2, early_stopping_min_delta=0.01,
        )
    assert stopped.value.code == worker.CHECKPOINT_EXIT_CODE
    marker.unlink()

    checkpoint = worker._load_checkpoint(tmp_path, "early-stop-job")
    assert checkpoint["early_stopping_state"]["best_epoch"] == 1
    resumed_model = ControlledLoss()
    resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.1)
    resumed_model.load_state_dict(checkpoint["model_state"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer_state"])
    worker._train_epochs(
        resumed_model, DeterministicAdapter(), resumed_optimizer, 5, 1, 1,
        checkpoint["metrics"], lambda: None, resume=checkpoint,
        early_stopping_patience=2, early_stopping_min_delta=0.01,
    )

    torch.testing.assert_close(resumed_model.weight, expected_model.weight)
    assert checkpoint["metrics"] == expected_metrics


def test_metrics_reader_preserves_and_validates_early_stopping(tmp_path):
    store = JobStore(tmp_path / "jobs")
    job = store.create({"manifest": {"schemaVersion": 2}, "nodes": [], "edges": []}, {}, {}, {}, "owner")
    path = store.job_dir(job["id"]) / "output/metrics.json"
    metrics = {
        "epochs": [{"epoch": 1, "training_loss": 1.0, "validation_loss": 0.5}],
        "steps": [], "test_loss": None,
        "early_stopping": {"best_epoch": 1, "best_validation_loss": 0.5,
                           "non_improving_epochs": 0, "stopped_epoch": None},
    }
    import json

    path.write_text(json.dumps(metrics))
    assert store.get(job["id"], "owner")["metrics"] == metrics
    metrics["early_stopping"]["stopped_epoch"] = True
    path.write_text(json.dumps(metrics))
    assert store.get(job["id"], "owner")["metrics"] == {"epochs": [], "test_loss": None, "steps": []}
