from __future__ import annotations

import json
import threading

from backend.store import JobStore


def test_queued_start_and_cancel_have_one_atomic_winner(tmp_path):
    store = JobStore(tmp_path / "jobs")
    job = store.create({"manifest": {"schemaVersion": 2}, "nodes": [], "edges": []}, {}, {}, {}, "owner")
    barrier = threading.Barrier(3)
    outcomes = []

    def transition(status):
        barrier.wait()
        outcomes.append(store.transition(job["id"], "queued", status=status))

    start = threading.Thread(target=transition, args=("running",))
    cancel = threading.Thread(target=transition, args=("cancelled",))
    start.start()
    cancel.start()
    barrier.wait()
    start.join()
    cancel.join()
    assert sum(result is not None for result in outcomes) == 1
    assert store.internal(job["id"])["status"] in {"running", "cancelled"}


def test_epoch_only_metrics_remain_readable_and_step_metrics_are_validated(tmp_path):
    store = JobStore(tmp_path / "jobs")
    job = store.create({"manifest": {"schemaVersion": 2}, "nodes": [], "edges": []}, {}, {}, {}, "owner")
    path = store.job_dir(job["id"]) / "output/metrics.json"
    old_metrics = {"epochs": [{"epoch": 1, "training_loss": 1.0, "validation_loss": 2.0}], "test_loss": None}
    path.write_text(json.dumps(old_metrics))
    assert store.get(job["id"], "owner")["metrics"] == old_metrics

    new_metrics = {
        **old_metrics,
        "steps": [
            {"step": 2, "epoch": 1, "training_loss": 1.5, "validation_loss": 2.0},
            {"step": 3, "epoch": 1, "training_loss": 0.5, "validation_loss": 1.0},
        ],
    }
    path.write_text(json.dumps(new_metrics))
    assert store.get(job["id"], "owner")["metrics"] == new_metrics

    new_metrics["steps"][1]["step"] = 2
    path.write_text(json.dumps(new_metrics))
    assert store.get(job["id"], "owner")["metrics"] == {"epochs": [], "test_loss": None, "steps": []}


def test_final_result_requires_epoch_and_finite_test_metrics(tmp_path):
    store = JobStore(tmp_path / "jobs")
    job = store.create({"manifest": {"schemaVersion": 2}, "nodes": [], "edges": []}, {}, {}, {}, "owner")
    output = store.job_dir(job["id"]) / "output"
    (output / "weights.safetensors").write_bytes(b"weights")
    (output / "model-0.1.0-py3-none-any.whl").write_bytes(b"wheel")
    path = output / "metrics.json"
    path.write_text(json.dumps({"epochs": [], "steps": [], "test_loss": None}))
    metrics, error = store.validated_result(job["id"])
    assert metrics is None and "epoch summary" in error

    path.write_text(json.dumps({
        "epochs": [{"epoch": 1, "training_loss": 1.0, "validation_loss": 0.5}],
        "steps": [{"step": 1, "epoch": 1, "training_loss": 1.0, "validation_loss": 0.5}],
        "test_loss": 0.25,
    }))
    metrics, error = store.validated_result(job["id"])
    assert error is None
    assert metrics["test_loss"] == 0.25
