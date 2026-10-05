from __future__ import annotations

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
