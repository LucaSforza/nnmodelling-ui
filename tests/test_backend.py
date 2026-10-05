from __future__ import annotations

import base64
import asyncio
import json
import os
from pathlib import Path

os.environ.setdefault("NNMODELLING_JOB_ROOT", "/tmp/nnmodelling-backend-test-import")

import backend.app as service
from backend.runner import JobRunner
from backend.store import JobStore, utc_now
from backend.app import RequestSizeLimit


REPOSITORY = Path(__file__).resolve().parents[1]
EXAMPLE = REPOSITORY / "examples/local-training"


class RunnerStub:
    def __init__(self) -> None:
        self.submitted: list[str] = []
        self.cancelled: list[str] = []

    def start(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def submit(self, job_id: str) -> None:
        self.submitted.append(job_id)

    def cancel(self, job_id: str) -> None:
        self.cancelled.append(job_id)


class LocalResponse:
    def __init__(self, events):
        start = next(event for event in events if event["type"] == "http.response.start")
        self.status_code = start["status"]
        self.headers = {key.decode().lower(): value.decode() for key, value in start["headers"]}
        self.content = b"".join(event.get("body", b"") for event in events if event["type"] == "http.response.body")

    def json(self):
        return json.loads(self.content)


class LocalClient:
    def __init__(self, application):
        self.app = application
        self.loop = asyncio.new_event_loop()
        self.lifespan = application.router.lifespan_context(application)
        self.loop.run_until_complete(self.lifespan.__aenter__())

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.loop.run_until_complete(self.lifespan.__aexit__(None, None, None))
        self.loop.close()

    def _request(self, method: str, path: str, **kwargs):
        body = json.dumps(kwargs["json"]).encode("utf-8") if "json" in kwargs else b""
        headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
        for key, value in kwargs.get("headers", {}).items():
            headers.append((key.lower().encode(), value.encode()))
        events = []
        received = False

        async def receive():
            nonlocal received
            if received:
                return {"type": "http.disconnect"}
            received = True
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(event):
            events.append(event)

        async def dispatch():
            await self.app({
                "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
                "http_version": "1.1", "method": method, "scheme": "http",
                "path": path, "raw_path": path.encode(), "query_string": b"",
                "root_path": "", "headers": headers, "client": ("test", 1234),
                "server": ("test", 80), "state": {},
            }, receive, send)

        self.loop.run_until_complete(dispatch())
        return LocalResponse(events)

    def get(self, path: str, **kwargs):
        return self._request("GET", path, **kwargs)

    def post(self, path: str, **kwargs):
        return self._request("POST", path, **kwargs)


def payload() -> dict:
    project = json.loads((EXAMPLE / "model.json").read_text(encoding="utf-8"))
    files = {}
    for path in (EXAMPLE / "datasets/tiny-regression").rglob("*"):
        if path.is_file():
            relative = path.relative_to(EXAMPLE).as_posix()
            files[relative] = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "project": project,
        "files": files,
        "training": {"epochs": 2, "batch_size": 2, "learning_rate": 0.05, "seed": 7},
    }


def client(tmp_path: Path, monkeypatch) -> tuple[LocalClient, JobStore, RunnerStub]:
    store = JobStore(tmp_path / "jobs")
    runner = RunnerStub()
    monkeypatch.setattr(service, "store", store)
    monkeypatch.setattr(service, "runner", runner)
    monkeypatch.setattr(service.config, "AUTH_TOKEN", None)
    return LocalClient(service.app), store, runner


def test_submit_snapshot_and_list_keep_original_resource_bytes(tmp_path, monkeypatch):
    http, store, runner = client(tmp_path, monkeypatch)
    with http:
        submitted = payload()
        response = http.post("/v1/jobs", json=submitted)
        assert response.status_code == 202
        job = response.json()
        assert job["status"] == "queued"
        assert job["metrics"] == {"epochs": [], "test_loss": None}
        assert runner.submitted == [job["id"]]
        assert http.get("/v1/jobs").json() == {"jobs": [job]}
        assert http.get(f"/v1/jobs/{job['id']}/snapshot").json() == {
            "project": submitted["project"], "files": submitted["files"]
        }
        assert store.job_dir(job["id"]).joinpath("snapshot").stat().st_mode & 0o222 == 0


def test_rejects_unsafe_and_colliding_paths_and_invalid_graph(tmp_path, monkeypatch):
    http, _, runner = client(tmp_path, monkeypatch)
    with http:
        base = payload()
        for path in ("../escape", "/absolute", "datasets\\escape", "datasets/./escape"):
            invalid = {**base, "files": {**base["files"], path: ""}}
            assert http.post("/v1/jobs", json=invalid).status_code == 422
        invalid = {**base, "files": {"datasets/conflict": "", "datasets/conflict/file": ""}}
        assert http.post("/v1/jobs", json=invalid).status_code == 422
        invalid = json.loads(json.dumps(base))
        invalid["project"]["nodes"].append(invalid["project"]["nodes"][0])
        assert http.post("/v1/jobs", json=invalid).status_code == 422
        invalid = json.loads(json.dumps(base))
        invalid["project"]["nodes"][1]["data"]["package"]["id"] = "core.not-activated"
        assert http.post("/v1/jobs", json=invalid).status_code == 422
        invalid = json.loads(json.dumps(base))
        invalid["project"]["nodes"][1]["data"]["package"]["id"] = ["core.linear"]
        assert http.post("/v1/jobs", json=invalid).status_code == 422
        invalid = json.loads(json.dumps(base))
        invalid["project"]["manifest"]["activeDataset"]["id"] = ["tiny-regression"]
        assert http.post("/v1/jobs", json=invalid).status_code == 422
        invalid = json.loads(json.dumps(base))
        invalid["project"]["edges"][0]["source"] = ["node-input"]
        assert http.post("/v1/jobs", json=invalid).status_code == 422
        assert runner.submitted == []


def test_auth_cancel_race_and_restart_recovery(tmp_path, monkeypatch):
    http, store, runner = client(tmp_path, monkeypatch)
    monkeypatch.setattr(service.config, "AUTH_TOKEN", "test-secret")
    with http:
        assert http.get("/v1/jobs").status_code == 401
        response = http.post("/v1/jobs", json=payload(), headers={"Authorization": "Bearer test-secret"})
        assert response.status_code == 202
        job_id = response.json()["id"]
        assert http.get(f"/v1/jobs/{job_id}", headers={"Authorization": "Bearer wrong"}).status_code == 401
        cancelled = http.post(f"/v1/jobs/{job_id}/cancel", headers={"Authorization": "Bearer test-secret"}).json()
        assert cancelled["status"] == "cancelled"

        running = store.create(payload()["project"], {}, payload()["training"], {}, "configured-owner")
        store.transition(running["id"], "queued", status="running")
        request_cancel = http.post(f"/v1/jobs/{running['id']}/cancel", headers={"Authorization": "Bearer test-secret"})
        assert request_cancel.status_code == 200
        assert request_cancel.json()["status"] == "running"
        assert store.internal(running["id"])["cancel_requested"] is True
        assert runner.cancelled == [running["id"]]

    recovered = JobStore(tmp_path / "jobs")
    recovered.recover()
    record = recovered.get(running["id"], "configured-owner")
    assert record["status"] == "failed"
    assert "restarted" in record["error"].lower()


def test_completed_download_refuses_worker_symlink_artifacts(tmp_path, monkeypatch):
    http, store, _ = client(tmp_path, monkeypatch)
    with http:
        response = http.post("/v1/jobs", json=payload())
        job_id = response.json()["id"]
        output = store.job_dir(job_id) / "output"
        outside = tmp_path / "outside-marker"
        outside.write_text("not an artifact", encoding="utf-8")
        (output / "weights.safetensors").symlink_to(outside)
        (output / "nnmodel_test-0.1.0-py3-none-any.whl").write_bytes(b"wheel")
        store.update(job_id, status="completed", metrics={"epochs": [{"epoch": 1, "training_loss": 2.0, "validation_loss": 1.0}], "test_loss": 0.5})
        assert http.get(f"/v1/jobs/{job_id}/weights").status_code == 409
        assert store.validated_result(job_id)[0] is None


def test_runner_cannot_start_a_job_cancelled_while_queued(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs")
    job = store.create(payload()["project"], {}, payload()["training"], {}, "local-owner")
    runner = JobRunner(store)
    store.transition(job["id"], "queued", status="cancelled")
    monkeypatch.setattr("backend.runner.container_status", lambda: (_ for _ in ()).throw(AssertionError("cancelled job reached runtime")))
    runner._execute(job["id"])
    assert store.get(job["id"], "local-owner")["status"] == "cancelled"


def test_invalid_job_ids_are_not_filesystem_paths(tmp_path, monkeypatch):
    http, _, _ = client(tmp_path, monkeypatch)
    with http:
        assert http.get("/v1/jobs/not-a-uuid").status_code == 404


def test_body_limit_counts_chunked_bytes_without_content_length():
    response_events = []
    chunks = iter([
        {"type": "http.request", "body": b"1234", "more_body": True},
        {"type": "http.request", "body": b"5678", "more_body": False},
    ])

    async def receive():
        return next(chunks)

    async def send(event):
        response_events.append(event)

    async def reading_app(scope, receive, send):
        while (await receive())["type"] != "http.disconnect":
            pass
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    asyncio.run(RequestSizeLimit(reading_app, maximum=7)({"type": "http", "headers": []}, receive, send))
    assert response_events[0]["status"] == 413
