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
EXAMPLE = REPOSITORY / "examples/models/local-training"


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
        if path.is_file() and "__pycache__" not in path.parts:
            relative = path.relative_to(EXAMPLE).as_posix()
            files[relative] = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "project": project,
        "files": files,
        "training": {"epochs": 2, "batch_size": 2, "learning_rate": 0.05, "seed": 7},
    }


def payload_for(example: str) -> dict:
    directory = REPOSITORY / "examples" / "models" / example
    project = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    files = {
        path.relative_to(directory).as_posix(): base64.b64encode(path.read_bytes()).decode("ascii")
        for path in directory.rglob("*")
        if path.is_file() and path.name != "model.json" and "__pycache__" not in path.parts
    }
    return {
        "project": project,
        "files": files,
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": 0.01, "seed": 7},
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
        assert job["metrics"] == {"epochs": [], "test_loss": None, "steps": []}
        saved_training = json.loads((store.job_dir(job["id"]) / "snapshot/training.json").read_text())
        assert saved_training["publish_every_steps"] == 10
        assert runner.submitted == [job["id"]]
        assert http.get("/v1/jobs").json() == {"jobs": [job]}
        assert http.get(f"/v1/jobs/{job['id']}/snapshot").json() == {
            "project": submitted["project"], "files": submitted["files"]
        }
        assert store.job_dir(job["id"]).joinpath("snapshot").stat().st_mode & 0o222 == 0


def test_schema_three_submission_freezes_authored_and_concrete_graphs(tmp_path, monkeypatch):
    http, store, runner = client(tmp_path, monkeypatch)
    with http:
        submitted = payload()
        source = submitted["project"]
        source["manifest"]["schemaVersion"] = 3
        source["manifest"]["metavariables"] = [{"name": "width", "type": "integer", "value": 1}]
        source["nodes"][1]["data"]["params"]["in_features"] = {"$ref": "width"}
        execution = json.loads(json.dumps(source))
        execution["manifest"]["schemaVersion"] = 2
        execution["manifest"].pop("metavariables")
        execution["nodes"][1]["data"]["params"]["in_features"] = 1
        submitted["execution_project"] = execution

        response = http.post("/v1/jobs", json=submitted)
        assert response.status_code == 202, response.content.decode()
        job = response.json()
        snapshot = store.job_dir(job["id"]) / "snapshot"
        assert json.loads((snapshot / "project.json").read_text()) == source
        assert json.loads((snapshot / "execution-project.json").read_text()) == execution
        assert http.get(f"/v1/jobs/{job['id']}/snapshot").json()["project"] == source
        assert runner.submitted == [job["id"]]


def test_schema_three_requires_execution_project_and_rejects_literal_changes(tmp_path, monkeypatch):
    http, _store, _runner = client(tmp_path, monkeypatch)
    with http:
        submitted = payload()
        submitted["project"]["manifest"]["schemaVersion"] = 3
        missing = http.post("/v1/jobs", json=submitted)
        assert missing.status_code == 422

        source = submitted["project"]
        source["manifest"]["metavariables"] = [{"name": "width", "type": "integer", "value": 1}]
        source["nodes"][1]["data"]["params"]["in_features"] = {"$ref": "width"}
        execution = json.loads(json.dumps(source))
        execution["manifest"]["schemaVersion"] = 2
        execution["manifest"].pop("metavariables")
        execution["nodes"][1]["data"]["params"]["in_features"] = 1
        execution["nodes"][1]["data"]["params"]["bias"] = False
        submitted["execution_project"] = execution
        rejected = http.post("/v1/jobs", json=submitted)
        assert rejected.status_code == 422
        assert "literal parameter" in rejected.json()["detail"]


def test_early_stopping_controls_are_accepted_frozen_and_returned(tmp_path, monkeypatch):
    http, store, runner = client(tmp_path, monkeypatch)
    with http:
        submitted = payload()
        submitted["training"].update(early_stopping_patience=4, early_stopping_min_delta=0)
        response = http.post("/v1/jobs", json=submitted)
        assert response.status_code == 202, response.content.decode()
        job = response.json()
        training = json.loads((store.job_dir(job["id"]) / "snapshot/training.json").read_text())
        assert training["early_stopping_patience"] == 4
        assert training["early_stopping_min_delta"] == 0.0
        assert runner.submitted == [job["id"]]

        output_metrics = store.job_dir(job["id"]) / "output/metrics.json"
        early = {
            "best_epoch": 2,
            "best_validation_loss": 0.125,
            "non_improving_epochs": 3,
            "stopped_epoch": 5,
        }
        output_metrics.write_text(json.dumps({
            "epochs": [
                {"epoch": 1, "training_loss": 1.0, "validation_loss": 0.25},
                {"epoch": 2, "training_loss": 0.8, "validation_loss": 0.125},
                {"epoch": 3, "training_loss": 0.7, "validation_loss": 0.2},
                {"epoch": 4, "training_loss": 0.6, "validation_loss": 0.3},
                {"epoch": 5, "training_loss": 0.5, "validation_loss": 0.4},
            ],
            "steps": [], "test_loss": 0.2, "early_stopping": early,
        }))
        assert http.get(f"/v1/jobs/{job['id']}").json()["metrics"]["early_stopping"] == early
        assert http.get("/v1/jobs").json()["jobs"][0]["metrics"]["early_stopping"] == early

        for name, invalid_value in (
            ("early_stopping_patience", True),
            ("early_stopping_patience", 10001),
            ("early_stopping_patience", 1.0),
            ("early_stopping_min_delta", True),
            ("early_stopping_min_delta", -0.1),
            ("early_stopping_min_delta", float("inf")),
            ("early_stopping_min_delta", "0.1"),
        ):
            invalid = payload()
            invalid["training"][name] = invalid_value
            assert http.post("/v1/jobs", json=invalid).status_code == 422, name
        assert runner.submitted == [job["id"]]


def test_nested_input_boundary_does_not_require_dataset_binding(tmp_path, monkeypatch):
    http, _store, runner = client(tmp_path, monkeypatch)
    submitted = payload_for("rnn-sine")
    nested = next(node for node in submitted["project"]["nodes"] if node["id"] == "rnn-input")
    nested["data"]["params"].pop("binding")
    root_input = next(node for node in submitted["project"]["nodes"] if node["id"] == "sequence")
    root_input["data"]["params"].pop("binding")
    with http:
        invalid = http.post("/v1/jobs", json=submitted)
        assert invalid.status_code == 422
        assert runner.submitted == []
        root_input["data"]["params"]["binding"] = "sequence"
        response = http.post("/v1/jobs", json=submitted)
        assert response.status_code == 202, response.content.decode()
        assert runner.submitted == [response.json()["id"]]


def test_external_objective_inputs_accept_only_declared_dataset_slots(tmp_path, monkeypatch):
    http, _store, runner = client(tmp_path, monkeypatch)

    def job_with_source(source: str) -> dict:
        job = payload_for("tiny-decoder-llm")
        package = next(
            ref
            for ref in job["project"]["manifest"]["customPackages"]
            if ref["id"] == "llm.token-cross-entropy"
        )
        manifest_path = f"{package['path']}/manifest.json"
        package_manifest = json.loads(base64.b64decode(job["files"][manifest_path]))
        definition_path = f"{package['path']}/{package_manifest['entrypoints']['definition']}"
        definition = json.loads(base64.b64decode(job["files"][definition_path]))
        definition["objective"]["externalInputs"][0]["source"] = source
        job["files"][definition_path] = base64.b64encode(json.dumps(definition).encode()).decode()
        return job

    with http:
        # The legacy target binding and the new root-input binding both validate.
        target_job = job_with_source("batch.targets.target")
        assert http.post("/v1/jobs", json=target_job).status_code == 202
        input_job = job_with_source("batch.inputs.tokens")
        assert http.post("/v1/jobs", json=input_job).status_code == 202

        invalid = job_with_source("batch.inputs.undeclared")
        response = http.post("/v1/jobs", json=invalid)
        assert response.status_code == 422
        assert "undeclared dataset input" in response.json()["detail"]
        assert len(runner.submitted) == 2


def test_publish_every_steps_is_optional_and_strictly_bounded(tmp_path, monkeypatch):
    http, _, runner = client(tmp_path, monkeypatch)
    with http:
        valid = payload()
        valid["training"]["publish_every_steps"] = 1
        assert http.post("/v1/jobs", json=valid).status_code == 202
        for invalid_value in (0, 100001, True, 1.0, "2"):
            invalid = payload()
            invalid["training"]["publish_every_steps"] = invalid_value
            assert http.post("/v1/jobs", json=invalid).status_code == 422
        assert len(runner.submitted) == 1


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


def test_wheel_download_preserves_readable_and_historical_artifact_names(tmp_path, monkeypatch):
    http, store, _ = client(tmp_path, monkeypatch)
    with http:
        for filename in ("nnm_llm-0.1.0-py3-none-any.whl", "nnmodel_job_old-0.1.0-py3-none-any.whl"):
            job_id = http.post("/v1/jobs", json=payload()).json()["id"]
            artifact = store.job_dir(job_id) / "output" / filename
            artifact.write_bytes(b"stored wheel bytes")
            assert http.get(f"/v1/jobs/{job_id}/wheel").status_code == 409
            store.update(job_id, status="completed")
            response = http.loop.run_until_complete(service.get_wheel(job_id, "local-owner"))
            assert response.status_code == 200
            assert response.headers["content-disposition"] == f'attachment; filename="{filename}"'
            assert Path(response.path) == artifact
            assert artifact.read_bytes() == b"stored wheel bytes"
            assert http.get(f"/v1/jobs/{job_id}/package").status_code == 404


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
