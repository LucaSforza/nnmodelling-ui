from __future__ import annotations

import base64
import binascii
import hmac
import json
import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt
from typing import Literal

from . import config
from .runner import JobRunner, container_status
from .store import JobStore, utc_now
from .validation import resolve_core_packages, safe_relative_path, validate_project, validate_training


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    epochs: StrictInt = Field(ge=1, le=10000)
    batch_size: StrictInt = Field(ge=1, le=4096)
    learning_rate: StrictFloat = Field(gt=0, le=1, allow_inf_nan=False)
    seed: StrictInt = Field(ge=-(2**31), lt=2**32)
    publish_every_steps: StrictInt = Field(default=10, ge=1, le=100000)


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project: dict[str, Any]
    files: dict[str, str]
    training: TrainingConfig


class EpochMetric(BaseModel):
    epoch: int
    training_loss: float
    validation_loss: float


class StepMetric(BaseModel):
    step: int
    epoch: int
    training_loss: float
    validation_loss: float


class Metrics(BaseModel):
    epochs: list[EpochMetric]
    test_loss: float | None
    steps: list[StepMetric] = Field(default_factory=list)


class JobResponse(BaseModel):
    id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    created_at: str
    error: str | None
    metrics: Metrics


class SnapshotResponse(BaseModel):
    project: dict[str, Any]
    files: dict[str, str]


class RequestSizeLimit:
    def __init__(self, app, maximum: int):
        self.app = app
        self.maximum = maximum

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        state = {"size": 0, "oversized": False, "responded": False}

        async def limited_receive():
            event = await receive()
            if event["type"] == "http.request":
                state["size"] += len(event.get("body", b""))
                if state["size"] > self.maximum:
                    state["oversized"] = True
                    return {"type": "http.disconnect"}
            return event

        async def limited_send(event):
            if state["oversized"]:
                if event["type"] == "http.response.start":
                    body = b'{"detail":"Request body exceeds the configured bundle limit."}'
                    state["responded"] = True
                    await send({"type": "http.response.start", "status": 413,
                                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
                    await send({"type": "http.response.body", "body": body})
                return
            await send(event)

        try:
            await self.app(scope, limited_receive, limited_send)
        except Exception:
            if not state["oversized"]:
                raise
            if not state["responded"]:
                body = b'{"detail":"Request body exceeds the configured bundle limit."}'
                await send({"type": "http.response.start", "status": 413,
                            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
                await send({"type": "http.response.body", "body": body})


store = JobStore()
runner = JobRunner(store)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    runner.start()
    try:
        yield
    finally:
        runner.shutdown()


app = FastAPI(title="NNModelling local training API", version="1.0.0", lifespan=lifespan)
app.add_middleware(RequestSizeLimit, maximum=(config.MAX_BUNDLE_BYTES * 4 // 3) + config.MAX_PROJECT_BYTES + 1024 * 1024)


async def current_owner(authorization: str | None = Header(default=None)) -> str:
    expected = config.AUTH_TOKEN
    if expected:
        provided = authorization.removeprefix("Bearer ") if authorization and authorization.startswith("Bearer ") else ""
        if not hmac.compare_digest(provided, expected):
            raise HTTPException(status_code=401, detail="A valid bearer token is required.", headers={"WWW-Authenticate": "Bearer"})
        return "configured-owner"
    return "local-owner"


@app.get("/health")
async def health() -> dict[str, Any]:
    runtime = await asyncio.to_thread(container_status)
    return {"status": "ok", "container": runtime}


@app.post("/v1/jobs", status_code=202, response_model=JobResponse)
async def create_job(body: JobRequest, owner: str = Depends(current_owner)) -> dict[str, Any]:
    project = body.project
    training = validate_training(body.training.model_dump())
    if len(body.files) > config.MAX_FILES:
        raise HTTPException(413, f"At most {config.MAX_FILES} files may be submitted.")
    files: dict[str, bytes] = {}
    seen: set[str] = set()
    seen_paths: list[tuple[str, ...]] = []
    total = 0
    for raw_path, encoded in body.files.items():
        path = safe_relative_path(raw_path)
        identity = path.as_posix().casefold()
        if identity in seen:
            raise HTTPException(422, f"Duplicate project file identity: {raw_path!r}.")
        parts = tuple(part.casefold() for part in path.parts)
        if any(parts[:len(previous)] == previous or previous[:len(parts)] == parts for previous in seen_paths):
            raise HTTPException(422, f"File and directory paths collide: {raw_path!r}.")
        seen.add(identity)
        seen_paths.append(parts)
        try:
            data = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (UnicodeEncodeError, binascii.Error, ValueError):
            raise HTTPException(422, f"File {raw_path!r} is not valid base64.") from None
        if len(data) > config.MAX_FILE_BYTES:
            raise HTTPException(413, f"File {raw_path!r} exceeds the per-file size limit.")
        total += len(data)
        if total > config.MAX_BUNDLE_BYTES:
            raise HTTPException(413, "Decoded project bundle exceeds the configured size limit.")
        files[path.as_posix()] = data
    project_bytes = len(json.dumps(project, separators=(",", ":")).encode("utf-8"))
    if project_bytes > config.MAX_PROJECT_BYTES:
        raise HTTPException(413, "Project model exceeds the configured size limit.")
    core_packages = resolve_core_packages(project, files)
    project = validate_project(project, files, core_packages)
    record = store.create(project, files, training, core_packages, owner)
    runner.submit(record["id"])
    return record


@app.get("/v1/jobs", response_model=dict[str, list[JobResponse]])
async def list_jobs(owner: str = Depends(current_owner)) -> dict[str, Any]:
    return {"jobs": store.list(owner)}


def _job(job_id: str, owner: str) -> dict[str, Any]:
    record = store.get(job_id, owner)
    if record is None:
        raise HTTPException(404, "Job not found.")
    return record


@app.get("/v1/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, owner: str = Depends(current_owner)) -> dict[str, Any]:
    return _job(job_id, owner)


@app.get("/v1/jobs/{job_id}/snapshot", response_model=SnapshotResponse)
async def get_snapshot(job_id: str, owner: str = Depends(current_owner)) -> dict[str, Any]:
    value = store.snapshot_response(job_id, owner)
    if value is None:
        raise HTTPException(404, "Job not found.")
    return value


@app.get("/v1/jobs/{job_id}/weights")
async def get_weights(job_id: str, owner: str = Depends(current_owner)) -> FileResponse:
    path = store.artifact(job_id, owner, "weights")
    if path is None:
        _job(job_id, owner)
        raise HTTPException(409, "Weights are available only after successful training.")
    return FileResponse(path, media_type="application/octet-stream", filename="weights.safetensors")


@app.get("/v1/jobs/{job_id}/wheel")
async def get_wheel(job_id: str, owner: str = Depends(current_owner)) -> FileResponse:
    path = store.artifact(job_id, owner, "wheel")
    if path is None:
        _job(job_id, owner)
        raise HTTPException(409, "The model wheel is available only after successful training.")
    return FileResponse(path, media_type="application/zip", filename=path.name)


@app.post("/v1/jobs/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(job_id: str, owner: str = Depends(current_owner)) -> dict[str, Any]:
    record = _job(job_id, owner)
    if record["status"] == "queued":
        if store.transition(job_id, "queued", status="cancelled", error=None, finished_at=utc_now()) is not None:
            return _job(job_id, owner)
        record = _job(job_id, owner)
    if record["status"] == "running":
        if store.transition(job_id, "running", cancel_requested=True) is not None:
            runner.cancel(job_id)
            return _job(job_id, owner)
        record = _job(job_id, owner)
    if record["status"] == "cancelled":
        return record
    raise HTTPException(409, f"A {record['status']} job cannot be cancelled.")
