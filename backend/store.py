from __future__ import annotations

import base64
import json
import os
import shutil
import threading
import uuid
import math
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from . import config


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


class JobStore:
    """Filesystem-backed immutable snapshots and atomically replaced job records."""

    def __init__(self, root: Path = config.JOB_ROOT) -> None:
        self.root = root
        self._lock = threading.RLock()

    def job_dir(self, job_id: str) -> Path:
        if not _valid_job_id(job_id):
            raise ValueError("Invalid job id.")
        return self.root / job_id

    def _metadata_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.json"

    def _read(self, job_id: str) -> dict[str, Any] | None:
        if not _valid_job_id(job_id):
            return None
        path = self._metadata_path(job_id)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _write(self, record: dict[str, Any]) -> None:
        _atomic_json(self._metadata_path(record["id"]), record)

    def recover(self, stop_errors: dict[str, str] | None = None) -> list[str]:
        """Mark work interrupted by a service restart and return queued jobs."""
        queued: list[str] = []
        with self._lock:
            if not self.root.is_dir():
                return queued
            for path in self.root.iterdir():
                if not path.is_dir():
                    continue
                record = self._read(path.name)
                if not record:
                    continue
                if record["status"] == "running":
                    message = "Backend restarted while this job was running."
                    if stop_errors and path.name in stop_errors:
                        message += " " + stop_errors[path.name]
                    record.update(status="failed", error=message, finished_at=utc_now())
                    self._write(record)
                elif record["status"] == "queued":
                    queued.append(path.name)
        return queued

    def interrupted_running(self) -> list[str]:
        ids: list[str] = []
        with self._lock:
            if not self.root.is_dir():
                return ids
            for path in self.root.iterdir():
                if path.is_dir():
                    record = self._read(path.name)
                    if record and record["status"] == "running":
                        ids.append(path.name)
        return ids

    def create(self, project: dict[str, Any], files: dict[str, bytes], training: dict[str, Any],
               core_packages: dict[str, Path], owner: str) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        job_dir = self.job_dir(job_id)
        snapshot = job_dir / "snapshot"
        out = job_dir / "output"
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            (snapshot / "files").mkdir(parents=True)
            out.mkdir()
            _atomic_json(snapshot / "project.json", project)
            _atomic_json(snapshot / "training.json", training)
            for rel, data in files.items():
                dest = snapshot / "files" / PurePosixPath(rel)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
            if core_packages:
                from .validation import copy_core_packages
                copy_core_packages(core_packages, snapshot / "core")
            record = {
                "id": job_id,
                "owner": owner,
                "status": "queued",
                "created_at": utc_now(),
                "started_at": None,
                "finished_at": None,
                "error": None,
                "metrics": {"epochs": [], "test_loss": None, "steps": []},
            }
            self._write(record)
            for path in snapshot.rglob("*"):
                if path.is_file():
                    path.chmod(0o444)
            for path in sorted(snapshot.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if path.is_dir():
                    path.chmod(0o555)
            snapshot.chmod(0o555)
            return self.public(record)
        except Exception:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise

    def update(self, job_id: str, **changes: Any) -> dict[str, Any] | None:
        with self._lock:
            record = self._read(job_id)
            if record is None:
                return None
            record.update(changes)
            self._write(record)
            return self.public(record)

    def transition(self, job_id: str, expected_status: str, **changes: Any) -> dict[str, Any] | None:
        """Atomically update only when a lifecycle state still matches."""
        with self._lock:
            record = self._read(job_id)
            if record is None or record["status"] != expected_status:
                return None
            record.update(changes)
            self._write(record)
            return self.public(record)

    def complete(self, job_id: str, metrics: dict[str, Any]) -> bool:
        """Commit success only if cancellation did not win the lifecycle race."""
        with self._lock:
            record = self._read(job_id)
            if record is None or record["status"] != "running" or record.get("cancel_requested"):
                return False
            record.update(status="completed", metrics=metrics, error=None, finished_at=utc_now())
            self._write(record)
            return True

    def get(self, job_id: str, owner: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._read(job_id)
            if record is None or record["owner"] != owner:
                return None
            metrics_path = self.job_dir(job_id) / "output" / "metrics.json"
            live_metrics = _read_metrics(metrics_path)
            if live_metrics is not None:
                record["metrics"] = live_metrics
            return self.public(record)

    def list(self, owner: str) -> list[dict[str, Any]]:
        records = []
        with self._lock:
            if not self.root.is_dir():
                return records
            for path in self.root.iterdir():
                if path.is_dir():
                    value = self.get(path.name, owner)
                    if value is not None:
                        records.append(value)
        return sorted(records, key=lambda x: x["created_at"], reverse=True)

    def owner_matches(self, job_id: str, owner: str) -> bool:
        with self._lock:
            record = self._read(job_id)
            return record is not None and record["owner"] == owner

    def internal(self, job_id: str) -> dict[str, Any] | None:
        """Return the service-private record for queue and lifecycle decisions."""
        with self._lock:
            return self._read(job_id)

    def snapshot_response(self, job_id: str, owner: str) -> dict[str, Any] | None:
        if not self.owner_matches(job_id, owner):
            return None
        job_dir = self.job_dir(job_id)
        project = json.loads((job_dir / "snapshot/project.json").read_text(encoding="utf-8"))
        files: dict[str, str] = {}
        root = job_dir / "snapshot/files"
        for path in root.rglob("*"):
            if path.is_file():
                files[path.relative_to(root).as_posix()] = base64.b64encode(path.read_bytes()).decode("ascii")
        return {"project": project, "files": files}

    def artifact(self, job_id: str, owner: str, name: str) -> Path | None:
        record = self.get(job_id, owner)
        if record is None or record["status"] != "completed":
            return None
        out = self.job_dir(job_id) / "output"
        if name == "weights":
            path = out / "weights.safetensors"
        elif name == "wheel":
            wheels = sorted(path for path in out.glob("*.whl") if _safe_output_file(path, out))
            path = wheels[0] if len(wheels) == 1 else None
        else:
            path = None
        return path if path is not None and _safe_output_file(path, out) else None

    def validated_result(self, job_id: str) -> tuple[dict[str, Any] | None, str | None]:
        """Read and validate result artifacts without following worker-created links."""
        output = self.job_dir(job_id) / "output"
        metrics_path = output / "metrics.json"
        metrics = _read_metrics(metrics_path)
        if metrics is None:
            return None, "worker completed without valid metrics.json"
        if not metrics["epochs"] or not _finite_number(metrics["test_loss"]):
            return None, "worker completed without an epoch summary and finite test loss"
        weights = output / "weights.safetensors"
        wheels = [path for path in output.glob("*.whl") if _safe_output_file(path, output)]
        if not _safe_output_file(weights, output) or len(wheels) != 1:
            return None, "worker completed without safe weights and exactly one wheel"
        return metrics, None

    def safe_output_file(self, job_id: str, name: str) -> Path | None:
        if Path(name).name != name or name in {".", ".."}:
            return None
        output = self.job_dir(job_id) / "output"
        path = output / name
        return path if _safe_output_file(path, output) else None

    @staticmethod
    def public(record: dict[str, Any]) -> dict[str, Any]:
        return {key: record[key] for key in ("id", "status", "created_at", "error", "metrics")}


def _valid_job_id(value: str) -> bool:
    try:
        return str(uuid.UUID(value)) == value
    except (ValueError, AttributeError, TypeError):
        return False


def _safe_output_file(path: Path, output: Path) -> bool:
    try:
        return not path.is_symlink() and path.is_file() and path.resolve(strict=True).parent == output.resolve(strict=True)
    except OSError:
        return False


def _read_metrics(path: Path) -> dict[str, Any] | None:
    try:
        if not _safe_output_file(path, path.parent) or path.stat().st_size > config.MAX_METRICS_BYTES:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        epochs = value["epochs"]
        test_loss = value["test_loss"]
        if not isinstance(epochs, list) or (test_loss is not None and not _finite_number(test_loss)):
            return None
        if not isinstance(value, dict) or set(value) not in ({"epochs", "test_loss"}, {"epochs", "test_loss", "steps"}):
            return None
        last_epoch = 0
        for item in epochs:
            if not isinstance(item, dict) or set(item) != {"epoch", "training_loss", "validation_loss"}:
                return None
            number = item["epoch"]
            if isinstance(number, bool) or not isinstance(number, int) or number <= last_epoch:
                return None
            if not _finite_number(item["training_loss"]) or not _finite_number(item["validation_loss"]):
                return None
            last_epoch = number
        result = {"epochs": epochs, "test_loss": test_loss}
        if "steps" in value:
            steps = value["steps"]
            if not isinstance(steps, list):
                return None
            last_step = 0
            last_step_epoch = 0
            for item in steps:
                if not isinstance(item, dict) or set(item) != {"step", "epoch", "training_loss", "validation_loss"}:
                    return None
                step, epoch = item["step"], item["epoch"]
                if (isinstance(step, bool) or not isinstance(step, int) or step <= last_step
                        or isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1
                        or epoch < last_step_epoch or epoch > last_step_epoch + 1):
                    return None
                if not _finite_number(item["training_loss"]) or not _finite_number(item["validation_loss"]):
                    return None
                last_step, last_step_epoch = step, epoch
            result["steps"] = steps
        return result
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
        return None


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
