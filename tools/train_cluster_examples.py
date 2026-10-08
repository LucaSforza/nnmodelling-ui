#!/usr/bin/env python3
"""Submit bundled model examples to a configured local Slurm backend."""
from __future__ import annotations

import argparse
import base64
import email.message
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[1] / "examples" / "models"
DEFAULT_OUTPUT_DIR = Path(".computer-use/cluster-training")
TERMINAL = {"completed", "failed", "cancelled"}
EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules", "cache", "caches"}


class TrainingError(RuntimeError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def safe_relative(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise TrainingError(f"Unsafe project resource path: {value!r}")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise TrainingError(f"Unsafe project resource path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise TrainingError(f"Unsafe project resource path: {value!r}")
    return path


def checked_path(root: Path, relative: PurePosixPath) -> Path:
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise TrainingError(f"Resource symlink is not allowed: {relative.as_posix()}")
    try:
        current.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        raise TrainingError(f"Resource path escapes or is missing: {relative.as_posix()}") from None
    return current


def iter_resource_files(root: Path, relative: PurePosixPath):
    directory = checked_path(root, relative)
    if not directory.is_dir():
        raise TrainingError(f"Manifest resource is not a directory: {relative.as_posix()}")
    for current, dirs, names in os.walk(directory, followlinks=False):
        base = Path(current)
        for name in list(dirs):
            child = base / name
            if child.is_symlink():
                raise TrainingError(f"Resource symlink is not allowed: {child.relative_to(root).as_posix()}")
            if name.lower() in EXCLUDED_DIRS:
                dirs.remove(name)
        for name in names:
            child = base / name
            rel = child.relative_to(root).as_posix()
            if child.is_symlink():
                raise TrainingError(f"Resource symlink is not allowed: {rel}")
            if name.lower() in EXCLUDED_DIRS or name.lower().endswith((".pyc", ".pyo")):
                continue
            yield child


def project_files(root: Path, project: dict[str, Any]) -> dict[str, str]:
    manifest = project.get("manifest")
    if not isinstance(manifest, dict):
        raise TrainingError(f"{root}: model manifest is missing")
    files: dict[str, str] = {}
    datasets = manifest.get("customDatasets", [])
    packages = manifest.get("customPackages", [])
    if not isinstance(datasets, list) or not isinstance(packages, list):
        raise TrainingError(f"{root}: manifest resource lists are invalid")
    resource_paths: list[PurePosixPath] = []
    for resource in datasets + packages:
        if not isinstance(resource, dict) or "path" not in resource:
            raise TrainingError(f"{root}: malformed manifest resource entry")
        relative = safe_relative(resource["path"])
        resource_paths.append(relative)
        for path in iter_resource_files(root, relative):
            files[path.relative_to(root).as_posix()] = base64.b64encode(path.read_bytes()).decode("ascii")

    # Manifests are inside declared resource roots. Only their explicit inference
    # assets may pull files from elsewhere in the project snapshot.
    for relative in resource_paths:
        manifest_path = checked_path(root, relative / "manifest.json")
        try:
            resource_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise TrainingError(f"Cannot read {relative.as_posix()}/manifest.json: {error}") from error
        if not isinstance(resource_manifest, dict):
            raise TrainingError(f"{relative.as_posix()}/manifest.json must contain a JSON object")
        assets = resource_manifest.get("inferenceAssets", [])
        if not isinstance(assets, list):
            raise TrainingError(f"{relative.as_posix()}/manifest.json has invalid inferenceAssets")
        for value in assets:
            asset = safe_relative(value)
            path = checked_path(root, asset)
            if not path.is_file():
                raise TrainingError(f"Inference asset is not a file: {asset.as_posix()}")
            files.setdefault(asset.as_posix(), base64.b64encode(path.read_bytes()).decode("ascii"))
    files.pop("model.json", None)
    return dict(sorted(files.items()))


def discover_projects(models_dir: Path) -> list[tuple[str, Path, dict[str, Any]]]:
    projects = []
    for directory in sorted(models_dir.iterdir(), key=lambda item: item.name.casefold()):
        if "deepseek" in directory.name.casefold():
            continue
        if directory.is_symlink():
            raise TrainingError(f"Model project directory cannot be a symlink: {directory}")
        if not directory.is_dir():
            continue
        model_path = directory / "model.json"
        if model_path.is_symlink():
            raise TrainingError(f"Project model.json cannot be a symlink: {model_path}")
        if not model_path.is_file():
            continue
        try:
            project = json.loads(model_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise TrainingError(f"Cannot read {model_path}: {error}") from error
        if not isinstance(project, dict):
            raise TrainingError(f"{model_path} must contain a JSON object")
        projects.append((directory.name, directory, project))
    return projects


def request(base_url: str, method: str, route: str, *, payload: Any = None, timeout: float = 30.0) -> tuple[int, dict[str, str], bytes]:
    data = None if payload is None else canonical_json(payload)
    headers = {"Accept": "application/json"}
    token = os.environ.get("NNMODELLING_BEARER_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base_url.rstrip("/") + route, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, {k.lower(): v for k, v in response.headers.items()}, response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise TrainingError(f"HTTP {error.code} {method} {route}: {detail}") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise TrainingError(f"Cannot reach backend at {base_url}: {error}") from error


def get_json(base_url: str, route: str) -> dict[str, Any]:
    status, _headers, body = request(base_url, "GET", route)
    if not 200 <= status < 300:
        raise TrainingError(f"HTTP {status} GET {route}")
    try:
        value = json.loads(body)
    except json.JSONDecodeError as error:
        raise TrainingError(f"Backend returned invalid JSON for {route}: {error}") from error
    if not isinstance(value, dict):
        raise TrainingError(f"Backend returned unexpected JSON for {route}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as target:
        target.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)


def safe_filename(headers: dict[str, str]) -> str:
    message = email.message.Message()
    message["content-disposition"] = headers.get("content-disposition", "")
    filename = message.get_filename()
    if not filename or Path(filename).name != filename or not re.fullmatch(r"[A-Za-z0-9._-]+", filename):
        raise TrainingError("Artifact response has no safe Content-Disposition filename")
    return filename


def download_artifact(base_url: str, job_id: str, kind: str, directory: Path) -> dict[str, str]:
    status, headers, body = request(base_url, "GET", f"/v1/jobs/{job_id}/{kind}", timeout=120)
    if not 200 <= status < 300:
        raise TrainingError(f"HTTP {status} downloading {kind} for job {job_id}")
    filename = safe_filename(headers)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    return {"filename": filename, "path": str(target), "sha256": sha256(body)}


def training_args(args: argparse.Namespace) -> dict[str, Any]:
    return {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "seed": args.seed, "publish_every_steps": args.publish_every_steps}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    result.add_argument("--base-url", default="http://127.0.0.1:8765")
    result.add_argument("--epochs", type=int, default=3)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--learning-rate", type=float, default=0.001)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--publish-every-steps", type=int, default=10)
    result.add_argument("--wait", action="store_true", help="poll jobs to a terminal status and download completed artifacts")
    result.add_argument("--wait-timeout", type=int, default=3600, help="maximum seconds spent polling jobs (1..86400)")
    result.add_argument("--poll-interval", type=float, default=5.0, help="seconds between status polls (minimum 1)")
    return result


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.epochs <= 10000:
        raise TrainingError("--epochs must be between 1 and 10000")
    if not 1 <= args.batch_size <= 4096:
        raise TrainingError("--batch-size must be between 1 and 4096")
    if not 0 < args.learning_rate <= 1:
        raise TrainingError("--learning-rate must be greater than 0 and at most 1")
    if not -(2**31) <= args.seed < 2**32:
        raise TrainingError("--seed must be in [-2^31, 2^32)")
    if not 1 <= args.publish_every_steps <= 100000:
        raise TrainingError("--publish-every-steps must be between 1 and 100000")
    if not 1 <= args.wait_timeout <= 86400:
        raise TrainingError("--wait-timeout must be between 1 and 86400")
    if not math.isfinite(args.poll_interval) or args.poll_interval < 1:
        raise TrainingError("--poll-interval must be finite and at least 1 second")


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    projects = discover_projects(args.models_dir)
    if not projects:
        raise TrainingError(f"No model.json projects found under {args.models_dir}")
    health = get_json(args.base_url, "/health")
    executor = health.get("container", {}).get("executor")
    available = health.get("container", {}).get("available")
    if executor != "slurm" or available is not True:
        raise TrainingError(f"Refusing submission: backend health must report container.executor='slurm' and available=true (got {executor!r}, {available!r})")

    config = training_args(args)
    report_path = args.output_dir / "report.json"
    requested = []
    snapshots = []
    for name, root, project in projects:
        model_bytes = (root / "model.json").read_bytes()
        files = project_files(root, project)
        file_hashes = [(path, sha256(base64.b64decode(encoded))) for path, encoded in files.items()]
        snapshot_hash = sha256(canonical_json({"model_sha256": sha256(model_bytes), "files": file_hashes}))
        requested.append({"name": name, "project_path": str((root / "model.json").resolve()),
                          "model_sha256": sha256(model_bytes), "snapshot_sha256": snapshot_hash,
                          "training": config})
        snapshots.append(files)
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TrainingError(f"Cannot resume from {report_path}: {error}") from error
        if report.get("base_url") != args.base_url or report.get("executor") != "slurm":
            raise TrainingError(f"Existing report {report_path} belongs to a different backend; choose a new --output-dir")
        if report.get("projects") != requested:
            raise TrainingError(f"Existing report {report_path} is incompatible with these project snapshots/configuration; choose a new --output-dir")
        entries = report.get("jobs")
        if not isinstance(entries, list) or [item.get("name") for item in entries] != [item["name"] for item in requested]:
            raise TrainingError(f"Existing report {report_path} has an invalid jobs list")
    else:
        entries = [{**item, "status": "not_submitted", "job_id": None, "metrics": None, "error": None, "artifacts": {}}
                   for item in requested]
        report = {"base_url": args.base_url, "executor": "slurm", "training": config, "projects": requested, "jobs": entries}
        atomic_json(report_path, report)

    deadline = time.monotonic() + args.wait_timeout
    for entry, (_name, root, project), files in zip(entries, projects, snapshots):
        if entry.get("job_id") is None:
            body = {"project": project, "files": files, "training": config}
            status, _headers, response = request(args.base_url, "POST", "/v1/jobs", payload=body)
            if status not in {200, 202}:
                raise TrainingError(f"Unexpected submission status {status} for {entry['name']}")
            job = json.loads(response)
            if not isinstance(job, dict) or not job.get("id"):
                raise TrainingError(f"Backend returned no job id for {entry['name']}")
            entry.update(job_id=job["id"], status=job.get("status", "queued"), metrics=job.get("metrics"), error=job.get("error"))
            atomic_json(report_path, report)
        if entry.get("status") in {"failed", "cancelled"}:
            raise TrainingError(f"Job {entry['job_id']} for {entry['name']} ended {entry['status']}: {entry.get('error') or 'no backend error supplied'}")
        if not args.wait:
            continue
        if entry.get("status") == "completed":
            artifact_dir = args.output_dir / entry["name"]
            for kind in ("weights", "wheel"):
                if kind not in entry.get("artifacts", {}):
                    entry.setdefault("artifacts", {})[kind] = download_artifact(args.base_url, entry["job_id"], kind, artifact_dir)
                    atomic_json(report_path, report)
            continue
        while True:
            job = get_json(args.base_url, f"/v1/jobs/{entry['job_id']}")
            entry.update(status=job.get("status"), metrics=job.get("metrics"), error=job.get("error"))
            atomic_json(report_path, report)
            if entry["status"] in TERMINAL:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                entry["error"] = f"Polling timed out after {args.wait_timeout} seconds; job remains {entry['status']}"
                atomic_json(report_path, report)
                raise TrainingError(entry["error"])
            time.sleep(min(args.poll_interval, remaining))
        if entry["status"] == "completed":
            artifact_dir = args.output_dir / entry["name"]
            entry["artifacts"] = {kind: download_artifact(args.base_url, entry["job_id"], kind, artifact_dir)
                                  for kind in ("weights", "wheel")}
            atomic_json(report_path, report)
        elif entry["status"] in {"failed", "cancelled"}:
            atomic_json(report_path, report)
            raise TrainingError(f"Job {entry['job_id']} for {entry['name']} ended {entry['status']}: {entry.get('error') or 'no backend error supplied'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return run(args)
    except (TrainingError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"cluster training: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
