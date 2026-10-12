from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.parse
import urllib.request
import uuid
from pathlib import Path, PurePosixPath
from typing import Any



_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_FORBIDDEN = {"", ".", "..", ".git", ".venv", "venv", "__pycache__"}


def _relative(value: Any, label: str) -> PurePosixPath:
    if (not isinstance(value, str) or not value or "\\" in value or "\x00" in value
            or any(part in _FORBIDDEN for part in value.split("/"))):
        raise ValueError(f"{label} must be a confined relative POSIX path.")
    path = PurePosixPath(value)
    if path.is_absolute() or ":" in path.parts[0]:
        raise ValueError(f"{label} must be a confined relative POSIX path.")
    return path


def validate_remote_files(manifest: dict[str, Any], label: str = "Dataset") -> list[dict[str, Any]]:
    """Return validated immutable public-HF file records from a dataset manifest."""
    if "remoteFiles" not in manifest:
        return []
    value = manifest["remoteFiles"]
    if not isinstance(value, dict) or set(value) != {"repo", "revision", "files"}:
        raise ValueError(f"{label} remoteFiles must contain repo, revision, and files.")
    repo, revision, files = value["repo"], value["revision"], value["files"]
    if not isinstance(repo, str) or not _REPO.fullmatch(repo):
        raise ValueError(f"{label} remoteFiles.repo must be a public Hugging Face owner/repository.")
    if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
        raise ValueError(f"{label} remoteFiles.revision must be an immutable 40-hex commit.")
    if not isinstance(files, list):
        raise ValueError(f"{label} remoteFiles.files must be an ordered array.")
    records: list[dict[str, Any]] = []
    paths: set[str] = set()
    destinations: set[str] = set()
    destination_parts: list[tuple[str, ...]] = []
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "destination", "size", "sha256"}:
            raise ValueError(f"{label} remoteFiles entries require path, destination, size, and sha256.")
        try:
            path = _relative(item["path"], "Remote file path")
            destination = _relative(item["destination"], "Remote file destination")
        except ValueError as error:
            raise ValueError(f"{label}: {error}") from error
        size, digest = item["size"], item["sha256"]
        if type(size) is not int or size <= 0:
            raise ValueError(f"{label} remote file size must be a positive integer.")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"{label} remote file sha256 must contain 64 hexadecimal characters.")
        path_name = path.as_posix().casefold()
        destination_name = destination.as_posix().casefold()
        parts = tuple(part.casefold() for part in destination.parts)
        if (path_name in paths or destination_name in destinations
                or any(parts[:len(old)] == old or old[:len(parts)] == parts for old in destination_parts)):
            raise ValueError(f"{label} remoteFiles paths and destinations must be unique.")
        paths.add(path_name)
        destinations.add(destination_name)
        destination_parts.append(parts)
        records.append({"path": path.as_posix(), "destination": destination.as_posix(),
                        "size": size, "sha256": digest.lower(), "repo": repo,
                        "revision": revision.lower()})
    return records


def remote_records(project: dict[str, Any], files: dict[str, bytes]) -> list[dict[str, Any]]:
    result = []
    destinations: set[str] = set()
    manifest = project.get("manifest", {})
    active = manifest.get("activeDataset") if isinstance(manifest, dict) else None
    refs = manifest.get("customDatasets", []) if isinstance(manifest, dict) else []
    for ref in refs:
        if not isinstance(active, dict) or (ref.get("id"), ref.get("version")) != (active.get("id"), active.get("version")):
            continue
        prefix = ref["path"].rstrip("/")
        try:
            manifest = json.loads(files[f"{prefix}/manifest.json"].decode("utf-8"))
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
            continue
        for record in validate_remote_files(manifest, f"Dataset {prefix}"):
            key = record["destination"].casefold()
            if key in destinations:
                raise ValueError("Remote dataset destinations must be unique across the selected project.")
            destinations.add(key)
            result.append(record)
    return result


def _digest(path: Path) -> tuple[int, str]:
    size = 0
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _destination(root: Path, relative: str) -> Path:
    path = _relative(relative, "Remote file destination")
    current = root
    for part in path.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Acquired data path traverses a symbolic link: {relative!r}.")
    result = root.joinpath(*path.parts)
    if result.is_symlink():
        raise ValueError(f"Acquired data destination is a symbolic link: {relative!r}.")
    return result


def acquire(records: list[dict[str, Any]], data_root: Path) -> None:
    """Stream public immutable HF files to verified sibling temporaries."""
    data_root.mkdir(parents=True, exist_ok=True)
    if data_root.is_symlink() or not data_root.is_dir():
        raise ValueError("Acquired data root must be a regular directory.")
    for record in records:
        destination = _destination(data_root, record["destination"])
        if destination.is_file():
            size, digest = _digest(destination)
            if size == record["size"] and digest == record["sha256"]:
                continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.parent.is_symlink():
            raise ValueError(f"Acquired data directory is a symbolic link: {record['destination']!r}.")
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.download")
        repo = urllib.parse.quote(record["repo"], safe="/")
        revision = urllib.parse.quote(record["revision"], safe="")
        file_path = urllib.parse.quote(record["path"], safe="/")
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{file_path}?download=true"
        size = 0
        digest = hashlib.sha256()
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "nnmodelling-dataset-acquirer/1"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("xb") as output:
                if urllib.parse.urlsplit(response.geturl()).scheme != "https":
                    raise ValueError("Hugging Face redirected the dataset download away from HTTPS.")
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > record["size"]:
                        raise ValueError(f"Downloaded file exceeds declared size: {record['path']}.")
                    output.write(block)
                    digest.update(block)
                output.flush()
                os.fsync(output.fileno())
            if size != record["size"] or digest.hexdigest() != record["sha256"]:
                raise ValueError(f"Downloaded file failed size or SHA256 verification: {record['path']}.")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def records_from_snapshot(snapshot: Path) -> list[dict[str, Any]]:
    project = json.loads((snapshot / "project.json").read_text(encoding="utf-8"))
    root = snapshot / "files"
    result = []
    destinations: set[str] = set()
    project_manifest = project.get("manifest", {})
    active = project_manifest.get("activeDataset") if isinstance(project_manifest, dict) else None
    for ref in project_manifest.get("customDatasets", []):
        if not isinstance(active, dict) or (ref.get("id"), ref.get("version")) != (active.get("id"), active.get("version")):
            continue
        resource = root / _relative(ref["path"], "Dataset resource path")
        manifest_path = resource / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError(f"Dataset manifest is missing or linked: {ref['path']}.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in validate_remote_files(manifest, f"Dataset {ref['path']}"):
            key = record["destination"].casefold()
            if key in destinations:
                raise ValueError("Remote dataset destinations must be unique across the selected project.")
            destinations.add(key)
            result.append(record)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Acquire immutable public Hugging Face dataset files.")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    records = records_from_snapshot(args.snapshot)
    acquire(records, args.data_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
