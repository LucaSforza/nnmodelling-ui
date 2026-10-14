from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import select
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import tempfile
import time
import tomllib
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from .user_config import ProfileError, load_profile
from .slurm import _ALIAS, _MEMORY, _PARTITION, _TIME, _safe_remote_path

_CHUNK = 1024 * 1024
_MAX_DIAGNOSTIC = 2000
_MAX_CAPTURE = 64 * 1024
_SAFE_JOB = re.compile(r"^[0-9]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OMIT = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
_BASE_DEPENDENCIES = {"torch>=2.2", "safetensors>=0.4", "numpy>=1.26"}


class PreparationError(RuntimeError):
    pass


def _diagnostic(value: str) -> str:
    value = re.sub(r"(?i)((?:https?|git\+https?)://)[^\s/@]+@", r"\1[redacted]@", value)
    value = re.sub(r"(?i)(bearer\s+)[^\s]+", r"\1[redacted]", value)
    value = re.sub(r"(?i)(token|password|secret)(\s*[=:]\s*)[^\s]+", r"\1\2[redacted]", value)
    value = "".join(char if char in "\n\t" or ord(char) >= 32 else " " for char in value)
    return value[-_MAX_DIAGNOSTIC:]


def _capture_tail(pipe: BinaryIO, target: bytearray) -> None:
    while chunk := pipe.read(8192):
        target.extend(chunk)
        if len(target) > _MAX_CAPTURE:
            del target[:len(target) - _MAX_CAPTURE]


def _safe_project_path(value: str) -> PurePosixPath:
    if not value or "\\" in value or "\x00" in value:
        raise PreparationError("Project resource paths must be relative POSIX paths.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise PreparationError("Project resource paths must stay inside the project directory.")
    return path


def _resource_dependencies(project_dir: Path | None) -> list[str]:
    if project_dir is None:
        return []
    model_path = project_dir / "model.json"
    try:
        model = json.loads(model_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        raise PreparationError("Could not read project model metadata for dependency preparation.") from None
    if not isinstance(model, dict) or not isinstance(model.get("manifest"), dict):
        raise PreparationError("Project model metadata is malformed.")
    manifest = model["manifest"]
    packages = manifest.get("customPackages", [])
    datasets = manifest.get("customDatasets", [])
    if not isinstance(packages, list) or not isinstance(datasets, list):
        raise PreparationError("Project resource declarations are malformed.")
    references = packages + datasets
    directories: list[Path] = []
    for reference in references:
        if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
            raise PreparationError("Project resource declaration has no valid path.")
        relative = _safe_project_path(reference["path"])
        current = project_dir
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise PreparationError("Project resource path traverses a symlink.")
        directory = current.resolve()
        if not directory.is_relative_to(project_dir.resolve()) or not directory.is_dir():
            raise PreparationError("Project resource path escapes the project directory.")
        pyproject = directory / "pyproject.toml"
        if pyproject.is_symlink():
            raise PreparationError("Project resource dependency metadata cannot be a symlink.")
        if pyproject.is_file():
            directories.append(directory)
    if not directories:
        return []
    # Reuse the wheel's dependency parser so preparation rejects the same local,
    # authenticated, and unsupported requirements as inference export.
    try:
        from nnmodelling_runtime.wheel import _declared_python_dependencies

        dependencies = _declared_python_dependencies(directories)
    except (OSError, ValueError, tomllib.TOMLDecodeError):
        raise PreparationError("Project Python dependencies are malformed or use an unsupported source.") from None
    return [item for item in dependencies if item not in _BASE_DEPENDENCIES]


def _copy_tree(source: Path, destination: Path, *, suffixes: set[str]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.rglob("*"):
        relative = item.relative_to(source)
        if any(part in _OMIT or part.startswith(".env") for part in relative.parts):
            continue
        if item.is_symlink():
            raise PreparationError(f"Build source contains a symlink: {relative.as_posix()}.")
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file() and (item.suffix in suffixes or item.name == "pyproject.toml"):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, target)


def _build_context(repository: Path, destination: Path, dependencies: list[str]) -> Path:
    context = destination / "context"
    context.mkdir()
    for name in ("pyproject.toml", "uv.lock"):
        source = repository / name
        if not source.is_file() or source.is_symlink():
            raise PreparationError(f"Workspace build input {name} is missing or unsafe.")
        shutil.copyfile(source, context / name)
    shutil.copyfile(repository / ".dockerignore", context / ".dockerignore")
    shutil.copyfile(repository / "backend/Dockerfile", context / "Dockerfile")
    _copy_tree(repository / "backend", context / "backend", suffixes={".py"})
    _copy_tree(repository / "python/nnmodelling-runtime", context / "python/nnmodelling-runtime", suffixes={".py"})
    if dependencies:
        text = (context / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r"(?m)^examples\s*=\s*\[([^\]]*)\]", text)
        if not match:
            raise PreparationError("Workspace examples dependency group is missing.")
        values = [line.strip().strip(",").strip().strip('"') for line in match.group(1).splitlines() if line.strip()]
        # The checked-in group is one line today; parse it as TOML for exact values.
        try:
            groups = tomllib.loads("[dependency-groups]\n" + match.group(0))["dependency-groups"]
            values = groups["examples"]
        except (tomllib.TOMLDecodeError, KeyError):
            raise PreparationError("Workspace examples dependency group is malformed.") from None
        additions = sorted(set(dependencies) - set(values))
        updated = values + additions
        replacement = "examples = [" + ", ".join(json.dumps(value) for value in updated) + "]"
        text = text[:match.start()] + replacement + text[match.end():]
        (context / "pyproject.toml").write_text(text, encoding="utf-8")
        # The copied workspace lock is the resolver seed; the derived lock is
        # frozen before Docker builds and never resolves in the worker.
        return context
    return context


def _source_archive(context: Path, target: Path) -> Path:
    with tarfile.open(target, "w:gz") as archive:
        for relative in (Path("backend"), Path("python/nnmodelling-runtime/src")):
            root = context / relative
            for item in sorted(root.rglob("*")):
                if item.is_symlink():
                    raise PreparationError("Worker source contains a symlink.")
                if item.is_file() and item.suffix == ".py":
                    archive.add(item, arcname=(relative / item.relative_to(root)).as_posix(), recursive=False)
    return target


class WorkerPreparation:
    def __init__(self, profile: dict[str, dict[str, Any]], *, repository: Path | None = None,
                 project: Path | None = None, check_only: bool = False) -> None:
        self.profile = profile
        self.repository = (repository or Path(__file__).resolve().parents[1]).resolve()
        self.project = project.resolve() if project else None
        self.project_dependencies: list[str] = []
        self.check_only = check_only
        self.slurm = profile["slurm"]
        self.remote_stage = ""
        self.scheduler_id: str | None = None
        self.submission_pending = False
        self.cancel_error: str | None = None
        self.cancel_requested_id: str | None = None
        self.preserved_receipt: Path | None = None
        self.cancelled = False
        self.process: subprocess.Popen | None = None
        self._process_ignores_cancel = False
        self.local_stage: Path | None = None
        self._old_handlers: dict[int, Any] = {}

    def emit(self, phase: str, message: str = "", *, current: int = 0, total: int = 0,
             image: str | None = None, error: str | None = None) -> None:
        event: dict[str, Any] = {"phase": phase, "message": _diagnostic(message), "current": current, "total": total}
        if image is not None:
            event["image"] = image
        if error is not None:
            event["error"] = _diagnostic(error)
        sys.stdout.write(json.dumps(event, ensure_ascii=True, separators=(",", ":")) + "\n")
        sys.stdout.flush()

    def cancel(self) -> None:
        self.cancelled = True
        process = self.process
        if process is not None and not self._process_ignores_cancel and process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass

    def _cancel_recorded_allocation(self) -> None:
        self.cancel_error = None
        scheduler_id = self.scheduler_id
        # Submission runs detached so cancelling its SSH client cannot lose an
        # allocation accepted just before the receipt is written.
        for _ in range(30 if scheduler_id is None and self.submission_pending else 1):
            scheduler_id = scheduler_id or self._read_scheduler_receipt()
            if scheduler_id:
                break
            time.sleep(1)
        if scheduler_id and _SAFE_JOB.fullmatch(scheduler_id):
            if scheduler_id != self.cancel_requested_id:
                try:
                    self._ssh(f"scancel {scheduler_id}", timeout=15, ignore_cancel=True)
                    self.cancel_requested_id = scheduler_id
                except PreparationError as error:
                    self.cancel_error = _diagnostic(str(error)) or "scancel failed."
            self.scheduler_id = scheduler_id
            if self.local_stage and self.local_stage.is_dir():
                (self.local_stage / "scheduler.id").write_text(scheduler_id + "\n", encoding="ascii")

    def _preserve_scheduler_receipt(self) -> Path | None:
        if not self.scheduler_id:
            return None
        try:
            fd, name = tempfile.mkstemp(prefix="nnmodelling-worker-receipt-")
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="ascii") as receipt:
                receipt.write(self.scheduler_id + "\n")
            self.preserved_receipt = Path(name)
            return self.preserved_receipt
        except OSError:
            return None

    def _read_scheduler_receipt(self) -> str | None:
        if not self.remote_stage:
            return None
        try:
            receipt = shlex.quote(self.remote_stage + "/scheduler.id")
            result = self._ssh(f"test ! -f {receipt} || cat -- {receipt}", timeout=10, ignore_cancel=True)
        except PreparationError:
            return None
        value = result.strip().splitlines()[-1] if result.strip() else ""
        return value if _SAFE_JOB.fullmatch(value) else None

    def _install_signals(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                self._old_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, lambda *_args: self.cancel())
            except ValueError:  # Signal handlers can only be installed on the main thread.
                self._old_handlers.clear()
                return

    def _restore_signals(self) -> None:
        for sig, handler in self._old_handlers.items():
            signal.signal(sig, handler)
        self._old_handlers.clear()

    def run(self) -> str | None:
        self._install_signals()
        try:
            self._validate_profile()
            self._prerequisites()
            if not self.check_only and self.slurm["gpus"]:
                raise PreparationError("The frozen workspace lock contains CPU-only PyTorch; prepare with GPUs set to 0.")
            with tempfile.TemporaryDirectory(prefix="nnmodelling-worker-prepare-") as temporary:
                self.local_stage = Path(temporary)
                dependencies = _resource_dependencies(self.project)
                self.project_dependencies = dependencies
                if self.check_only:
                    context = self.repository
                else:
                    context = _build_context(self.repository, self.local_stage, dependencies)
                    if dependencies:
                        self.emit("dependencies", "Deriving a frozen preparation lock from the workspace lock.")
                        self._command(["uv", "lock", "--project", str(context)], timeout=1800)
                source_archive = _source_archive(context, self.local_stage / "worker-source.tar.gz")
                self.remote_stage = str(PurePosixPath(self.slurm["root"]) / (".nnmodelling-worker-" + uuid.uuid4().hex))
                self._prepare_remote_stage()
                self._stream_upload(source_archive, self.remote_stage + "/worker-source.tar.gz", "source")
                if self.check_only:
                    self._probe_configured()
                    self._cleanup_remote_stage()
                    self.emit("success", "Configured worker dependencies and startup are valid.")
                    return None
                self._build_and_convert(context)
                return self._final_image
        except Exception as error:
            if self.cancelled:
                self._cancel_recorded_allocation()
                message = "Worker preparation cancelled."
                if self.cancel_error and self.scheduler_id:
                    receipt = self._preserve_scheduler_receipt()
                    saved = f" Receipt saved to {receipt}." if receipt else f" Receipt remains on the cluster at {self.remote_stage}/scheduler.id."
                    message += f" Slurm allocation {self.scheduler_id} may still be active; cancellation failed: {self.cancel_error}.{saved}"
                elif self.cancel_requested_id:
                    message += f" Cancellation requested for owned allocation {self.cancel_requested_id}."
            else:
                message = _diagnostic(str(error)) or "Worker preparation failed."
                if self.scheduler_id or self.submission_pending:
                    self._cancel_recorded_allocation()
                    if self.scheduler_id:
                        if self.cancel_error:
                            receipt = self._preserve_scheduler_receipt()
                            saved = f" Receipt saved to {receipt}." if receipt else f" Receipt remains on the cluster at {self.remote_stage}/scheduler.id."
                            message += f" Owned allocation {self.scheduler_id} may still be active; scancel failed: {self.cancel_error}.{saved}"
                        else:
                            message += f" Cancellation requested for owned allocation {self.scheduler_id}."
                    elif self.submission_pending:
                        message += f" A Slurm allocation may still be active; no receipt was recovered from {self.remote_stage}."
            self.emit("error", message, error=message)
            raise PreparationError(message) from None
        finally:
            self._restore_signals()

    def _validate_profile(self) -> None:
        if self.profile["backend"]["executor"] != "slurm":
            raise PreparationError("Worker preparation requires the SSH / Slurm executor.")
        s = self.slurm
        if not isinstance(s.get("host"), str) or not _ALIAS.fullmatch(s["host"]):
            raise PreparationError("Slurm SSH host must be an alias or hostname.")
        for key in ("root", "image"):
            if not isinstance(s.get(key), str) or not _safe_remote_path(s[key]):
                raise PreparationError(f"Slurm {key} must be an absolute safe remote path.")
        if not isinstance(s.get("partition"), str) or not _PARTITION.fullmatch(s["partition"]):
            raise PreparationError("Slurm partition contains unsupported characters.")
        if type(s.get("cpus")) is not int or not 1 <= s["cpus"] <= 256:
            raise PreparationError("Slurm CPUs must be between 1 and 256.")
        if type(s.get("gpus")) is not int or not 0 <= s["gpus"] <= 8:
            raise PreparationError("Slurm GPUs must be between 0 and 8.")
        if not isinstance(s.get("memory"), str) or not _MEMORY.fullmatch(s["memory"]):
            raise PreparationError("Slurm memory must use a value such as 4G.")
        if not isinstance(s.get("time"), str) or not _TIME.fullmatch(s["time"]):
            raise PreparationError("Slurm time must use HH:MM:SS or D-HH:MM:SS.")
        ssh = s.get("ssh_executable")
        if not isinstance(ssh, str) or not ssh or (not Path(ssh).is_absolute() and Path(ssh).name != ssh) or ".." in Path(ssh).parts:
            raise PreparationError("SSH executable must name a PATH command or an absolute path.")
        if self.project and not self.project.is_dir():
            raise PreparationError("Selected project directory does not exist.")

    def _prerequisites(self) -> None:
        runtime = str(self.profile["backend"]["container_runtime"])
        runtime_path = True if self.check_only else shutil.which(runtime)
        ssh_path = shutil.which(self.slurm["ssh_executable"])
        if runtime_path is None:
            raise PreparationError(f"Docker-compatible runtime {runtime!r} was not found on PATH.")
        if ssh_path is None:
            raise PreparationError(f"SSH executable {self.slurm['ssh_executable']!r} was not found on PATH.")
        if not self.check_only and shutil.which("uv") is None:
            raise PreparationError("uv is required to prepare the worker image.")
        self.emit("checking", "Checking local tools and configured Slurm host.")
        image_parent = str(PurePosixPath(self.slurm["image"]).parent)
        checks = [
            f"test -d {shlex.quote(image_parent)}",
            f"test \"$(realpath -e -- {shlex.quote(self.slurm['root'])})\" = {shlex.quote(self.slurm['root'])}",
            f"test \"$(realpath -e -- {shlex.quote(image_parent)})\" = {shlex.quote(image_parent)}",
            "command -v sbatch squeue sacct scancel singularity python3 sha256sum realpath >/dev/null",
            f"sinfo -h -p {shlex.quote(self.slurm['partition'])} | grep -q .",
        ]
        if not self.check_only:
            checks[0:0] = [
                f"mkdir -m 700 -p -- {shlex.quote(self.slurm['root'])}",
                f"test -w {shlex.quote(image_parent)}",
            ]
        command = " && ".join(checks)
        self._ssh(command, timeout=45)
        if self.check_only:
            check = (f"test -f {shlex.quote(self.slurm['image'])} && test -r {shlex.quote(self.slurm['image'])} "
                     f"&& test ! -L {shlex.quote(self.slurm['image'])}")
            self._ssh(check, timeout=20)

    def _ssh_args(self, command: str) -> list[str]:
        return [self.slurm["ssh_executable"], "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                self.slurm["host"], command]

    def _command(self, args: list[str], *, timeout: float) -> str:
        return self._run_process(args, timeout=timeout)

    def _ssh(self, command: str, *, timeout: float = 60, input_data: bytes | None = None,
             ignore_cancel: bool = False) -> str:
        return self._run_process(self._ssh_args(command), timeout=timeout, input_data=input_data,
                                 ignore_cancel=ignore_cancel)

    def _run_process(self, args: list[str], *, timeout: float, input_data: bytes | None = None,
                     ignore_cancel: bool = False) -> str:
        process = subprocess.Popen(args, stdin=subprocess.PIPE if input_data is not None else subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self.process = process
        self._process_ignores_cancel = ignore_cancel
        assert process.stdout is not None and process.stderr is not None
        stdout_tail, stderr_tail = bytearray(), bytearray()
        readers = [threading.Thread(target=_capture_tail, args=(pipe, target), daemon=True)
                   for pipe, target in ((process.stdout, stdout_tail), (process.stderr, stderr_tail))]
        for reader in readers:
            reader.start()
        try:
            if input_data is not None and process.stdin:
                process.stdin.write(input_data)
                process.stdin.close()
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                if self.cancelled and not ignore_cancel:
                    raise PreparationError("Worker preparation cancelled.")
                if time.monotonic() >= deadline:
                    raise PreparationError(f"Command timed out after {int(timeout)} seconds: {args[0]}.")
                time.sleep(0.1)
            if self.cancelled and not ignore_cancel:
                raise PreparationError("Worker preparation cancelled.")
            status = process.returncode
        finally:
            if process.poll() is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for reader in readers:
                reader.join()
            if self.process is process:
                self.process = None
                self._process_ignores_cancel = False
        stdout = bytes(stdout_tail).decode("utf-8", errors="replace")
        stderr = bytes(stderr_tail).decode("utf-8", errors="replace")
        if status:
            detail = _diagnostic(stderr or stdout)
            raise PreparationError(detail or f"Command failed with exit code {status}: {args[0]}.")
        return stdout

    def _prepare_remote_stage(self) -> None:
        self.emit("staging", "Creating a private temporary directory on the cluster.")
        self._ssh(f"umask 077 && mkdir -- {shlex.quote(self.remote_stage)} "
                  f"&& mkdir -m 700 -- {shlex.quote(self.remote_stage + '/source')}")

    def _cleanup_remote_stage(self) -> None:
        root = shlex.quote(self.slurm["root"])
        stage = shlex.quote(self.remote_stage)
        command = (f"test ! -L {stage} && test -d {stage} && "
                   f"case {stage} in {root}/.nnmodelling-worker-*) rm -rf -- {stage};; *) exit 1;; esac")
        try:
            self._ssh(command, timeout=30)
        except PreparationError as error:
            self.emit("cleanup", "Could not remove the completed temporary worker stage.", error=str(error))

    def _stream_upload(self, source: Path, destination: str, label: str) -> tuple[int, str]:
        size = source.stat().st_size
        digest = hashlib.sha256()
        self.emit("transfer", f"Transferring {label} to the cluster.", current=0, total=size)
        remote = f"umask 077 && cat > {shlex.quote(destination + '.tmp')} && mv -- {shlex.quote(destination + '.tmp')} {shlex.quote(destination)}"
        with tempfile.TemporaryFile() as errors, source.open("rb") as stream:
            process = subprocess.Popen(self._ssh_args(remote), stdin=subprocess.PIPE,
                                       stdout=subprocess.DEVNULL, stderr=errors)
            self.process = process
            sent = 0
            assert process.stdin is not None
            fd = process.stdin.fileno()
            os.set_blocking(fd, False)
            deadline = time.monotonic() + 6 * 60 * 60
            try:
                while chunk := stream.read(_CHUNK):
                    if self.cancelled:
                        raise PreparationError("Worker preparation cancelled.")
                    digest.update(chunk)
                    pending = memoryview(chunk)
                    while pending:
                        if self.cancelled:
                            raise PreparationError("Worker preparation cancelled.")
                        if time.monotonic() >= deadline:
                            raise PreparationError(f"Timed out transferring {label} to the cluster.")
                        _, writable, _ = select.select([], [fd], [], 1)
                        if not writable:
                            continue
                        written = os.write(fd, pending)
                        pending = pending[written:]
                        sent += written
                        self.emit("transfer", f"Transferring {label} to the cluster.", current=sent, total=size)
                process.stdin.close()
                status = process.wait(timeout=max(1, deadline - time.monotonic()))
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired, PreparationError):
                if process.poll() is None:
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise
            finally:
                if self.process is process:
                    self.process = None
            if status:
                errors.seek(0, os.SEEK_END)
                size_errors = errors.tell()
                errors.seek(max(0, size_errors - _MAX_DIAGNOSTIC))
                raise PreparationError(_diagnostic(errors.read().decode("utf-8", errors="replace")) or "Cluster transfer failed.")
        return size, digest.hexdigest()

    def _probe_command(self, image: str, source: str, *, legacy: bool = False) -> str:
        python = ("sh -c 'if [ -x /opt/nnmodelling/venv/bin/python ]; then "
                  "exec /opt/nnmodelling/venv/bin/python \"$@\"; "
                  "else exec python3 \"$@\"; fi' sh" if legacy else "/opt/nnmodelling/venv/bin/python")
        singularity = (f"singularity exec --cleanenv --containall --no-home --net --network none "
                       f"--bind {shlex.quote(source + ':/app')} --pwd /app "
                       f"--env PYTHONPATH=/app:/app/python/nnmodelling-runtime/src "
                       f"{shlex.quote(image)} {python}")
        check = self._probe_code()
        if self.slurm["gpus"] and image == self.slurm["image"]:
            check += "; assert torch.version.cuda, 'configured GPU worker contains CPU-only PyTorch'"
        return f"{singularity} -c {shlex.quote(check)} && {singularity} -m backend.worker --help >/dev/null"

    def _probe_code(self) -> str:
        external = sorted({re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", item).group(0)
                           for item in self.project_dependencies})
        return f"""import importlib.util, importlib.metadata, torch
required = ('numpy', 'safetensors', 'backend.worker', 'nnmodelling_runtime')
missing = [name for name in required if importlib.util.find_spec(name) is None]
declared = {external!r}
missing_distributions = []
for name in declared:
    try:
        importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        missing_distributions.append(name)
assert not missing, 'missing worker modules: ' + ','.join(missing)
assert not missing_distributions, 'missing project dependencies: ' + ','.join(missing_distributions)
"""

    def _probe_configured(self) -> None:
        self.emit("checking", "Probing dependencies and worker startup in the training mount layout.")
        self._ssh(f"tar -xzf {shlex.quote(self.remote_stage + '/worker-source.tar.gz')} "
                  f"-C {shlex.quote(self.remote_stage + '/source')}", timeout=120)
        self._ssh(self._probe_command(self.slurm["image"], self.remote_stage + "/source", legacy=True), timeout=180)

    def _build_and_convert(self, context: Path) -> None:
        runtime = str(self.profile["backend"]["container_runtime"])
        runtime_path = shutil.which(runtime) or runtime
        tag = "nnmodelling-worker:prepare-" + uuid.uuid4().hex
        image_archive = self.local_stage / "worker-image.tar"
        self.emit("building", "Building the worker from the frozen workspace lock.")
        version = self._command([runtime_path, "--version"], timeout=30).lower()
        self._command([runtime_path, "build", "--file", str(context / "Dockerfile"), "--tag", tag,
                       str(context)], timeout=4 * 60 * 60)
        save = [runtime_path, "save"]
        if "podman" in version:
            save.extend(["--format", "docker-archive"])
        save.extend(["--output", str(image_archive), tag])
        self._command(save, timeout=2 * 60 * 60)
        try:
            self._stream_upload(image_archive, self.remote_stage + "/worker-image.tar", "worker image")
            self._verify_remote_archive(image_archive)
            self._submit_conversion(image_archive.stat().st_size)
            sha = self._ssh(f"cat -- {shlex.quote(self.remote_stage + '/result.sha256')}", timeout=30).strip()
            if not _SHA256.fullmatch(sha):
                raise PreparationError("Cluster returned an invalid SIF SHA256 receipt.")
            target = str(PurePosixPath(self.slurm["image"]).parent / f"worker-{sha}.sif")
            self._final_image = target
            self._cleanup_remote_stage()
            self.emit("success", "Worker image converted, checked, and installed.", image=target)
        finally:
            try:
                self._command([runtime_path, "image", "rm", tag], timeout=60)
            except PreparationError:
                pass

    def _verify_remote_archive(self, archive: Path) -> None:
        local_hash = hashlib.sha256()
        with archive.open("rb") as stream:
            for chunk in iter(lambda: stream.read(_CHUNK), b""):
                local_hash.update(chunk)
        remote = self._ssh(f"sha256sum -- {shlex.quote(self.remote_stage + '/worker-image.tar')}", timeout=60)
        remote_hash = remote.split()[0] if remote.split() else ""
        if remote_hash != local_hash.hexdigest():
            raise PreparationError("Transferred worker archive SHA256 does not match the local archive.")
        self._ssh(f"printf '%s\\n' {shlex.quote(remote_hash)} > {shlex.quote(self.remote_stage + '/archive.sha256')}")

    def _submit_conversion(self, archive_size: int) -> None:
        stage = self.remote_stage
        parent = str(PurePosixPath(self.slurm["image"]).parent)
        script_path = stage + "/convert.sbatch"
        sif_temp = stage + "/worker.tmp.sif"
        expected_arch = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}.get(platform.machine(), platform.machine())
        free_needed = max(archive_size * 2, 4 * 1024 * 1024 * 1024)
        probe = self._probe_code()
        script = f"""#!/bin/bash
#SBATCH --partition={self.slurm['partition']}
#SBATCH --time={self.slurm['time']}
#SBATCH --mem={self.slurm['memory']}
#SBATCH --cpus-per-task={self.slurm['cpus']}
#SBATCH --output={stage}/convert-%j.log
set -euo pipefail
stage={shlex.quote(stage)}
parent={shlex.quote(parent)}
archive={shlex.quote(stage + '/worker-image.tar')}
sif={shlex.quote(sif_temp)}
source={shlex.quote(stage + '/source')}
publish_temp=""
cleanup() {{ rm -f -- "$sif" "$sif.sha256"; [ -z "$publish_temp" ] || rm -f -- "$publish_temp"; }}
trap cleanup EXIT
test ! -L "$stage" && test -d "$stage"
test ! -L "$parent" && test -d "$parent" && test -w "$parent"
test "$(realpath -e -- "$parent")" = "$parent"
test "$(uname -m)" = {shlex.quote(expected_arch)}
available=$(df -Pk -- "$parent" | awk 'NR==2 {{printf "%.0f", $4 * 1024}}')
test "$available" -ge {free_needed} || {{ echo 'insufficient disk space for worker SIF conversion' >&2; exit 70; }}
test "$(sha256sum -- "$archive" | awk '{{print $1}}')" = "$(cat -- "$stage/archive.sha256")"
singularity build "$sif" "docker-archive:$archive"
tar -xzf "$stage/worker-source.tar.gz" -C "$source"
singularity exec --cleanenv --containall --no-home --net --network none --bind "$source:/app" --pwd /app --env PYTHONPATH=/app:/app/python/nnmodelling-runtime/src "$sif" /opt/nnmodelling/venv/bin/python -c {shlex.quote(probe)}
singularity exec --cleanenv --containall --no-home --net --network none --bind "$source:/app" --pwd /app --env PYTHONPATH=/app:/app/python/nnmodelling-runtime/src "$sif" /opt/nnmodelling/venv/bin/python -m backend.worker --help >/dev/null
sha=$(sha256sum -- "$sif" | awk '{{print $1}}')
[[ "$sha" =~ ^[0-9a-f]{{64}}$ ]]
target="$parent/worker-$sha.sif"
if [ -e "$target" ] || [ -L "$target" ]; then
  test ! -L "$target" && test "$(sha256sum -- "$target" | awk '{{print $1}}')" = "$sha"
else
  publish_temp="$parent/.worker-$sha.sif.tmp-{uuid.uuid4().hex}"
  cp -- "$sif" "$publish_temp"
  chmod 0644 "$publish_temp"
  test "$(sha256sum -- "$publish_temp" | awk '{{print $1}}')" = "$sha"
  mv -n -- "$publish_temp" "$target"
  test -f "$target" && test ! -L "$target"
  test "$(sha256sum -- "$target" | awk '{{print $1}}')" = "$sha"
fi
printf '%s\\n' "$sha" > "$stage/result.sha256.tmp"
mv -- "$stage/result.sha256.tmp" "$stage/result.sha256"
"""
        self._ssh(f"umask 077 && cat > {shlex.quote(script_path)} && chmod 700 {shlex.quote(script_path)}", timeout=30,
                   input_data=script.encode())
        submit_script = f"""#!/bin/bash
set -euo pipefail
id=$(sbatch --parsable {shlex.quote(script_path)})
id=${{id%%;*}}
[[ $id =~ ^[0-9]+$ ]]
printf '%s\\n' "$id" > {shlex.quote(stage + '/scheduler.id.tmp')}
mv -- {shlex.quote(stage + '/scheduler.id.tmp')} {shlex.quote(stage + '/scheduler.id')}
"""
        submit_path = stage + "/submit.sh"
        self._ssh(f"umask 077 && cat > {shlex.quote(submit_path)} && chmod 700 {shlex.quote(submit_path)}",
                  timeout=30, input_data=submit_script.encode())
        submit = f"nohup /bin/bash {shlex.quote(submit_path)} > {shlex.quote(stage + '/submit.log')} 2>&1 </dev/null &"
        self.emit("submitting", "Submitting the owned CPU SIF conversion allocation.")
        self.submission_pending = True
        try:
            self._ssh(submit, timeout=30)
        except PreparationError:
            self._cancel_recorded_allocation()
            raise
        deadline = time.monotonic() + 60
        job_id = ""
        while time.monotonic() < deadline:
            if self.cancelled:
                self._cancel_recorded_allocation()
                raise PreparationError("Worker preparation cancelled.")
            receipt = shlex.quote(stage + "/scheduler.id")
            scheduler = self._ssh(f"test ! -f {receipt} || cat -- {receipt}", timeout=15).strip().splitlines()
            job_id = scheduler[-1].strip() if scheduler else ""
            if job_id:
                break
            time.sleep(1)
        if not _SAFE_JOB.fullmatch(job_id):
            detail = self._ssh(f"cat -- {shlex.quote(stage + '/submit.log')} 2>/dev/null || true", timeout=10).strip()
            raise PreparationError(_diagnostic(detail) or "Timed out waiting for Slurm to record the preparation allocation.")
        self.scheduler_id = job_id
        (self.local_stage / "scheduler.id").write_text(job_id + "\n", encoding="ascii")
        self.submission_pending = False
        if self.cancelled:
            self._cancel_recorded_allocation()
            raise PreparationError("Worker preparation cancelled.")
        self._wait_for_conversion(job_id)

    def _wait_for_conversion(self, job_id: str) -> None:
        limit = min(_walltime_seconds(self.slurm["time"]) + 600, 7 * 24 * 60 * 60)
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            if self.cancelled:
                self._cancel_recorded_allocation()
                raise PreparationError("Worker preparation cancelled.")
            try:
                queue = self._ssh(f"squeue -h -j {job_id} -o %T", timeout=30).strip()
            except PreparationError as error:
                raise PreparationError(f"Lost SSH while monitoring owned allocation {job_id}: {_diagnostic(str(error))}") from None
            if queue:
                self.emit("converting", f"SIF conversion allocation {job_id} is {queue.lower()}.")
            else:
                try:
                    result = self._ssh(f"sacct -n -X -j {job_id} --format=State,ExitCode", timeout=30).strip().splitlines()
                except PreparationError as error:
                    raise PreparationError(f"Lost SSH while checking owned allocation {job_id}: {_diagnostic(str(error))}") from None
                state, exit_code = result[-1].split()[:2] if result else ("", "")
                state = state.split("+", 1)[0]
                if state == "COMPLETED" and exit_code == "0:0":
                    self.scheduler_id = None
                    (self.local_stage / "scheduler.id").unlink(missing_ok=True)
                    return
                if state:
                    self.scheduler_id = None
                    (self.local_stage / "scheduler.id").unlink(missing_ok=True)
                    log = self._ssh(f"cat -- {shlex.quote(self.remote_stage + '/convert-' + job_id + '.log')} 2>/dev/null || true",
                                    timeout=15).strip()
                    detail = f": {_diagnostic(log)}" if log else ""
                    raise PreparationError(f"SIF conversion allocation {job_id} ended in {state} ({exit_code or 'unknown exit'}){detail}.")
            time.sleep(5)
        self._cancel_recorded_allocation()
        raise PreparationError(f"Timed out waiting for SIF conversion allocation {job_id}.")


def _walltime_seconds(value: str) -> int:
    days = 0
    if "-" in value:
        day_text, value = value.split("-", 1)
        days = int(day_text)
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare or check a contained NNModelling Slurm worker.")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        _, profile = load_profile(args.profile)
        preparation = WorkerPreparation(profile, project=args.project, check_only=args.check)
        image = preparation.run()
        if not args.check:
            # run() emits success only after image verification; this line is
            # retained as a guard against accidentally returning no artifact.
            if not image:
                raise PreparationError("Worker preparation completed without an image path.")
        return 0
    except (ProfileError, PreparationError) as error:
        # WorkerPreparation has already emitted structured errors. Profile
        # loading happens before it is constructed, so report that failure here.
        if isinstance(error, ProfileError):
            event = {"phase": "error", "message": str(error), "current": 0, "total": 0, "error": str(error)}
            sys.stdout.write(json.dumps(event, separators=(",", ":")) + "\n")
            sys.stdout.flush()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
